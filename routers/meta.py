"""
routers/meta.py — Platform metadata endpoints.

Endpoints
─────────
GET  /api/sources              Distinct source_type values with doc counts from DB
GET  /api/health/services      Live health check for every platform dependency
GET  /api/search               Full-text document search across metadata + classification
"""

import time
import logging
from typing import Optional

from fastapi import APIRouter
from psycopg2.extras import RealDictCursor

from database import get_connection
from config import (
    SOURCE_CONN_STR, SOURCE_CONTAINER,
    LLM_AZURE_ENDPOINT, LLM_API_KEY, LLM_API_VERSION, LLM_DEPLOYMENT,
    EMBEDDING_MODEL,
)

log = logging.getLogger(__name__)
router = APIRouter(tags=["Meta"])


# ── GET /api/sources ──────────────────────────────────────────────────────────

@router.get("/sources")
def list_sources():
    """
    Return all distinct source_type values that actually exist in document_metadata,
    with their document counts (excluding Deleted records).
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT source_type,
                       COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE processing_status = 'Original')  AS originals,
                       COUNT(*) FILTER (WHERE processing_status = 'Duplicate') AS duplicates
                FROM   document_metadata
                WHERE  processing_status != 'Deleted'
                  AND  source_type IS NOT NULL
                GROUP  BY source_type
                ORDER  BY total DESC
                """
            )
            rows = [dict(r) for r in cur.fetchall()]
        return {"sources": rows}
    finally:
        conn.close()


# ── GET /api/health/services ─────────────────────────────────────────────────

def _check_postgres() -> dict:
    t = time.time()
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return {"status": "healthy", "latency_ms": round((time.time() - t) * 1000, 1)}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc), "latency_ms": None}


def _check_blob() -> dict:
    t = time.time()
    try:
        from azure.storage.blob import BlobServiceClient
        client = BlobServiceClient.from_connection_string(SOURCE_CONN_STR)
        # lightweight call — just get service properties
        props = client.get_service_properties()
        _ = props  # consumed
        return {"status": "healthy", "latency_ms": round((time.time() - t) * 1000, 1)}
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)[:120], "latency_ms": None}


def _check_openai() -> dict:
    t = time.time()
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=LLM_AZURE_ENDPOINT,
            api_key=LLM_API_KEY,
            api_version=LLM_API_VERSION,
        )
        # cheapest possible call — list deployments via models endpoint
        models = client.models.list()
        _ = list(models)[:1]
        return {"status": "healthy", "latency_ms": round((time.time() - t) * 1000, 1)}
    except Exception as exc:
        # 404 / auth errors still mean the endpoint is reachable
        err = str(exc)
        if "404" in err or "Resource not found" in err:
            return {"status": "healthy", "note": "endpoint reachable", "latency_ms": round((time.time() - t) * 1000, 1)}
        return {"status": "degraded", "error": err[:120], "latency_ms": round((time.time() - t) * 1000, 1)}


def _check_embeddings() -> dict:
    """Quick probe — send a tiny embedding request."""
    t = time.time()
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=LLM_AZURE_ENDPOINT,
            api_key=LLM_API_KEY,
            api_version=LLM_API_VERSION,
        )
        client.embeddings.create(input="ping", model=EMBEDDING_MODEL)
        return {"status": "healthy", "latency_ms": round((time.time() - t) * 1000, 1)}
    except Exception as exc:
        return {"status": "degraded", "error": str(exc)[:120], "latency_ms": round((time.time() - t) * 1000, 1)}


@router.get("/health/services")
def services_health():
    """
    Probe each platform dependency and return live status + latency.
    Runs checks sequentially to avoid overwhelming external services.
    """
    checks = {}

    checks["document_intelligence_api"] = {"status": "healthy", "note": "this service", "latency_ms": 0}

    checks["postgresql"] = _check_postgres()

    checks["azure_blob_storage"] = _check_blob()

    checks["azure_openai_llm"] = _check_openai()

    checks["azure_openai_embeddings"] = _check_embeddings()

    # Rule engine is always healthy if this endpoint is reachable
    checks["rule_engine"] = {"status": "healthy", "note": "in-process", "latency_ms": 0}

    overall = "healthy" if all(v["status"] == "healthy" for v in checks.values()) else "degraded"

    return {
        "overall": overall,
        "services": [
            {
                "name":       name,
                "status":     info["status"],
                "latency_ms": info.get("latency_ms"),
                "note":       info.get("note"),
                "error":      info.get("error"),
            }
            for name, info in checks.items()
        ],
    }


# ── GET /api/stats ───────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats():
    """
    Dashboard statistics: totals, per-source breakdown, and 30-day activity.
    Combines document_metadata counts with a daily time-series.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Overall counts
            cur.execute(
                """
                SELECT
                    COUNT(*)                                                  AS total,
                    COUNT(*) FILTER (WHERE processing_status = 'Original')   AS originals,
                    COUNT(*) FILTER (WHERE processing_status = 'Duplicate')  AS duplicates
                FROM document_metadata
                WHERE processing_status != 'Deleted'
                """
            )
            totals = dict(cur.fetchone())

            # Per-source breakdown
            cur.execute(
                """
                SELECT source_type,
                       COUNT(*) FILTER (WHERE processing_status = 'Original')  AS original,
                       COUNT(*) FILTER (WHERE processing_status = 'Duplicate') AS duplicate
                FROM   document_metadata
                WHERE  processing_status != 'Deleted'
                  AND  source_type IS NOT NULL
                GROUP  BY source_type
                """
            )
            by_source = {
                row["source_type"]: {
                    "original":  row["original"],
                    "duplicate": row["duplicate"],
                }
                for row in cur.fetchall()
            }

            # Daily ingestion for the last 30 days
            cur.execute(
                """
                SELECT DATE(current_timestamp_ist AT TIME ZONE 'Asia/Kolkata') AS date,
                       COUNT(*) AS count
                FROM   document_metadata
                WHERE  current_timestamp_ist >= NOW() - INTERVAL '30 days'
                GROUP  BY date
                ORDER  BY date
                """
            )
            by_day = [
                {"date": str(row["date"]), "count": row["count"]}
                for row in cur.fetchall()
            ]

        return {
            "total":      totals["total"],
            "originals":  totals["originals"],
            "duplicates": totals["duplicates"],
            "by_source":  by_source,
            "by_day":     by_day,
        }
    finally:
        conn.close()


# ── GET /api/search/metadata ─────────────────────────────────────────────────

@router.get("/search/metadata")
def search_documents(
    q:             Optional[str] = None,
    document_type: Optional[str] = None,
    source_type:   Optional[str] = None,
    limit:  int = 20,
    offset: int = 0,
):
    """
    Full-text search across document_metadata + classification_docs.

    Searches:
      - file_name ILIKE %q%  (filename match)
      - file_path ILIKE %q%  (path match)

    Optional filters: document_type, source_type.

    Returns documents with their latest classification info.
    """
    conn = get_connection()
    try:
        where = ["dm.processing_status != 'Deleted'"]
        params: list = []

        if q:
            where.append("(dm.file_name ILIKE %s OR dm.file_path ILIKE %s)")
            params += [f"%{q}%", f"%{q}%"]
        if document_type:
            where.append("cd.document_type = %s")
            params.append(document_type)
        if source_type:
            where.append("dm.source_type = %s")
            params.append(source_type)

        where_sql = " AND ".join(where)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM   document_metadata dm
                LEFT JOIN LATERAL (
                    SELECT document_type, classification_status,
                           confidence_score, type_of_classification
                    FROM   classification_docs
                    WHERE  document_id = dm.document_id
                    ORDER  BY current_timestamp_ist DESC LIMIT 1
                ) cd ON TRUE
                WHERE  {where_sql}
                """,
                params,
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT dm.document_id, dm.file_name, dm.file_path,
                       dm.source_type, dm.file_size, dm.processing_status,
                       dm.is_duplicate, dm.current_timestamp_ist,
                       cd.document_type, cd.classification_status,
                       cd.confidence_score, cd.type_of_classification
                FROM   document_metadata dm
                LEFT JOIN LATERAL (
                    SELECT document_type, classification_status,
                           confidence_score, type_of_classification
                    FROM   classification_docs
                    WHERE  document_id = dm.document_id
                    ORDER  BY current_timestamp_ist DESC LIMIT 1
                ) cd ON TRUE
                WHERE  {where_sql}
                ORDER  BY dm.current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params + [limit, offset],
            )
            results = [dict(r) for r in cur.fetchall()]

        return {
            "query":   q,
            "total":   total,
            "limit":   limit,
            "offset":  offset,
            "results": results,
        }
    finally:
        conn.close()
