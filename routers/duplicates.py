"""
routers/duplicates.py — All duplicate detection endpoints.

Endpoints
─────────
GET  /api/files                          list files (filter by is_duplicate)
GET  /api/files/all                      list ALL records (no filter)
GET  /api/files/summary                  counts by processing_status and source
GET  /api/files/duplicates-by-name       search duplicates by file_name
POST /api/ingest/check-duplicate         mark new ingestion as Original or Duplicate
POST /api/admin/scan-duplicates          bulk reclassify all existing records
DELETE /api/files/{document_id}          delete file from Blob + soft-delete in DB
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor

from database import get_connection
from models import IngestCheckRequest
from utils.blob import delete_source_blob, file_path_to_source_key

log = logging.getLogger(__name__)

router = APIRouter(tags=["Duplicates"])


# Lateral join to pull the latest classification row for each document
_CLASSIFY_JOIN = """
    LEFT JOIN LATERAL (
        SELECT document_type, classification_status, confidence_score,
               type_of_classification
        FROM   classification_docs
        WHERE  document_id = dm.document_id
        ORDER  BY current_timestamp_ist DESC
        LIMIT  1
    ) cd ON TRUE
"""

# ════════════════════════════════════════════════════════════════════
#  CORE BUSINESS LOGIC
# ════════════════════════════════════════════════════════════════════

def _check_and_mark_duplicate(conn, file_name: str, file_path: str) -> dict:
    """
    Called immediately after a new file row is inserted into document_metadata.

    Logic:
    • Search for any existing ORIGINAL or UNCLASSIFIED record with the same
      file_name (Duplicate records are excluded — a Duplicate cannot make
      another file a Duplicate).
    • If found → new file is a Duplicate.
    • If not found → new file is the Original.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT file_name, file_path, current_timestamp_ist, processing_status
            FROM   document_metadata
            WHERE  file_name         = %s
              AND  file_path        != %s
              AND  processing_status NOT IN ('Deleted', 'Duplicate')
            ORDER  BY current_timestamp_ist ASC
            LIMIT  1
            """,
            (file_name, file_path),
        )
        existing = cur.fetchone()

    if existing:
        log.info(
            f"Duplicate detected: '{file_name}' already exists at "
            f"'{existing['file_path']}'"
        )
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE document_metadata
                SET    is_duplicate      = TRUE,
                       duplicate_of      = %s,
                       processing_status = 'Duplicate'
                WHERE  file_path = %s
                """,
                (existing["file_path"], file_path),
            )
        conn.commit()
        return {"is_duplicate": True, "duplicate_of": existing["file_path"]}

    log.info(f"Original file: '{file_name}'")
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE document_metadata
            SET    is_duplicate      = FALSE,
                   duplicate_of      = NULL,
                   processing_status = 'Original'
            WHERE  file_path = %s
            """,
            (file_path,),
        )
    conn.commit()
    return {"is_duplicate": False, "duplicate_of": None}


def _delete_file(conn, document_id: str) -> dict:
    """
    Delete a file from Blob Storage and soft-delete its DB row.

    The row is kept in document_metadata with processing_status = 'Deleted'
    so the audit trail is preserved. The actual blob is removed.
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT document_id, file_name, file_path, processing_status
            FROM   document_metadata
            WHERE  document_id       = %s
              AND  processing_status != 'Deleted'
            """,
            (document_id,),
        )
        record = cur.fetchone()

    if not record:
        raise ValueError(
            f"File not found or already deleted (document_id={document_id})"
        )

    # Delete from Blob Storage
    delete_source_blob(record["file_path"])

    # Soft-delete in DB
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE document_metadata
            SET    processing_status = 'Deleted'
            WHERE  document_id = %s
            """,
            (document_id,),
        )
    conn.commit()
    log.info(f"Deleted: {record['file_name']} ({document_id})")

    return {
        "deleted":     True,
        "file_name":   record["file_name"],
        "file_path":   record["file_path"],
        "document_id": document_id,
    }


def _scan_all_existing_duplicates(conn) -> dict:
    """
    One-time bulk scan to reclassify all existing records as Original or Duplicate.

    Rules:
    • file_name appears more than once → oldest row = Original, others = Duplicate
    • file_name appears exactly once  → Original (single-occurrence files are also
      explicitly marked to keep summary counts correct)
    """
    total_originals  = 0
    total_duplicates = 0
    total_groups     = 0

    # Pass 1 — file_names that appear more than once
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT   file_name, COUNT(*) AS cnt
            FROM     document_metadata
            WHERE    processing_status != 'Deleted'
            GROUP BY file_name
            HAVING   COUNT(*) > 1
            """
        )
        duplicated_names = cur.fetchall()

    for row in duplicated_names:
        fname = row["file_name"]
        total_groups += 1

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT document_id, file_path, current_timestamp_ist
                FROM   document_metadata
                WHERE  file_name         = %s
                  AND  processing_status != 'Deleted'
                ORDER  BY current_timestamp_ist ASC
                """,
                (fname,),
            )
            records = cur.fetchall()

        original = records[0]
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE document_metadata "
                "SET is_duplicate=FALSE, duplicate_of=NULL, processing_status='Original' "
                "WHERE document_id=%s",
                (original["document_id"],),
            )
        total_originals += 1

        for dup in records[1:]:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE document_metadata "
                    "SET is_duplicate=TRUE, duplicate_of=%s, processing_status='Duplicate' "
                    "WHERE document_id=%s",
                    (original["file_path"], dup["document_id"]),
                )
            total_duplicates += 1

    # Pass 2 — single-occurrence files → mark as Original
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE document_metadata
            SET    is_duplicate      = FALSE,
                   duplicate_of      = NULL,
                   processing_status = 'Original'
            WHERE  processing_status NOT IN ('Original', 'Duplicate', 'Deleted')
            """
        )
        single_files_marked = cur.rowcount

    conn.commit()
    total_originals += single_files_marked

    log.info(
        f"Scan: {total_groups} group(s), {total_duplicates} duplicate(s), "
        f"{single_files_marked} single-file(s) marked as Original"
    )
    return {
        "duplicate_groups_found":   total_groups,
        "records_marked_duplicate": total_duplicates,
        "records_marked_original":  total_originals,
    }


# ════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════════

@router.get("/files")
def list_files(
    duplicates_only: bool = False,
    source_type:     Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
):
    """
    List files from document_metadata with filters.

    duplicates_only=true  → returns ONLY duplicate records  (is_duplicate=TRUE)
    duplicates_only=false → returns ONLY original records   (is_duplicate=FALSE)

    Use GET /api/files/all to retrieve all records without any is_duplicate filter.
    """
    conn = get_connection()
    try:
        # Always filter by is_duplicate based on the flag value
        where_clauses = [
            "processing_status != 'Deleted'",
            f"is_duplicate = {'TRUE' if duplicates_only else 'FALSE'}",
        ]
        params: list = []

        if source_type:
            where_clauses.append("source_type = %s")
            params.append(source_type)

        where_sql = " AND ".join(where_clauses)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT document_id, source_type, file_name, file_path,
                       processing_status, is_duplicate, duplicate_of,
                       current_timestamp_ist, file_size
                FROM   document_metadata
                WHERE  {where_sql}
                ORDER  BY current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params + [limit, offset],
            )
            files = [dict(r) for r in cur.fetchall()]

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM document_metadata WHERE {where_sql}",
                params,
            )
            total = cur.fetchone()[0]

        return {"total": total, "limit": limit, "offset": offset, "files": files}
    finally:
        conn.close()


# @router.get("/files/all")
# def list_all_files(
#     source_type: Optional[str] = None,
#     limit:  int = 100,
#     offset: int = 0,
# ):
#     """
#     Return ALL records from document_metadata without any is_duplicate filter.
#     Optionally filter by source_type.
#     """
#     conn = get_connection()
#     try:
#         where_clauses = ["processing_status != 'Deleted'"]
#         params: list = []

#         if source_type:
#             where_clauses.append("source_type = %s")
#             params.append(source_type)

#         where_sql = " AND ".join(where_clauses)

#         with conn.cursor(cursor_factory=RealDictCursor) as cur:
#             cur.execute(
#                 f"""
#                 SELECT document_id, source_type, file_name, file_path,
#                        processing_status, is_duplicate, duplicate_of,
#                        current_timestamp_ist, file_size
#                 FROM   document_metadata
#                 WHERE  {where_sql}
#                 ORDER  BY current_timestamp_ist DESC
#                 LIMIT  %s OFFSET %s
#                 """,
#                 params + [limit, offset],
#             )
#             files = [dict(r) for r in cur.fetchall()]

#         with conn.cursor() as cur:
#             cur.execute(
#                 f"SELECT COUNT(*) FROM document_metadata WHERE {where_sql}",
#                 params,
#             )
#             total = cur.fetchone()[0]

#         return {"total": total, "limit": limit, "offset": offset, "files": files}
#     finally:
#         conn.close()


@router.get("/files/all")
def list_all_files(
    source_type: Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
):
    """
    Return ALL records from document_metadata with latest classification info.
    Optionally filter by source_type.
    """
    conn = get_connection()
    try:
        where_clauses = ["dm.processing_status != 'Deleted'"]
        params: list = []

        if source_type:
            where_clauses.append("dm.source_type = %s")
            params.append(source_type)

        where_sql = " AND ".join(where_clauses)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT dm.document_id, dm.source_type, dm.file_name, dm.file_path,
                       dm.processing_status, dm.is_duplicate, dm.duplicate_of,
                       dm.current_timestamp_ist, dm.file_size,
                       cd.document_type, cd.classification_status,
                       cd.confidence_score, cd.type_of_classification
                FROM   document_metadata dm
                {_CLASSIFY_JOIN}
                WHERE  {where_sql}
                ORDER  BY dm.current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params + [limit, offset],
            )
            files = [dict(r) for r in cur.fetchall()]

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM document_metadata dm WHERE {where_sql}",
                params,
            )
            total = cur.fetchone()[0]

        return {"total": total, "limit": limit, "offset": offset, "files": files}
    finally:
        conn.close()

@router.get("/files/summary")
def get_summary():
    """
    Dashboard summary counts from document_metadata.

    Includes all processing status values:
    Original, Duplicate, Deleted, Ingested (and any other status in the table).
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                                      AS total,
                    COUNT(*) FILTER (WHERE processing_status = 'Original')       AS originals,
                    COUNT(*) FILTER (WHERE processing_status = 'Duplicate')      AS duplicates,
                    COUNT(*) FILTER (WHERE processing_status = 'Deleted')        AS deleted,
                    COUNT(*) FILTER (WHERE processing_status = 'Ingested')       AS ingested,
                    COUNT(*) FILTER (WHERE processing_status NOT IN
                        ('Original','Duplicate','Deleted','Ingested'))            AS other,
                    COUNT(*) FILTER (WHERE source_type = 'Outlook')              AS from_outlook,
                    COUNT(*) FILTER (WHERE source_type = 'Sharepoint')           AS from_sharepoint,
                    COUNT(*) FILTER (WHERE source_type = 'manual_upload')        AS from_manual
                FROM document_metadata
                """
            )
            return dict(cur.fetchone())
    finally:
        conn.close()


@router.get("/files/duplicates-by-name")
def get_duplicates_by_name(file_name: str):
    """
    Return all duplicate records that share the given file_name.

    Only records with is_duplicate=TRUE are returned.
    The original record (is_duplicate=FALSE) for this file_name is not included.

    Example:
        GET /api/files/duplicates-by-name?file_name=bama_10534.pdf
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT document_id, source_type, file_name, file_path,
                       processing_status, is_duplicate, duplicate_of,
                       current_timestamp_ist, file_size
                FROM   document_metadata
                WHERE  file_name         = %s
                  AND  is_duplicate      = TRUE
                  AND  processing_status != 'Deleted'
                ORDER  BY current_timestamp_ist ASC
                """,
                (file_name,),
            )
            duplicates = [dict(r) for r in cur.fetchall()]

        return {
            "file_name":  file_name,
            "total":      len(duplicates),
            "duplicates": duplicates,
        }
    finally:
        conn.close()

@router.get("/files/deleted")
def list_deleted_files(
    source_type: Optional[str] = None,
    filename:    Optional[str] = None,
    limit:  int = 20,
    offset: int = 0,
):
    """Paginated list of soft-deleted document_metadata records."""
    conn = get_connection()
    try:
        where  = ["processing_status = 'Deleted'"]
        params: list = []

        if source_type:
            where.append("source_type = %s")
            params.append(source_type)
        if filename:
            where.append("file_name ILIKE %s")
            params.append(f"%{filename}%")

        where_sql = " AND ".join(where)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM document_metadata WHERE {where_sql}",
                params,
            )
            total = cur.fetchone()["count"]

            cur.execute(
                f"""
                SELECT document_id, file_name, file_path, processing_status,
                       is_duplicate, duplicate_of, source_type, file_size,
                       current_timestamp_ist, processed_at
                FROM   document_metadata
                WHERE  {where_sql}
                ORDER  BY processed_at DESC NULLS LAST, current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params + [limit, offset],
            )
            items = [dict(r) for r in cur.fetchall()]

        return {
            "total":    total,
            "limit":    limit,
            "offset":   offset,
            "has_more": offset + len(items) < total,
            "items":    items,
        }
    finally:
        conn.close()

@router.delete("/files/{document_id}")
def delete_file_endpoint(document_id: str):
    """
    Delete a file from Azure Blob Storage and soft-delete its row in document_metadata.
    The row is kept with processing_status='Deleted' for audit purposes.
    """
    conn = get_connection()
    try:
        return _delete_file(conn, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Deletion failed: {exc}")
    finally:
        conn.close()


@router.post("/ingest/check-duplicate")
def check_duplicate_endpoint(body: IngestCheckRequest):
    """
    Call immediately after inserting a new row into document_metadata.
    Returns whether the new file is a duplicate and what it duplicates.
    """
    conn = get_connection()
    try:
        return _check_and_mark_duplicate(conn, body.file_name, body.file_path)
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.post("/admin/scan-duplicates")
def scan_duplicates_endpoint():
    """
    Bulk reclassify all existing records as Original or Duplicate.
    Run ONCE after schema changes or data migrations.
    After that, /api/ingest/check-duplicate handles each new ingestion.
    """
    conn = get_connection()
    try:
        return _scan_all_existing_duplicates(conn)
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.get("/duplicates")
def list_duplicate_groups(
    source_type: Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
):
    """
    Return files grouped by file_name where at least one copy is a Duplicate.
    Each group includes the original + all duplicate copies.
    """
    conn = get_connection()
    try:
        where_extra = ""
        params: list = []
        if source_type:
            where_extra = "AND source_type = %s"
            params.append(source_type)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) FROM (
                    SELECT 1 FROM document_metadata
                    WHERE  processing_status != 'Deleted' {where_extra}
                    GROUP  BY file_name
                    HAVING COUNT(*) FILTER (WHERE is_duplicate) > 0
                ) sub
                """,
                params,
            )
            total_groups = cur.fetchone()["count"]

            cur.execute(
                f"""
                SELECT   file_name,
                         COUNT(*)                          AS copy_count,
                         MIN(current_timestamp_ist)        AS earliest,
                         MAX(current_timestamp_ist)        AS latest,
                         ARRAY_AGG(DISTINCT source_type)   AS sources
                FROM     document_metadata
                WHERE    processing_status != 'Deleted' {where_extra}
                GROUP BY file_name
                HAVING   COUNT(*) FILTER (WHERE is_duplicate) > 0
                ORDER BY copy_count DESC, file_name
                LIMIT    %s OFFSET %s
                """,
                params + [limit, offset],
            )
            group_rows = cur.fetchall()

            groups = []
            for gr in group_rows:
                cur.execute(
                    """
                    SELECT document_id, file_name, file_path, processing_status,
                           is_duplicate, source_type, file_size,
                           current_timestamp_ist, processed_at, duplicate_of
                    FROM   document_metadata
                    WHERE  file_name         = %s
                      AND  processing_status != 'Deleted'
                    ORDER  BY is_duplicate ASC, current_timestamp_ist ASC
                    """,
                    (gr["file_name"],),
                )
                copies = [dict(c) for c in cur.fetchall()]
                groups.append({
                    "file_name":  gr["file_name"],
                    "copy_count": gr["copy_count"],
                    "sources":    [s for s in gr["sources"] if s],
                    "earliest":   gr["earliest"].isoformat() if gr["earliest"] else None,
                    "latest":     gr["latest"].isoformat()   if gr["latest"]   else None,
                    "copies":     copies,
                })

        return {"total_groups": total_groups, "offset": offset, "limit": limit, "groups": groups}
    finally:
        conn.close()