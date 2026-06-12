"""
routers/rbac_documents.py — RBAC-aware document endpoints.

All endpoints here are NEW — existing /api/files/* are UNTOUCHED.

Visibility rules:
  Admins     → see ALL documents from ALL sources
  Users      → see Outlook + Sharepoint + rdbms + rdbms_flow (all org docs)
               AND their own manual_upload documents

GET    /api/rbac/documents                 list documents (visibility-filtered)
GET    /api/rbac/documents/{document_id}   single document (access checked)
GET    /api/rbac/documents/{document_id}/download   download (access checked)
POST   /api/rbac/upload                    upload + create ownership record
"""

import io
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from psycopg2.extras import RealDictCursor

from auth.dependencies import get_current_user, require_admin
from database import get_connection
from rbac_config import ORG_SOURCE_TYPES
from utils.blob import (
    download_source_blob, upload_source_blob,
    file_path_to_source_key,
)

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rbac", tags=["RBAC Documents"])

# The lateral join that attaches the latest classification to each document row
_CLS_JOIN = """
    LEFT JOIN LATERAL (
        SELECT document_type, classification_status,
               combined_confidence_score AS confidence_score,
               type_of_classification
        FROM   classification_docs
        WHERE  document_id = dm.document_id
        ORDER  BY current_timestamp_ist DESC
        LIMIT  1
    ) cd ON TRUE
"""

# Tuple → SQL IN list string helper
def _in(vals: tuple) -> str:
    return "(" + ",".join(f"'{v}'" for v in vals) + ")"


def _visibility_clause(user: dict) -> tuple[str, list]:
    """Return (WHERE clause fragment, params) based on user role."""
    if user["role"] == "admin":
        return "dm.processing_status != 'Deleted'", []
    return (
        f"""
        dm.processing_status != 'Deleted'
        AND (
            dm.source_type IN {_in(ORG_SOURCE_TYPES)}
            OR (
                dm.source_type = 'manual_upload'
                AND EXISTS (
                    SELECT 1 FROM rbac_document_ownership rdo
                    WHERE  rdo.document_id = dm.document_id
                      AND  rdo.user_id     = %s
                )
            )
        )
        """,
        [str(user["user_id"])],
    )


def _can_access(conn, document_id: str, user: dict) -> Optional[dict]:
    """Return the document row if the user can access it, else None."""
    vis_clause, params = _visibility_clause(user)
    params.append(document_id)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT dm.document_id, dm.file_name, dm.file_path, dm.source_type,
                   dm.processing_status, dm.is_duplicate, dm.duplicate_of,
                   dm.current_timestamp_ist, dm.file_size,
                   cd.document_type, cd.classification_status,
                   cd.confidence_score, cd.type_of_classification
            FROM   document_metadata dm
            {_CLS_JOIN}
            WHERE  {vis_clause}
              AND  dm.document_id = %s
            """,
            params,
        )
        row = cur.fetchone()
    return dict(row) if row else None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/documents")
def list_documents(
    source_type:           Optional[str] = None,
    document_type:         Optional[str] = None,
    classification_status: Optional[str] = None,
    search:                Optional[str] = None,
    limit:  int = 50,
    offset: int = 0,
    user: dict = Depends(get_current_user),
):
    conn = get_connection()
    try:
        vis_clause, params = _visibility_clause(user)

        extra = []
        if source_type:
            extra.append("dm.source_type = %s");     params.append(source_type)
        if document_type:
            extra.append("cd.document_type = %s");   params.append(document_type)
        if classification_status:
            extra.append("cd.classification_status = %s"); params.append(classification_status)
        if search:
            extra.append("dm.file_name ILIKE %s");   params.append(f"%{search}%")

        where_sql = vis_clause
        if extra:
            where_sql += " AND " + " AND ".join(extra)

        count_params = list(params)
        params += [limit, offset]

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT dm.document_id, dm.source_type, dm.file_name,
                       dm.processing_status, dm.is_duplicate, dm.duplicate_of,
                       dm.current_timestamp_ist, dm.file_size,
                       cd.document_type, cd.classification_status,
                       cd.confidence_score, cd.type_of_classification
                FROM   document_metadata dm
                {_CLS_JOIN}
                WHERE  {where_sql}
                ORDER  BY dm.current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params,
            )
            docs = [dict(r) for r in cur.fetchall()]

            cur.execute(
                f"SELECT COUNT(*) FROM document_metadata dm {_CLS_JOIN} WHERE {where_sql}",
                count_params,
            )
            total = cur.fetchone()["count"]

        return {"total": total, "limit": limit, "offset": offset, "documents": docs}
    finally:
        conn.close()


@router.get("/documents/{document_id}")
def get_document(document_id: str, user: dict = Depends(get_current_user)):
    conn = get_connection()
    try:
        doc = _can_access(conn, document_id, user)
        if not doc:
            raise HTTPException(status_code=404,
                                detail="Document not found or access denied.")
        return doc
    finally:
        conn.close()


@router.get("/documents/{document_id}/download")
def download_document(document_id: str, user: dict = Depends(get_current_user)):
    conn = get_connection()
    try:
        doc = _can_access(conn, document_id, user)
        if not doc:
            raise HTTPException(status_code=404,
                                detail="Document not found or access denied.")
    finally:
        conn.close()

    try:
        file_bytes = download_source_blob(doc["file_path"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Storage unavailable: {exc}")

    content_type = mimetypes.guess_type(doc["file_name"])[0] or "application/octet-stream"
    safe_name    = doc["file_name"].replace('"', '')
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@router.post("/upload", status_code=201)
def rbac_upload(
    file:          UploadFile = File(...),
    username:      str        = Form(...),
    tags:          str        = Form(""),
    version_label: str        = Form(""),
    user: dict = Depends(get_current_user),
):
    """
    RBAC-aware upload — same as /api/upload but also creates an
    rbac_document_ownership record so the uploader owns the document.
    """
    data         = file.file.read()
    filename     = file.filename or "unknown"
    extension    = Path(filename).suffix.lower()
    content_type = (
        file.content_type
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )
    tag_list  = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    doc_id    = uuid.uuid4()
    stem      = Path(filename).stem
    guid_part = doc_id.hex[:12]
    folder    = f"{guid_part}_{stem}"
    blob_path = f"csg-tfm-test/manual_upload/{folder}/{filename}"

    # 1. Upload to blob
    try:
        upload_source_blob(
            data, blob_path, content_type=content_type,
            metadata={"source": "manual_upload", "uploaded_by": username,
                      "original_filename": filename},
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Storage upload failed: {exc}")

    # 2. DB insert + dedup + ownership
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_metadata
                    (document_id, file_name, file_path, source_type,
                     file_size, processing_status, is_duplicate)
                VALUES (%s, %s, %s, %s, %s, 'In Progress', FALSE)
                ON CONFLICT (file_path) DO NOTHING
                """,
                (str(doc_id), filename, blob_path, "manual_upload", len(data)),
            )

        # duplicate check (same logic as existing upload.py)
        from routers.duplicates import _check_and_mark_duplicate
        dedup = _check_and_mark_duplicate(conn, filename, blob_path)

        # ownership record
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rbac_document_ownership
                    (document_id, user_id, source_type)
                VALUES (%s, %s, 'manual_upload')
                ON CONFLICT (document_id) DO NOTHING
                """,
                (str(doc_id), str(user["user_id"])),
            )

        conn.commit()

    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()

    return {
        "document_id":    str(doc_id),
        "username":       username,
        "file_name":      filename,
        "blob_path":      blob_path,
        "size_bytes":     len(data),
        "content_type":   content_type,
        "file_extension": extension,
        "tags":           tag_list,
        "version_label":  version_label or None,
        "source_type":    "manual_upload",
        "is_duplicate":   dedup["is_duplicate"],
        "duplicate_of":   dedup["duplicate_of"],
    }
