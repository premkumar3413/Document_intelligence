"""
routers/upload.py — Manual document upload endpoint.

Endpoint
────────
POST /api/upload
    Accept a multipart file + uploader metadata, upload the bytes to Azure
    Blob Storage under manual_upload/{guid12}_{stem}/{filename}, insert a row into
    document_metadata (document_id PK schema), then run the filename-based
    duplicate check to mark the row as Original or Duplicate.

Table ensured on startup (see ensure_document_metadata_table):
    document_metadata(
        document_id UUID PK, file_name, file_path UNIQUE,
        processing_status, is_duplicate, duplicate_of,
        source_type, file_size, current_timestamp_ist, processed_at
    )
"""

import logging
import mimetypes
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from psycopg2.extras import RealDictCursor

from database import get_connection
from utils.blob import upload_source_blob

log = logging.getLogger(__name__)

router = APIRouter(tags=["Upload"])

# ── Table bootstrap (idempotent) ──────────────────────────────────────────────

_ENSURE_TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS document_metadata (
        document_id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        file_name             VARCHAR(500) NOT NULL,
        file_path             VARCHAR(1000) NOT NULL,
        processing_status     VARCHAR(20)  NOT NULL DEFAULT 'In Progress',
        is_duplicate          BOOLEAN      NOT NULL DEFAULT FALSE,
        duplicate_of          VARCHAR(500),
        source_type           VARCHAR(50),
        file_size             BIGINT,
        current_timestamp_ist TIMESTAMPTZ  NOT NULL
                              DEFAULT (NOW() AT TIME ZONE 'Asia/Kolkata'),
        processed_at          TIMESTAMPTZ
    );
    """,
    # Idempotent column additions for tables created before processed_at existed
    "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ;",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_meta_file_path ON document_metadata(file_path);",
    "CREATE INDEX IF NOT EXISTS idx_doc_meta_file_name ON document_metadata(LOWER(file_name));",
    "CREATE INDEX IF NOT EXISTS idx_doc_meta_status    ON document_metadata(processing_status);",
    "CREATE INDEX IF NOT EXISTS idx_doc_meta_source    ON document_metadata(source_type);",
    "CREATE INDEX IF NOT EXISTS idx_doc_meta_ts        ON document_metadata(current_timestamp_ist DESC);",
]


def ensure_document_metadata_table() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for stmt in _ENSURE_TABLE_SQL:
                cur.execute(stmt)
        conn.commit()
        log.info("document_metadata schema ensured")
    except Exception as exc:
        conn.rollback()
        log.error("document_metadata schema setup failed: %s", exc)
    finally:
        conn.close()


# ── Dedup helper ──────────────────────────────────────────────────────────────

def _check_and_mark_duplicate(conn, file_name: str, file_path: str) -> dict:
    """
    Called right after inserting the new document_metadata row.
    Searches for an existing Original/unclassified record with the same
    file_name (Deleted and Duplicate rows are excluded from matching).
    Updates the new row to Original or Duplicate and returns the result.
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
        log.info("Duplicate detected: '%s' already at '%s'", file_name, existing["file_path"])
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

    log.info("Original file: '%s'", file_name)
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


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload a document to Azure Blob Storage")
def manual_upload(
    file: UploadFile = File(..., description="Document file — all types accepted"),
    username: str = Form(..., description="Name of the uploader"),
    tags: str = Form("", description="Comma-separated tags e.g. finance,q1"),
    version_label: str = Form("", description="Optional version label e.g. v1.0, draft"),
):
    data         = file.file.read()
    filename     = file.filename or "unknown"
    extension    = Path(filename).suffix.lower()
    content_type = (
        file.content_type
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Blob path mirrors the Outlook convention: container/source/guid_folder/filename
    # e.g. csg-tfm-test/manual_upload/a1b2c3d4e5f6_report/report.pdf
    # _parse_path() strips the container prefix automatically, so download/delete
    # and classify helpers all resolve the correct ADLS key without further changes.
    doc_id    = uuid.uuid4()
    stem      = Path(filename).stem
    guid_part = doc_id.hex[:12]          # first 12 hex chars — no hyphens
    folder    = f"{guid_part}_{stem}"
    blob_path = f"csg-tfm-test/manual_upload/{folder}/{filename}"

    # ── 1. Blob upload ────────────────────────────────────────────────────────
    try:
        upload_source_blob(
            data,
            blob_path,
            content_type=content_type,
            metadata={
                "source":            "manual_upload",
                "uploaded_by":       username,
                "original_filename": filename,
            },
        )
    except Exception as exc:
        log.error("Blob upload failed — file=%s  error=%s", filename, exc)
        raise HTTPException(status_code=502, detail=f"Storage upload failed: {exc}")

    # ── 2. DB insert + dedup ─────────────────────────────────────────────────
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
        conn.commit()

        dedup = _check_and_mark_duplicate(conn, filename, blob_path)

    except Exception as exc:
        conn.rollback()
        log.error("DB insert/dedup failed — file=%s  error=%s", filename, exc)
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()

    log.info(
        "Upload complete — id=%s  user=%s  file=%s  size=%d  is_duplicate=%s",
        doc_id, username, filename, len(data), dedup["is_duplicate"],
    )
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
