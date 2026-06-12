"""
utils/blob.py — Azure Blob Storage helpers.

Key path conventions
────────────────────
Source paths (stored in document_metadata.file_path):
    Outlook/SharePoint:  /csg-tfm-test/{source}/{guid_folder}/{filename}
                         /csg-tfm-test/{source}/{guid_folder}/{stem_folder}/{filename}
    manual_upload:       manual_upload/{guid_filename}

ADLS actual key (stem folder is skipped — it does not exist in ADLS):
    {source}/{guid_folder}/{filename}
    manual_upload/{guid_filename}

DIC output key (classified files):
    DIC/{category}/{source}/{guid_folder}/{filename}
    DIC/{category}/manual_upload/{guid_filename}
"""

import logging
from config import (
    SOURCE_CONN_STR, SOURCE_CONTAINER,
    DIC_CONN_STR, DIC_CONTAINER, DIC_ROOT_FOLDER,
)
from azure.storage.blob import BlobServiceClient, ContentSettings

log = logging.getLogger(__name__)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _parse_path(file_path: str) -> dict:
    """
    Parse file_path from document_metadata into its components.

    Returns:
        source_type  – "Outlook", "sharepoint", "manual_upload", …
        guid_folder  – GUID-prefixed folder name (None for manual_upload)
        filename     – bare filename with extension
        has_guid     – True if a separate guid_folder exists
    """
    parts = file_path.lstrip("/").split("/")

    # Strip container prefix
    if parts[0] == SOURCE_CONTAINER:
        parts = parts[1:]

    # parts: [source_type, guid_folder?, (stem_folder?), filename]
    source_type = parts[0]
    filename    = parts[-1]

    if len(parts) >= 3:
        # Outlook / SharePoint — has a guid_folder
        return {
            "source_type": source_type,
            "guid_folder": parts[1],          # always index 1
            "filename":    filename,
            "has_guid":    True,
        }
    else:
        # manual_upload — guid is baked into the filename itself
        return {
            "source_type": source_type,
            "guid_folder": None,
            "filename":    filename,
            "has_guid":    False,
        }


# ─── Public helpers ───────────────────────────────────────────────────────────

def file_path_to_source_key(file_path: str) -> str:
    """
    Convert file_path from document_metadata to an actual ADLS blob key.

    DB may store an extra stem_folder level that doesn't exist in ADLS —
    we always take source/guid_folder/filename (skip stem_folder).

    Examples:
        /csg-tfm-test/Outlook/a31d183_Attach/stem/Attach.pdf → Outlook/a31d183_Attach/Attach.pdf
        manual_upload/7549-guid_wordpress.pdf                → manual_upload/7549-guid_wordpress.pdf
    """
    p = _parse_path(file_path)
    if p["has_guid"]:
        return f"{p['source_type']}/{p['guid_folder']}/{p['filename']}"
    return f"{p['source_type']}/{p['filename']}"


def build_dic_blob_key(document_type: str, file_path: str) -> str:
    """
    Build the DIC output blob key, preserving the original GUID folder structure.

    Required format:
        DIC/{category}/{source}/{guid_folder}/{filename}

    Examples:
        DIC/Contracts/Outlook/a31d183_Attach/Attachment_3-_Contract_Document.pdf
        DIC/Invoices/manual_upload/7549-guid_wordpress.pdf

    Args:
        document_type: classified category (e.g. "Invoices", "Contracts")
        file_path:     file_path column from document_metadata
    """
    p = _parse_path(file_path)
    if p["has_guid"]:
        return (
            f"{DIC_ROOT_FOLDER}/{document_type}"
            f"/{p['source_type']}/{p['guid_folder']}/{p['filename']}"
        )
    return f"{DIC_ROOT_FOLDER}/{document_type}/{p['source_type']}/{p['filename']}"


def download_source_blob(file_path: str) -> bytes:
    """Download raw bytes from the source ADLS container."""
    key = file_path_to_source_key(file_path)
    log.info(f"  Downloading: {key}")
    client = BlobServiceClient.from_connection_string(SOURCE_CONN_STR)
    return (
        client.get_container_client(SOURCE_CONTAINER)
              .get_blob_client(key)
              .download_blob()
              .readall()
    )


def upload_to_dic(file_bytes: bytes, output_key: str,
                  content_type: str = "application/octet-stream") -> None:
    """Upload a classified file to the DIC output container/folder."""
    client = BlobServiceClient.from_connection_string(DIC_CONN_STR)
    client.get_container_client(DIC_CONTAINER) \
          .get_blob_client(output_key) \
          .upload_blob(
              file_bytes,
              overwrite=True,
              content_settings=ContentSettings(content_type=content_type),
          )
    log.info(f"  Uploaded to DIC → {output_key}")


def upload_source_blob(
    file_bytes: bytes,
    blob_path: str,
    content_type: str = "application/octet-stream",
    metadata: dict = None,
) -> None:
    """Upload a new file into the source container.

    blob_path may include the container prefix (e.g. csg-tfm-test/manual_upload/…)
    or be a bare key (manual_upload/…). The container prefix is stripped before
    the SDK call so the blob lands at the correct path within the container.
    """
    key = file_path_to_source_key(blob_path)
    client = BlobServiceClient.from_connection_string(SOURCE_CONN_STR)
    client.get_container_client(SOURCE_CONTAINER) \
          .get_blob_client(key) \
          .upload_blob(
              file_bytes,
              overwrite=True,
              content_settings=ContentSettings(content_type=content_type),
              metadata=metadata or {},
          )
    log.info(f"  Uploaded to source → {key}")


def delete_source_blob(file_path: str) -> None:
    """Delete a blob from the source container (used by the duplicate deletion API)."""
    key = file_path_to_source_key(file_path)
    try:
        client = BlobServiceClient.from_connection_string(SOURCE_CONN_STR)
        client.get_container_client(SOURCE_CONTAINER).delete_blob(key)
        log.info(f"  Blob deleted: {key}")
    except Exception as exc:
        log.warning(f"  Blob deletion warning ({key}): {exc}")