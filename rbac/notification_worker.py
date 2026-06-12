"""
rbac/notification_worker.py — Background worker that creates notifications
for document owners when classification events occur.

Does NOT modify any existing tables or endpoints.
Polls classification_docs every 60 seconds for new events that need
notifications, then inserts into rbac_notifications.

Notification triggers:
  soft_flag / hard_stop  → owner notified "document in review"
  auto_classified        → owner notified "document classified"
  human_approved         → owner notified "document approved"
  human_rejected         → owner notified "document rejected"

Only manual_upload documents have individual owners and receive notifications.
Outlook / SharePoint / RDBMS documents have no personal owner — no notification.
"""

import asyncio
import logging

from database import get_connection

log = logging.getLogger(__name__)


def _create_if_missing(conn, user_id: str, notif_type: str,
                        title: str, message: str,
                        document_id: str, classification_id: str) -> None:
    """Insert notification only if one of the same type doesn't already exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM rbac_notifications
            WHERE  user_id           = %s
              AND  classification_id = %s
              AND  type              = %s
            LIMIT  1
            """,
            (user_id, classification_id, notif_type),
        )
        if cur.fetchone():
            return  # already notified

        cur.execute(
            """
            INSERT INTO rbac_notifications
                (user_id, type, title, message, document_id, classification_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user_id, notif_type, title, message, document_id, classification_id),
        )


def check_and_create_notifications() -> None:
    """
    Single pass: find classification events that need notifications and create them.
    Only processes manual_upload documents (they have individual owners).
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Find all classification events for manual_upload docs that have an owner
            cur.execute(
                """
                SELECT
                    cd.classification_id,
                    cd.document_id,
                    cd.classification_status,
                    cd.document_type,
                    cd.combined_confidence_score,
                    cd.current_timestamp_ist,
                    dm.file_name,
                    rdo.user_id
                FROM   classification_docs  cd
                JOIN   document_metadata    dm  ON dm.document_id = cd.document_id
                JOIN   rbac_document_ownership rdo ON rdo.document_id = cd.document_id
                WHERE  dm.source_type    = 'manual_upload'
                  AND  rdo.user_id       IS NOT NULL
                  AND  cd.classification_status IN (
                       'soft_flag', 'hard_stop',
                       'auto_classified',
                       'human_approved', 'human_rejected'
                  )
                """
            )
            rows = cur.fetchall()

        for row in rows:
            (cls_id, doc_id, status, category,
             confidence, ts, file_name, user_id) = row

            doc_id_str = str(doc_id)
            cls_id_str = str(cls_id)
            user_id_str = str(user_id)

            if status in ("soft_flag", "hard_stop"):
                _create_if_missing(
                    conn, user_id_str,
                    "document_in_review",
                    "Your document is under review",
                    f'"{file_name}" has been flagged for human review '
                    f'(confidence {confidence:.0f}%). '
                    "An admin will review it shortly.",
                    doc_id_str, cls_id_str,
                )

            elif status == "auto_classified":
                _create_if_missing(
                    conn, user_id_str,
                    "document_classified",
                    "Document classified",
                    f'"{file_name}" was automatically classified '
                    f'as {category} ({confidence:.0f}% confidence).',
                    doc_id_str, cls_id_str,
                )

            elif status == "human_approved":
                _create_if_missing(
                    conn, user_id_str,
                    "document_approved",
                    "Document approved",
                    f'"{file_name}" was reviewed and approved '
                    f'as {category} by an admin.',
                    doc_id_str, cls_id_str,
                )

            elif status == "human_rejected":
                _create_if_missing(
                    conn, user_id_str,
                    "document_rejected",
                    "Document sent to Miscellaneous",
                    f'"{file_name}" was reviewed and could not be '
                    "classified into a specific category. "
                    "It has been moved to Miscellaneous.",
                    doc_id_str, cls_id_str,
                )

        conn.commit()

    except Exception as exc:
        log.error("Notification worker error: %s", exc)
        conn.rollback()
    finally:
        conn.close()


async def notification_worker_loop() -> None:
    """Asyncio background task — runs forever until cancelled."""
    log.info("Notification worker started (60s interval)")
    while True:
        await asyncio.sleep(60)
        try:
            check_and_create_notifications()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("Notification worker unhandled error: %s", exc)
