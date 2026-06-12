"""
routers/rbac_notifications.py — Notification endpoints.

GET  /api/rbac/notifications              list notifications for current user
GET  /api/rbac/notifications/unread-count unread count (for bell badge)
PUT  /api/rbac/notifications/{id}/read    mark one as read
PUT  /api/rbac/notifications/read-all     mark all as read
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import RealDictCursor

from auth.dependencies import get_current_user
from database import get_connection

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rbac/notifications", tags=["RBAC Notifications"])


@router.get("/unread-count")
def unread_count(user: dict = Depends(get_current_user)):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM rbac_notifications "
                "WHERE user_id = %s AND is_read = FALSE",
                (str(user["user_id"]),),
            )
            return {"unread_count": cur.fetchone()[0]}
    finally:
        conn.close()


@router.get("")
def list_notifications(
    unread_only: bool = False,
    limit:  int = 50,
    offset: int = 0,
    user: dict = Depends(get_current_user),
):
    conn = get_connection()
    try:
        extra = "AND is_read = FALSE" if unread_only else ""
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT notification_id, type, title, message,
                       document_id, classification_id,
                       is_read, created_at, read_at
                FROM   rbac_notifications
                WHERE  user_id = %s {extra}
                ORDER  BY created_at DESC
                LIMIT  %s OFFSET %s
                """,
                (str(user["user_id"]), limit, offset),
            )
            items = [dict(r) for r in cur.fetchall()]

            cur.execute(
                f"SELECT COUNT(*) FROM rbac_notifications WHERE user_id = %s {extra}",
                (str(user["user_id"]),),
            )
            total = cur.fetchone()["count"]

        return {"total": total, "limit": limit, "offset": offset, "items": items}
    finally:
        conn.close()


@router.put("/read-all")
def mark_all_read(user: dict = Depends(get_current_user)):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rbac_notifications
                SET    is_read = TRUE, read_at = NOW()
                WHERE  user_id = %s AND is_read = FALSE
                """,
                (str(user["user_id"]),),
            )
            count = cur.rowcount
        conn.commit()
        return {"marked_read": count}
    finally:
        conn.close()


@router.put("/{notification_id}/read")
def mark_read(notification_id: str, user: dict = Depends(get_current_user)):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rbac_notifications
                SET    is_read = TRUE, read_at = NOW()
                WHERE  notification_id = %s AND user_id = %s
                """,
                (notification_id, str(user["user_id"])),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Notification not found.")
        conn.commit()
        return {"message": "Marked as read."}
    except HTTPException:
        raise
    finally:
        conn.close()
