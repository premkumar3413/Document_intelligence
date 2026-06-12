"""
routers/rbac_audit.py — Audit log endpoint (admin only).

GET /api/rbac/audit   paginated, filterable audit log
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from psycopg2.extras import RealDictCursor

from auth.dependencies import require_admin
from database import get_connection

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rbac/audit", tags=["RBAC Audit"])


@router.get("")
def get_audit_log(
    action:        Optional[str] = None,
    resource_type: Optional[str] = None,
    user_id:       Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
    admin: dict = Depends(require_admin),
):
    conn = get_connection()
    try:
        where, params = ["1=1"], []
        if action:
            where.append("al.action = %s");        params.append(action)
        if resource_type:
            where.append("al.resource_type = %s"); params.append(resource_type)
        if user_id:
            where.append("al.user_id = %s");       params.append(user_id)

        where_sql = " AND ".join(where)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT  al.audit_id, al.action, al.resource_type, al.resource_id,
                        al.details, al.ip_address, al.created_at,
                        u.email AS user_email, u.full_name AS user_name
                FROM    rbac_audit_logs al
                LEFT JOIN rbac_users u ON u.user_id = al.user_id
                WHERE   {where_sql}
                ORDER   BY al.created_at DESC
                LIMIT   %s OFFSET %s
                """,
                params + [limit, offset],
            )
            items = [dict(r) for r in cur.fetchall()]

            cur.execute(
                f"SELECT COUNT(*) FROM rbac_audit_logs al WHERE {where_sql}",
                params,
            )
            total = cur.fetchone()["count"]

        return {"total": total, "limit": limit, "offset": offset, "items": items}
    finally:
        conn.close()
