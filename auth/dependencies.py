"""
auth/dependencies.py — FastAPI dependency injection for RBAC.

Usage in routers:
    from auth.dependencies import get_current_user, require_admin

    @router.get("/something")
    def endpoint(user: dict = Depends(get_current_user)):
        ...

    @router.delete("/something")
    def admin_endpoint(user: dict = Depends(require_admin)):
        ...
"""

import hashlib
import logging
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg2.extras import RealDictCursor

from auth.jwt_handler import decode_token
from database import get_connection

log = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


# ── Core user extraction ──────────────────────────────────────────────────────

def _user_from_token(credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[dict]:
    """Decode Bearer JWT and return user payload dict, or None."""
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload or payload.get("type") != "access":
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    # Verify user still exists and is active
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, email, full_name, role, is_active "
                "FROM rbac_users WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or not row["is_active"]:
        return None

    return dict(row)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Requires a valid JWT. Raises 401 if missing/invalid."""
    user = _user_from_token(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """Returns user dict if token is valid, None otherwise (no error)."""
    return _user_from_token(credentials)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Requires authenticated user with role='admin'."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user


# ── API key (Logic App identity) ──────────────────────────────────────────────

def get_api_key_identity(x_api_key: Optional[str] = Header(None)) -> dict:
    """
    Validate X-API-Key header for Logic App / service account requests.
    Returns { key_id, name, source_type } or raises 401.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header required.",
        )

    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT key_id, name, source_type
                FROM   rbac_api_keys
                WHERE  key_hash = %s AND is_active = TRUE
                  AND  (expires_at IS NULL OR expires_at > NOW())
                """,
                (key_hash,),
            )
            key_row = cur.fetchone()

        if not key_row:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key.",
            )

        # Update last_used_at
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE rbac_api_keys SET last_used_at = NOW() WHERE key_id = %s",
                (key_row["key_id"],),
            )
        conn.commit()
        return dict(key_row)

    finally:
        conn.close()
