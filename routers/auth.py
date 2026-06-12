"""
routers/auth.py — Authentication endpoints.

POST /api/auth/login    — email + password → access + refresh tokens
POST /api/auth/refresh  — refresh token → new access token
POST /api/auth/logout   — revoke refresh token
GET  /api/auth/me       — current user info
"""

import hashlib
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, EmailStr

from auth.dependencies import get_current_user
from auth.jwt_handler import (
    create_access_token, create_refresh_token, decode_token,
)
from auth.password import verify_password
from database import get_connection

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ── Request / response schemas ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str

class LoginResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    user_id:       str
    email:         str
    full_name:     str
    role:          str

class RefreshRequest(BaseModel):
    refresh_token: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_audit(conn, user_id: Optional[str], action: str,
               resource_type: str, resource_id: Optional[str],
               details: Optional[dict], ip: Optional[str]) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rbac_audit_logs
                    (user_id, action, resource_type, resource_id, details, ip_address)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, action, resource_type, resource_id,
                 json.dumps(details) if details else None, ip),
            )
    except Exception as exc:
        log.warning("Audit log write failed: %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT user_id, email, full_name, password_hash, role, is_active "
                "FROM rbac_users WHERE email = %s",
                (body.email.lower().strip(),),
            )
            user = cur.fetchone()

        ip = request.client.host if request.client else None

        if not user or not verify_password(body.password, user["password_hash"]):
            _log_audit(conn, None, "login_failed", "user",
                       body.email, {"reason": "bad credentials"}, ip)
            conn.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )

        if not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated. Contact an admin.",
            )

        user_id  = str(user["user_id"])
        role     = user["role"]
        email    = user["email"]

        access_token                       = create_access_token(user_id, email, role)
        refresh_token, token_id, exp_at    = create_refresh_token(user_id)
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rbac_refresh_tokens
                    (user_id, token_hash, expires_at, ip_address)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id, token_hash, exp_at, ip),
            )
            cur.execute(
                "UPDATE rbac_users SET last_login_at = NOW() WHERE user_id = %s",
                (user_id,),
            )

        _log_audit(conn, user_id, "login", "user", user_id, None, ip)
        conn.commit()

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user_id,
            email=email,
            full_name=user["full_name"],
            role=role,
        )
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        log.error("Login error: %s", exc)
        raise HTTPException(status_code=500, detail="Login failed.")
    finally:
        conn.close()


@router.post("/refresh")
def refresh_token(body: RefreshRequest):
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
        )

    user_id    = payload["sub"]
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT rt.token_id, rt.revoked
                FROM   rbac_refresh_tokens rt
                WHERE  rt.token_hash = %s
                  AND  rt.user_id    = %s
                  AND  rt.expires_at > NOW()
                """,
                (token_hash, user_id),
            )
            row = cur.fetchone()

        if not row or row["revoked"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token is invalid or expired.",
            )

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT email, role, is_active FROM rbac_users WHERE user_id = %s",
                (user_id,),
            )
            user = cur.fetchone()

        if not user or not user["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated.",
            )

        new_access = create_access_token(user_id, user["email"], user["role"])
        conn.commit()
        return {"access_token": new_access, "token_type": "bearer"}

    except HTTPException:
        raise
    finally:
        conn.close()


@router.post("/logout")
def logout(body: RefreshRequest, user: dict = Depends(get_current_user)):
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rbac_refresh_tokens
                SET    revoked = TRUE, revoked_at = NOW()
                WHERE  token_hash = %s AND user_id = %s
                """,
                (token_hash, user["user_id"]),
            )
            cur.execute(
                """
                INSERT INTO rbac_audit_logs (user_id, action, resource_type, resource_id)
                VALUES (%s, 'logout', 'user', %s)
                """,
                (str(user["user_id"]), str(user["user_id"])),
            )
        conn.commit()
        return {"message": "Logged out successfully."}
    finally:
        conn.close()


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    return {
        "user_id":   str(user["user_id"]),
        "email":     user["email"],
        "full_name": user["full_name"],
        "role":      user["role"],
    }
