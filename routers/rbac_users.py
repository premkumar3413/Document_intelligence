"""
routers/rbac_users.py — User management (admin only).

GET    /api/rbac/users               list all users
POST   /api/rbac/users               create a user
PUT    /api/rbac/users/{user_id}     update name / role / active status
DELETE /api/rbac/users/{user_id}     deactivate (soft delete)
POST   /api/rbac/api-keys            create an API key (for Logic Apps)
GET    /api/rbac/api-keys            list API keys
DELETE /api/rbac/api-keys/{key_id}   revoke an API key
"""

import hashlib
import json
import logging
import secrets
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

from auth.dependencies import require_admin
from auth.password import hash_password
from database import get_connection

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rbac", tags=["RBAC Users"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    email:     str
    full_name: str
    password:  str
    role:      str = "user"

class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    role:      Optional[str] = None
    is_active: Optional[bool] = None
    password:  Optional[str] = None

class CreateApiKeyRequest(BaseModel):
    name:        str
    source_type: str


# ── User endpoints ────────────────────────────────────────────────────────────

@router.get("/users")
def list_users(admin: dict = Depends(require_admin)):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT user_id, email, full_name, role, is_active,
                       created_at, last_login_at
                FROM   rbac_users
                ORDER  BY created_at DESC
                """
            )
            return {"users": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()


@router.post("/users", status_code=201)
def create_user(body: CreateUserRequest, admin: dict = Depends(require_admin)):
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'.")
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    conn = get_connection()
    try:
        new_id  = str(uuid.uuid4())
        pw_hash = hash_password(body.password)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rbac_users
                    (user_id, email, full_name, password_hash, role, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (new_id, body.email.lower().strip(), body.full_name,
                 pw_hash, body.role, str(admin["user_id"])),
            )
            cur.execute(
                """
                INSERT INTO rbac_audit_logs
                    (user_id, action, resource_type, resource_id, details)
                VALUES (%s, 'user_created', 'user', %s, %s)
                """,
                (str(admin["user_id"]), new_id,
                 json.dumps({"email": body.email, "role": body.role})),
            )
        conn.commit()
        return {"user_id": new_id, "email": body.email, "role": body.role,
                "message": "User created successfully."}

    except Exception as exc:
        conn.rollback()
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail="Email already registered.")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.put("/users/{user_id}")
def update_user(user_id: str, body: UpdateUserRequest,
                admin: dict = Depends(require_admin)):
    conn = get_connection()
    try:
        sets, params = [], []
        if body.full_name is not None:
            sets.append("full_name = %s"); params.append(body.full_name)
        if body.role is not None:
            if body.role not in ("admin", "user"):
                raise HTTPException(status_code=400, detail="Invalid role.")
            sets.append("role = %s"); params.append(body.role)
        if body.is_active is not None:
            sets.append("is_active = %s"); params.append(body.is_active)
        if body.password is not None:
            if len(body.password) < 8:
                raise HTTPException(status_code=400, detail="Password too short.")
            sets.append("password_hash = %s"); params.append(hash_password(body.password))

        if not sets:
            raise HTTPException(status_code=400, detail="No fields to update.")

        sets.append("updated_at = NOW()")
        params.append(user_id)

        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE rbac_users SET {', '.join(sets)} WHERE user_id = %s",
                params,
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found.")
            cur.execute(
                """
                INSERT INTO rbac_audit_logs
                    (user_id, action, resource_type, resource_id)
                VALUES (%s, 'user_updated', 'user', %s)
                """,
                (str(admin["user_id"]), user_id),
            )
        conn.commit()
        return {"message": "User updated."}
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.delete("/users/{user_id}")
def deactivate_user(user_id: str, admin: dict = Depends(require_admin)):
    if str(admin["user_id"]) == user_id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account.")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE rbac_users SET is_active = FALSE, updated_at = NOW() "
                "WHERE user_id = %s",
                (user_id,),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found.")
            cur.execute(
                """
                INSERT INTO rbac_audit_logs
                    (user_id, action, resource_type, resource_id)
                VALUES (%s, 'user_deactivated', 'user', %s)
                """,
                (str(admin["user_id"]), user_id),
            )
        conn.commit()
        return {"message": "User deactivated."}
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


# ── API key endpoints (Logic App / service identities) ────────────────────────

@router.post("/api-keys", status_code=201)
def create_api_key(body: CreateApiKeyRequest, admin: dict = Depends(require_admin)):
    """Generate an API key for a Logic App. The raw key is shown ONCE — store it now."""
    raw_key  = "dlk_" + secrets.token_hex(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id   = str(uuid.uuid4())

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rbac_api_keys
                    (key_id, key_hash, name, source_type, created_by)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (key_id, key_hash, body.name, body.source_type,
                 str(admin["user_id"])),
            )
        conn.commit()
        return {
            "key_id":      key_id,
            "name":        body.name,
            "source_type": body.source_type,
            "api_key":     raw_key,
            "warning":     "This key is shown ONCE. Copy it now and put it in your Logic App.",
        }
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.get("/api-keys")
def list_api_keys(admin: dict = Depends(require_admin)):
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT key_id, name, source_type, is_active,
                       created_at, last_used_at, expires_at
                FROM   rbac_api_keys
                ORDER  BY created_at DESC
                """
            )
            return {"api_keys": [dict(r) for r in cur.fetchall()]}
    finally:
        conn.close()


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, admin: dict = Depends(require_admin)):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE rbac_api_keys SET is_active = FALSE WHERE key_id = %s",
                (key_id,),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="API key not found.")
        conn.commit()
        return {"message": "API key revoked."}
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()
