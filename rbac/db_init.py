"""
rbac/db_init.py — Create all RBAC tables and seed the first admin.

Called once at server startup via main.py lifespan hook.
All statements are fully idempotent — safe to run on every restart.
"""

import logging
import uuid

from database import get_connection
from auth.password import hash_password
from rbac_config import FIRST_ADMIN_EMAIL, FIRST_ADMIN_PASSWORD, FIRST_ADMIN_NAME

log = logging.getLogger(__name__)

_DDL = [
    # ── rbac_users ────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rbac_users (
        user_id       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        email         VARCHAR(255) UNIQUE NOT NULL,
        full_name     VARCHAR(255) NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        role          VARCHAR(10)  NOT NULL DEFAULT 'user'
                                   CHECK (role IN ('admin','user')),
        is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
        created_by    UUID         REFERENCES rbac_users(user_id),
        created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        updated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        last_login_at TIMESTAMPTZ,
        azure_ad_oid  VARCHAR(100) UNIQUE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rbac_users_email  ON rbac_users(email)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_users_role   ON rbac_users(role)",

    # ── rbac_refresh_tokens ───────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rbac_refresh_tokens (
        token_id   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id    UUID         NOT NULL REFERENCES rbac_users(user_id) ON DELETE CASCADE,
        token_hash VARCHAR(255) NOT NULL UNIQUE,
        issued_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ  NOT NULL,
        revoked    BOOLEAN      NOT NULL DEFAULT FALSE,
        revoked_at TIMESTAMPTZ,
        ip_address VARCHAR(45),
        user_agent TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rbac_rt_user   ON rbac_refresh_tokens(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_rt_hash   ON rbac_refresh_tokens(token_hash)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_rt_expiry ON rbac_refresh_tokens(expires_at)",

    # ── rbac_api_keys ─────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rbac_api_keys (
        key_id       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        key_hash     VARCHAR(255) NOT NULL UNIQUE,
        name         VARCHAR(100) NOT NULL,
        source_type  VARCHAR(50)  NOT NULL,
        is_active    BOOLEAN      NOT NULL DEFAULT TRUE,
        created_by   UUID         NOT NULL REFERENCES rbac_users(user_id),
        created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        last_used_at TIMESTAMPTZ,
        expires_at   TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rbac_apikey_hash   ON rbac_api_keys(key_hash)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_apikey_source ON rbac_api_keys(source_type)",

    # ── rbac_document_ownership ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rbac_document_ownership (
        ownership_id UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id  UUID        NOT NULL UNIQUE,
        user_id      UUID        REFERENCES rbac_users(user_id),
        api_key_id   UUID        REFERENCES rbac_api_keys(key_id),
        source_type  VARCHAR(50) NOT NULL,
        uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT owner_or_key CHECK (user_id IS NOT NULL OR api_key_id IS NOT NULL)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rbac_own_user   ON rbac_document_ownership(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_own_key    ON rbac_document_ownership(api_key_id)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_own_source ON rbac_document_ownership(source_type)",

    # ── rbac_notifications ────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rbac_notifications (
        notification_id   UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id           UUID         NOT NULL REFERENCES rbac_users(user_id) ON DELETE CASCADE,
        type              VARCHAR(50)  NOT NULL CHECK (type IN (
                              'document_in_review',
                              'document_classified',
                              'document_approved',
                              'document_rejected'
                          )),
        title             VARCHAR(255) NOT NULL,
        message           TEXT         NOT NULL,
        document_id       UUID,
        classification_id UUID,
        is_read           BOOLEAN      NOT NULL DEFAULT FALSE,
        created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        read_at           TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rbac_notif_user    ON rbac_notifications(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_notif_unread  ON rbac_notifications(user_id, is_read) WHERE is_read = FALSE",
    "CREATE INDEX IF NOT EXISTS idx_rbac_notif_created ON rbac_notifications(created_at DESC)",

    # ── rbac_audit_logs ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS rbac_audit_logs (
        audit_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id       UUID         REFERENCES rbac_users(user_id),
        api_key_id    UUID         REFERENCES rbac_api_keys(key_id),
        action        VARCHAR(100) NOT NULL,
        resource_type VARCHAR(50),
        resource_id   VARCHAR(255),
        details       JSONB,
        ip_address    VARCHAR(45),
        user_agent    TEXT,
        created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rbac_audit_user    ON rbac_audit_logs(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_audit_action  ON rbac_audit_logs(action)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_audit_created ON rbac_audit_logs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_rbac_audit_res     ON rbac_audit_logs(resource_type, resource_id)",
]


def ensure_rbac_tables() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for stmt in _DDL:
                cur.execute(stmt)
        conn.commit()
        log.info("RBAC tables ensured")
    except Exception as exc:
        conn.rollback()
        log.error("RBAC table setup failed: %s", exc)
        raise
    finally:
        conn.close()


def seed_first_admin() -> None:
    """Create the initial admin account if no users exist at all."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rbac_users")
            count = cur.fetchone()[0]

        if count > 0:
            return

        admin_id = str(uuid.uuid4())
        pw_hash  = hash_password(FIRST_ADMIN_PASSWORD)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rbac_users
                    (user_id, email, full_name, password_hash, role, is_active)
                VALUES (%s, %s, %s, %s, 'admin', TRUE)
                ON CONFLICT (email) DO NOTHING
                """,
                (admin_id, FIRST_ADMIN_EMAIL, FIRST_ADMIN_NAME, pw_hash),
            )
        conn.commit()

        log.warning("=" * 60)
        log.warning("  FIRST ADMIN ACCOUNT CREATED")
        log.warning("  Email   : %s", FIRST_ADMIN_EMAIL)
        log.warning("  Password: %s", FIRST_ADMIN_PASSWORD)
        log.warning("  Change the password immediately after first login!")
        log.warning("=" * 60)

    except Exception as exc:
        conn.rollback()
        log.error("Admin seeding failed: %s", exc)
    finally:
        conn.close()
