"""
rbac_config.py — RBAC / auth configuration.

All RBAC settings live here, separate from the main config.py
so the existing pipeline is never disturbed.
"""

import os

# ── JWT ───────────────────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv(
    "RBAC_JWT_SECRET",
    "doc-intel-rbac-dev-secret-CHANGE-IN-PRODUCTION-2026",
)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60        # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS   = 7         # 7 days

# ── First admin (seeded on first startup if no users exist) ───────────────────
FIRST_ADMIN_EMAIL    = os.getenv("RBAC_ADMIN_EMAIL",    "admin@docintell.local")
FIRST_ADMIN_PASSWORD = os.getenv("RBAC_ADMIN_PASSWORD", "Admin@2026!")
FIRST_ADMIN_NAME     = "Platform Admin"

# ── Visibility rules ──────────────────────────────────────────────────────────
# Documents from these sources are visible to ALL authenticated users.
# manual_upload documents are only visible to their owner + admins.
ORG_SOURCE_TYPES = ("Outlook", "Sharepoint", "rdbms", "rdbms_flow")
