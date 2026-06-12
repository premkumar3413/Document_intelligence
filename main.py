"""
main.py — Single entry point for the Document Intelligence API.

Starts two routers on one FastAPI application:
    /api/files/*              → Duplicate detection (port 8000 by default)
    /api/classify/*
    /api/classification/*     → Classification pipeline

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Project structure:
    main.py                   ← you are here
    config.py                 ← all configuration constants
    database.py               ← get_connection()
    models.py                 ← Pydantic request/response schemas
    routers/
        duplicates.py         ← /api/files/* endpoints
        classification.py     ← /api/classify/* and /api/classification/*
    utils/
        blob.py               ← Azure Blob Storage helpers
        text_extractor.py     ← PDF/DOCX/XLSX/… text extraction
    classifier/
        rule_engine.py        ← JSON-driven rule-based classifier
    rules/
        ruleBasedConditions.json ← classification rules (10 categories)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import LOG_FORMAT, LOG_LEVEL
from routers import duplicates, classification
from routers.upload  import router as upload_router, ensure_document_metadata_table
from routers.meta    import router as meta_router
from routers.search  import router as search_router
# ── RBAC (new — existing routers above are untouched) ─────────────────────────
from rbac.db_init             import ensure_rbac_tables, seed_first_admin
from rbac.notification_worker import notification_worker_loop
from routers.auth             import router as auth_router
from routers.rbac_users       import router as rbac_users_router
from routers.rbac_documents   import router as rbac_docs_router
from routers.rbac_notifications import router as rbac_notif_router
from routers.rbac_audit       import router as rbac_audit_router


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format=LOG_FORMAT)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import logging as _log
    _lg = _log.getLogger("startup")
    try:
        ensure_document_metadata_table()
    except Exception as e:
        _lg.error("document_metadata setup failed (DB unreachable?): %s", e)
    try:
        ensure_rbac_tables()
        seed_first_admin()
    except Exception as e:
        _lg.error("RBAC table setup failed (DB unreachable?): %s", e)
    task = asyncio.create_task(notification_worker_loop())
    yield
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Intelligence API",
    lifespan=lifespan,
    description=(
        "Metadata-based duplicate detection + rule-based document classification. "
        "10 document categories driven by ruleBasedConditions.json."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:3000",
        # add your deployed UI origin here in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
        "X-API-Key",
    ],
    expose_headers=["Content-Disposition"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(duplicates.router,     prefix="/api")
app.include_router(classification.router, prefix="/api")
app.include_router(upload_router,         prefix="/api")
app.include_router(meta_router,           prefix="/api")
app.include_router(search_router,         prefix="/api")
# RBAC routers — all new, no conflict with existing routes
app.include_router(auth_router)
app.include_router(rbac_users_router)
app.include_router(rbac_docs_router)
app.include_router(rbac_notif_router)
app.include_router(rbac_audit_router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "Document Intelligence API", "version": "2.0.0"}


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)