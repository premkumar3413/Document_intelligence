"""
models.py — Pydantic schemas for all API request bodies and typed responses.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Duplicate service ────────────────────────────────────────────────────────

class IngestCheckRequest(BaseModel):
    """Body for POST /api/ingest/check-duplicate."""
    file_name: str = Field(..., description="Original file name as stored in document_metadata")
    file_path: str = Field(..., description="Full file path as stored in document_metadata")


# ─── Classification service ───────────────────────────────────────────────────

class HITLDecisionRequest(BaseModel):
    """
    Body for PUT /api/classification/{classification_id}/review.

    human_decision:
        True  → reviewer approves system suggestion (or uses override_category)
        False → reviewer rejects; document is moved to Miscellaneous

    override_category:
        Optional. When True and the reviewer disagrees with the suggested category,
        pass the correct category name here.

    reviewed_by:
        Name or email of the human reviewer (for audit trail).
    """
    human_decision:    bool            = Field(..., description="True=approve, False=reject→Miscellaneous")
    reviewed_by:       str             = Field(..., description="Reviewer name or email")
    override_category: Optional[str]   = Field(None, description="Override suggested category (optional)")