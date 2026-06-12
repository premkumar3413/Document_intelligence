"""
ai_classifier/llm_classifier.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zero-shot document classification using Azure OpenAI GPT-4o-mini.

Invoked ONLY when rule-based confidence < AUTO_THRESHOLD (70%).
(Zero-score documents are handled in the pipeline before reaching this.)

WHAT CHANGED vs previous version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX 1 — Module-level AzureOpenAI singleton
  OLD: A new AzureOpenAI(...) client was constructed inside classify_with_llm()
       on every classification call, rebuilding the HTTP connection pool
       every time.
  NEW: A single _llm_client is created at module import time and reused
       for the lifetime of the process.

FIX 2 — Tenacity retry on transient Azure OpenAI errors
  OLD: Any API error (including transient 429 / 503) was caught and
       silently returned None — causing the pipeline to fall back to
       rule-only classification.
  NEW: The API call is wrapped with a tenacity retry decorator:
         • Up to 3 attempts.
         • Exponential back-off: 2 s → 4 s → 8 s (max 10 s per wait).
         • Retries on RateLimitError (429) and APIStatusError (5xx).
         • Non-retryable errors (AuthenticationError, BadRequestError)
           surface immediately.
       After all retries are exhausted the exception is caught by the
       outer try/except and None is returned — preserving the existing
       fallback behaviour.

FIX 3 — Miscellaneous added as option 6 in the system prompt (previous fix,
       preserved unchanged).
  Previously: AI was FORCED to pick from the 5 categories even when
  the document was clearly an HR doc, IT report, financial statement,
  procurement document, or policy manual.
  Now: AI can return "Miscellaneous" when the document clearly does not
  belong to any of the 5 active categories. The pipeline then auto-routes
  the document to DIC/Miscellaneous/ without any HITL.

Everything else — prompt, _validate_and_clamp(), _empty_result(),
classify_with_llm() signature — is unchanged.
"""

import json
import logging
from typing import Optional

from openai import AzureOpenAI, RateLimitError, APIStatusError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import (
    OPENAI_ENDPOINT,
    OPENAI_KEY,
    OPENAI_API_VER,
    AI_CLASSIFICATION_MODEL,
    AI_MAX_INPUT_CHARS,
    AI_TEMPERATURE,
    ACTIVE_CATEGORIES,
)

log = logging.getLogger(__name__)

MISCELLANEOUS = "Miscellaneous"

# Valid categories the AI can return — 5 active + Miscellaneous
_VALID_AI_CATEGORIES = ACTIVE_CATEGORIES + [MISCELLANEOUS]


# ══════════════════════════════════════════════════════════════════
#  FIX 1 — Module-level AzureOpenAI singleton
#  Created once at import time; reused for every classification call.
# ══════════════════════════════════════════════════════════════════

_llm_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version=OPENAI_API_VER,
)

log.info("AzureOpenAI client (llm_classifier) initialised — endpoint=%s", OPENAI_ENDPOINT)


# ══════════════════════════════════════════════════════════════════
#  FIX 2 — Retry configuration
#  Retries on RateLimitError (429), APIStatusError (5xx),
#  and APIConnectionError (transient network issues).
# ══════════════════════════════════════════════════════════════════

_RETRY_EXCEPTIONS = (RateLimitError, APIStatusError, APIConnectionError)

_llm_retry = retry(
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)


# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert enterprise document classifier.

Classify the document into EXACTLY ONE of the following SIX options:

1. Contracts      — Legal agreements, NDAs, SOWs, MSAs, leases, MOUs,
                    service agreements, joint ventures, retainers
2. Compliance     — Audit reports, regulatory assessments, GDPR/HIPAA/SOX,
                    risk assessments, AML/KYC, corrective action plans
3. Governance     — Board resolutions, meeting minutes, shareholder documents,
                    AGM records, corporate charters, bylaws, proxy documents
4. Certifications — Professional certificates, accreditations, diplomas,
                    training completions, credentials, transcripts
5. Invoices       — Bills, invoices, purchase orders, receipts, payment
                    requests, credit notes, remittance advices
6. Miscellaneous  — The document CLEARLY does NOT belong to categories 1–5.
                    Use this when the document is an HR document, IT/security
                    report, financial statement, procurement document, policy
                    manual, personal document, or any other type outside the
                    five categories above.

DECISION RULES:
- If the document fits one of categories 1–5, choose that category.
- If the document clearly does NOT fit any of categories 1–5, choose Miscellaneous.
- Do NOT force-fit a document into a wrong category. Miscellaneous is the
  correct answer when the document is genuinely outside the five categories.

CONFIDENCE SCORE GUIDELINES:
  85–100 : Absolutely certain — unmistakably this category
  70–84  : Very confident — strong signals throughout
  50–69  : Moderately confident — clear signals, some ambiguity
  30–49  : Uncertain — mixed signals or limited content
  0–29   : Very uncertain — barely any signals present

For Miscellaneous: confidence reflects how certain you are that the document
does NOT belong to any of the five categories (not how well it fits Misc).

Respond ONLY with valid JSON — no other text:
{
  "predicted_category": "<exact name from options 1–6 above>",
  "confidence_score": <integer 0–100>,
  "reasoning": "<max 80 words citing specific evidence from the document>"
}"""


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_with_llm(text: str, filename: str) -> Optional[dict]:
    """
    Classify a document using GPT-4o-mini (zero-shot, 5 categories + Miscellaneous).

    Args:
        text:     cleaned document text (first AI_MAX_INPUT_CHARS chars sent)
        filename: original filename — strong classification signal

    Returns:
        {
            "predicted_category": str,   # one of 5 categories OR "Miscellaneous"
            "confidence_score":   float, # 0.0 – 100.0
            "reasoning":          str,
        }
        Returns None on API failure after all retries.
    """
    if not text.strip():
        log.warning("  AI classifier: empty text — returning low-confidence fallback")
        return _empty_result()

    excerpt = text[:AI_MAX_INPUT_CHARS]
    user_prompt = (
        f"Filename: {filename}\n\n"
        f"Document content (first {len(excerpt):,} characters):\n"
        f"{'─' * 60}\n"
        f"{excerpt}\n"
        f"{'─' * 60}\n\n"
        "Classify this document. Respond only with the JSON object."
    )

    try:
        raw    = _classify_with_retry(user_prompt)
        result = json.loads(raw)
        result = _validate_and_clamp(result)

        log.info(
            f"  AI: {result['predicted_category']} "
            f"({result['confidence_score']:.1f}%) — {result['reasoning'][:60]}…"
        )
        return result

    except json.JSONDecodeError as exc:
        log.error(f"  AI classifier: JSON parse error — {exc}")
        return None
    except Exception as exc:
        log.error(f"  AI classifier: API error after retries — {exc}")
        return None


@_llm_retry
def _classify_with_retry(user_prompt: str) -> str:
    """
    Inner function that makes the actual GPT-4o-mini API call.
    Decorated with retry so transient 429/5xx errors are automatically retried.
    Returns the raw response content string.
    """
    response = _llm_client.chat.completions.create(
        model=AI_CLASSIFICATION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=AI_TEMPERATURE,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_and_clamp(result: dict) -> dict:
    """Validate and sanitise the model's JSON response."""
    cat = result.get("predicted_category", "")
    if cat not in _VALID_AI_CATEGORIES:
        log.warning(f"  AI returned unknown category '{cat}' — defaulting to low confidence")
        result["predicted_category"] = ACTIVE_CATEGORIES[0]   # "Contracts" as safe fallback
        result["confidence_score"]   = max(0.0, float(result.get("confidence_score", 10)) - 20)

    try:
        result["confidence_score"] = float(
            max(0.0, min(100.0, float(result.get("confidence_score", 0))))
        )
    except (TypeError, ValueError):
        result["confidence_score"] = 0.0

    result.setdefault("reasoning", "No reasoning provided.")
    return result


def _empty_result() -> dict:
    """Fallback for documents with no extractable text."""
    return {
        "predicted_category": ACTIVE_CATEGORIES[0],
        "confidence_score":   0.0,
        "reasoning":          "No text extracted — classification based on filename only.",
    }