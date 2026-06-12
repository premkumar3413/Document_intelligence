
"""
routers/classification.py — Document classification endpoints.

Updated pipeline — complete Miscellaneous handling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Five classification outcomes:
  auto_classified    — combined score >= 70%, moved to DIC/{category}/
  soft_flag          — 40% <= combined < 70%, HITL recommended
  hard_stop          — 0% < combined < 40%, HITL mandatory
  auto_miscellaneous — rule_score == 0 OR AI returned Miscellaneous,
                       moved to DIC/Miscellaneous/ automatically, NO HITL
  human_rejected     — reviewer rejected, moved to DIC/Miscellaneous/

Miscellaneous routing rules (NEW):
  1. rule_score == 0.0 (all category scores = 0):
       → auto_miscellaneous immediately (skip AI, skip HITL)
       → reason: "Zero confidence: no signals detected"
  2. AI returns "Miscellaneous":
       → auto_miscellaneous (skip score combination, skip HITL)
       → reason: "AI classifier: " + ai_reasoning
  3. human_decision = False (human rejects):
       → human_rejected, moved to DIC/Miscellaneous/
       → classification_notes records reviewer decision

All auto_miscellaneous cases are recorded in classification_docs with
classification_notes explaining exactly why the document was routed there.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor

from config import AUTO_THRESHOLD, SOFT_FLAG_THRESHOLD
from database import get_connection
from models import HITLDecisionRequest
from utils.blob import download_source_blob, upload_to_dic, build_dic_blob_key
from utils.text_extractor import extract_text
from utils.text_cleaner import clean_text
from classifier.rule_engine import classifier, MISCELLANEOUS
from ai_classifier.llm_classifier import classify_with_llm
from ai_classifier.score_combiner import combine_scores
from ai_classifier.embedding_manager import embed_and_store

log = logging.getLogger(__name__)
router = APIRouter(tags=["Classification"])


# ════════════════════════════════════════════════════════════════════
#  DB PERSISTENCE
# ════════════════════════════════════════════════════════════════════

def _insert_classification_doc(
    conn,
    document_id:           str,
    rule_score:            float,
    rule_category:         str,
    ai_result:             Optional[dict],
    combined_score:        float,
    final_category:        str,
    final_status:          str,
    output_key:            Optional[str],
    class_type:            str,
    classification_tier:   str,
    all_rule_scores:       dict,
    classification_notes:  Optional[str] = None,
) -> str:
    """Insert one classification result. Returns classification_id."""
    ai_category  = ai_result["predicted_category"] if ai_result else None
    ai_score     = ai_result["confidence_score"]   if ai_result else None
    ai_reasoning = ai_result["reasoning"]          if ai_result else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO classification_docs (
                document_id,
                rule_confidence_score,
                ai_predicted_category,
                ai_confidence_score,
                ai_reasoning,
                combined_confidence_score,
                confidence_score,
                document_type,
                suggested_document_type,
                all_scores,
                output_document_path,
                type_of_classification,
                classification_status,
                classification_tier,
                classification_notes
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING classification_id
            """,
            (
                document_id,
                rule_score,
                ai_category,
                ai_score,
                ai_reasoning,
                combined_score,
                combined_score,
                final_category,
                final_category,
                json.dumps(all_rule_scores),
                output_key,
                class_type,
                final_status,
                classification_tier,
                classification_notes,
            ),
        )
        return str(cur.fetchone()[0])


# ════════════════════════════════════════════════════════════════════
#  MISCELLANEOUS REASON BUILDER
# ════════════════════════════════════════════════════════════════════

def _build_zero_score_reason(text_was_empty: bool, matched_rules: list) -> str:
    """Build a specific, auditable reason for zero-score auto_miscellaneous routing."""
    if text_was_empty and not matched_rules:
        return (
            "Auto-routed to Miscellaneous: zero confidence score. "
            "No text was extracted from the document (possible scanned/image-based PDF) "
            "and no filename pattern matched any of the 5 active categories. "
            "Cannot classify without readable content."
        )
    elif text_was_empty and matched_rules:
        # This shouldn't happen — if matched_rules, score would be > 0
        return (
            "Auto-routed to Miscellaneous: zero confidence despite filename match. "
            "Routing reason: no text content available for confirmation."
        )
    else:
        return (
            "Auto-routed to Miscellaneous: zero confidence score. "
            "Document text was readable but contained no keywords, sentences, "
            "or filename patterns matching any of the 5 active categories "
            "(Contracts, Compliance, Governance, Certifications, Invoices). "
            "Document likely belongs to a category outside the active set."
        )


def _build_ai_miscellaneous_reason(ai_result: dict) -> str:
    """Build the audit reason when AI returns Miscellaneous."""
    return (
        f"Auto-routed to Miscellaneous: AI classifier determined document does not "
        f"belong to any of the 5 active categories. "
        f"AI confidence: {ai_result['confidence_score']:.1f}%. "
        f"AI reasoning: {ai_result['reasoning']}"
    )


# ════════════════════════════════════════════════════════════════════
#  CORE PIPELINE
# ════════════════════════════════════════════════════════════════════

def _process_record(conn, record: dict) -> dict:
    """
    Full two-stage hybrid pipeline with complete Miscellaneous handling.
    """
    file_path   = record["file_path"]
    filename    = record["file_name"]
    source      = record["source_type"]
    document_id = record["document_id"]

    log.info(f"\n── {filename} ({source})")

    # ── Step 1: Download ──────────────────────────────────────────────────────
    file_bytes = download_source_blob(file_path)
    log.info(f"  Size      : {len(file_bytes):,} bytes")

    # ── Step 2: Extract text ──────────────────────────────────────────────────
    raw_text = extract_text(file_bytes, filename)
    log.info(f"  Extracted : {len(raw_text):,} chars")

    # ── Step 3: Clean text ────────────────────────────────────────────────────
    cleaned_text = clean_text(raw_text)
    log.info(f"  Cleaned   : {len(cleaned_text):,} chars")

    text_was_empty = not cleaned_text.strip()
    if text_was_empty:
        log.warning(
            f"  No text extracted from '{filename}'. "
            "Scanned PDF? Classification relies on filename patterns only."
        )

    # ── Step 4: Rule-based classification ─────────────────────────────────────
    rule_result   = classifier.classify(cleaned_text, filename)
    rule_score    = rule_result["confidence_score"]
    rule_category = rule_result["best_category"]
    all_scores    = rule_result["all_scores"]

    # ── Step 5: Scanned PDF upgrade ───────────────────────────────────────────
    if (text_was_empty and rule_result["matched_rules"]
            and rule_result["classification_status"] == "hard_stop"):
        rule_result["classification_status"] = "soft_flag"
        log.info(
            f"  Upgraded hard_stop → soft_flag: filename matched '{rule_category}' "
            "despite empty text."
        )

    # Initialise variables
    ai_result           = None
    combined_score      = rule_score
    final_category      = rule_category
    final_status        = rule_result["classification_status"]
    classification_tier = "rule_only"
    output_key          = None
    classification_notes: Optional[str] = None

    # ══════════════════════════════════════════════════════════════
    #  CASE 0: ZERO SCORE → AUTO-MISCELLANEOUS (no AI, no HITL)
    # ══════════════════════════════════════════════════════════════
    if rule_score == 0.0:
        # All category scores are zero — no signals whatsoever matched.
        # The "winner" from max() would be arbitrary (first key in dict).
        # Do NOT route to HITL — there is zero basis for a classification.
        # Auto-route to Miscellaneous immediately.
        final_category  = MISCELLANEOUS
        final_status    = "auto_miscellaneous"
        combined_score  = 0.0
        class_type      = "auto_miscellaneous"
        classification_notes = _build_zero_score_reason(
            text_was_empty, rule_result["matched_rules"]
        )
        output_key = build_dic_blob_key(MISCELLANEOUS, file_path)
        upload_to_dic(file_bytes, output_key)
        log.info(
            f"  → AUTO MISCELLANEOUS (zero score): {classification_notes[:80]}…"
        )

    # ══════════════════════════════════════════════════════════════
    #  CASE A: RULE SCORE >= 70% → AUTO (no AI needed)
    # ══════════════════════════════════════════════════════════════
    elif rule_score >= AUTO_THRESHOLD:
        class_type = "rule_based"
        output_key = build_dic_blob_key(final_category, file_path)
        upload_to_dic(file_bytes, output_key)
        log.info(f"  ✓ AUTO (rule): {final_category} ({rule_score:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    #  CASE B: RULE SCORE < 70% → INVOKE AI
    # ══════════════════════════════════════════════════════════════
    else:
        classification_tier = "rule_and_ai"
        log.info(f"  Rule {rule_score:.1f}% < {AUTO_THRESHOLD}% — invoking AI…")

        ai_result = classify_with_llm(cleaned_text, filename)

        if ai_result is None:
            # AI API failed → fall back to rule score only
            log.warning("  AI failed — using rule-only fallback")
            class_type   = "rule_based + human_in_the_loop"
            final_status = "soft_flag" if combined_score >= SOFT_FLAG_THRESHOLD else "hard_stop"

        elif ai_result["predicted_category"] == MISCELLANEOUS:
            # ── AI returned Miscellaneous → auto-route (no HITL) ─────────────
            final_category       = MISCELLANEOUS
            final_status         = "auto_miscellaneous"
            combined_score       = ai_result["confidence_score"]
            class_type           = "rule_based + ai"
            classification_notes = _build_ai_miscellaneous_reason(ai_result)
            output_key           = build_dic_blob_key(MISCELLANEOUS, file_path)
            upload_to_dic(file_bytes, output_key)
            log.info(
                f"  → AUTO MISCELLANEOUS (AI decision): "
                f"{ai_result['reasoning'][:80]}…"
            )

        else:
            # ── Normal AI path: combine scores, apply thresholds ──────────────
            combined = combine_scores(
                rule_score    = rule_score,
                rule_category = rule_category,
                ai_score      = ai_result["confidence_score"],
                ai_category   = ai_result["predicted_category"],
            )
            combined_score = combined["combined_score"]
            final_category = combined["final_category"]
            final_status   = combined["status"]

            if final_status == "auto_classified":
                class_type = "rule_based + ai"
                output_key = build_dic_blob_key(final_category, file_path)
                upload_to_dic(file_bytes, output_key)
                log.info(f"  ✓ AUTO (rule+AI): {final_category} ({combined_score:.1f}%)")
            elif final_status == "soft_flag":
                class_type = "rule_based + ai + human_in_the_loop"
                log.info(f"  ⚠ SOFT FLAG: {final_category} ({combined_score:.1f}%)")
            else:  # hard_stop
                class_type = "rule_based + ai + human_in_the_loop"
                log.info(f"  ✗ HARD STOP: {final_category} ({combined_score:.1f}%)")

    # ── Persist classification result ──────────────────────────────────────────
    cls_id = _insert_classification_doc(
        conn                 = conn,
        document_id          = document_id,
        rule_score           = rule_score,
        rule_category        = rule_category,
        ai_result            = ai_result,
        combined_score       = combined_score,
        final_category       = final_category,
        final_status         = final_status,
        output_key           = output_key,
        class_type           = class_type,
        classification_tier  = classification_tier,
        all_rule_scores      = all_scores,
        classification_notes = classification_notes,
    )

    # ── Generate embeddings for finalized documents ────────────────────────────
    # auto_classified and auto_miscellaneous → generate (embed_and_store skips Misc)
    # soft_flag and hard_stop → not finalized yet, embed in human_review()
    embedding_result = {"status": "not_applicable"}
    if final_status in ("auto_classified", "auto_miscellaneous"):
        log.info(f"  Embedding: generating for '{final_category}'…")
        embedding_result = embed_and_store(
            conn              = conn,
            document_id       = document_id,
            classification_id = cls_id,
            file_bytes        = file_bytes,   # raw bytes — embedding_manager extracts internally
            filename          = filename,
            final_category    = final_category,
        )

    conn.commit()

    return {
        "classification_id":         cls_id,
        "document_id":               document_id,
        "file_name":                 filename,
        "source_type":               source,
        "classification_tier":       classification_tier,
        "rule_category":             rule_category,
        "rule_confidence_score":     rule_score,
        "ai_predicted_category":     ai_result["predicted_category"] if ai_result else None,
        "ai_confidence_score":       ai_result["confidence_score"]   if ai_result else None,
        "ai_reasoning":              ai_result["reasoning"]          if ai_result else None,
        "final_category":            final_category,
        "combined_confidence_score": combined_score,
        "classification_status":     final_status,
        "output_document_path":      output_key,
        "text_was_empty":            text_was_empty,
        "classification_notes":      classification_notes,
        "embedding":                 embedding_result,
    }


# ════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════════════════

@router.post("/classify/batch")
def classify_batch():
    """Classify all unclassified original records."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT  dm.source_type, dm.file_name, dm.file_path,
                        dm.document_id, dm.is_duplicate
                FROM    document_metadata dm
                LEFT JOIN classification_docs cd ON cd.document_id = dm.document_id
                WHERE   dm.is_duplicate       = FALSE
                  AND   dm.processing_status != 'Deleted'
                  AND   cd.document_id        IS NULL
                ORDER   BY dm.current_timestamp_ist ASC
                """
            )
            pending = cur.fetchall()

        if not pending:
            return {"message": "No unclassified original records found.", "processed": 0}

        log.info(f"Batch: {len(pending)} record(s) to classify")
        results, succeeded, failed = [], 0, 0

        for row in pending:
            try:
                res = _process_record(conn, dict(row))
                results.append(res)
                succeeded += 1
            except Exception as exc:
                log.error(f"Failed: {row['file_name']} — {exc}")
                conn.rollback()
                failed += 1

        status_counts = {}
        for r in results:
            s = r["classification_status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        return {
            "total_processed":    succeeded,
            "failed":             failed,
            "auto_classified":    status_counts.get("auto_classified", 0),
            "auto_miscellaneous": status_counts.get("auto_miscellaneous", 0),
            "soft_flag":          status_counts.get("soft_flag", 0),
            "hard_stop":          status_counts.get("hard_stop", 0),
            "rule_only":          sum(1 for r in results if r["classification_tier"] == "rule_only"),
            "ai_assisted":        sum(1 for r in results if r["classification_tier"] == "rule_and_ai"),
            "results":            results,
        }
    finally:
        conn.close()


@router.post("/classify/{document_id}")
def classify_single(document_id: str):
    """Classify a single document by its document_id."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT source_type, file_name, file_path, document_id, is_duplicate
                FROM   document_metadata
                WHERE  document_id = %s AND is_duplicate = FALSE
                """,
                (document_id,),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Record not found or is a Duplicate.")
        return _process_record(conn, dict(row))
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.get("/classification/pending-review")
def get_pending_review(status_filter: Optional[str] = None):
    """
    Return documents awaiting human review (soft_flag + hard_stop).
    auto_miscellaneous documents are NOT returned here — they are already
    finalized and do not require human intervention.
    """
    conn = get_connection()
    try:
        if status_filter in ("soft_flag", "hard_stop"):
            status_condition = "cd.classification_status = %s"
            params = (status_filter,)
        else:
            status_condition = "cd.classification_status IN ('soft_flag', 'hard_stop')"
            params = ()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    cd.classification_id,
                    cd.document_id,
                    dm.source_type,
                    dm.file_name,
                    dm.file_path,
                    cd.document_type              AS suggested_category,
                    cd.rule_confidence_score,
                    cd.ai_predicted_category,
                    cd.ai_confidence_score,
                    cd.ai_reasoning,
                    cd.combined_confidence_score  AS confidence_score,
                    cd.classification_status,
                    cd.classification_tier,
                    cd.all_scores,
                    cd.current_timestamp_ist
                FROM    classification_docs cd
                JOIN    document_metadata   dm ON dm.document_id = cd.document_id
                WHERE   {status_condition}
                ORDER   BY cd.combined_confidence_score DESC
                """,
                params,
            )
            items = [dict(r) for r in cur.fetchall()]

        for item in items:
            if item["classification_status"] == "hard_stop":
                item["review_urgency"] = "mandatory"
                item["review_guidance"] = (
                    f"Confidence {item['confidence_score']:.1f}% is below 40%. "
                    "Use override_category to manually assign the correct category. "
                    "Valid categories: Contracts, Compliance, Governance, "
                    "Certifications, Invoices, Miscellaneous."
                )
            else:
                item["review_urgency"] = "recommended"
                item["review_guidance"] = (
                    f"Confidence {item['confidence_score']:.1f}% (between 40-70%). "
                    f"System suggests '{item['suggested_category']}'. "
                    "Confirm with human_decision=true or supply override_category."
                )

        return {
            "total_pending": len(items),
            "soft_flag":     sum(1 for r in items if r["classification_status"] == "soft_flag"),
            "hard_stop":     sum(1 for r in items if r["classification_status"] == "hard_stop"),
            "items":         items,
        }
    finally:
        conn.close()


@router.get("/classification/unfinalized")
def get_unfinalized(
    document_type:         Optional[str] = None,
    classification_status: Optional[str] = None,
    source_type:           Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
):
    """Pending documents — suggested category assigned but not yet moved to DIC."""
    conn = get_connection()
    try:
        where_clauses = [
            "cd.classification_status IN ('soft_flag', 'hard_stop')",
            "cd.document_type IS NOT NULL",
            "cd.output_document_path IS NULL",
        ]
        params: list = []

        if document_type:
            where_clauses.append("cd.document_type = %s"); params.append(document_type)
        if classification_status and classification_status in ("soft_flag", "hard_stop"):
            where_clauses.append("cd.classification_status = %s"); params.append(classification_status)
        if source_type:
            where_clauses.append("dm.source_type = %s"); params.append(source_type)

        where_sql = " AND ".join(where_clauses)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    cd.classification_id, cd.document_id,
                    dm.source_type, dm.file_name, dm.file_path, dm.is_duplicate,
                    cd.rule_confidence_score,
                    cd.ai_predicted_category, cd.ai_confidence_score,
                    cd.combined_confidence_score  AS confidence_score,
                    cd.document_type              AS suggested_category,
                    cd.classification_status,
                    cd.classification_tier,
                    cd.current_timestamp_ist
                FROM   classification_docs cd
                JOIN   document_metadata   dm ON dm.document_id = cd.document_id
                WHERE  {where_sql}
                ORDER  BY cd.current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params + [limit, offset],
            )
            items = [dict(r) for r in cur.fetchall()]

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM classification_docs cd "
                f"JOIN document_metadata dm ON dm.document_id = cd.document_id "
                f"WHERE {where_sql}", params,
            )
            total = cur.fetchone()[0]

        return {
            "total":     total,
            "soft_flag": sum(1 for r in items if r["classification_status"] == "soft_flag"),
            "hard_stop": sum(1 for r in items if r["classification_status"] == "hard_stop"),
            "limit": limit, "offset": offset, "items": items,
        }
    finally:
        conn.close()


@router.put("/classification/{classification_id}/review")
def human_review(classification_id: str, body: HITLDecisionRequest):
    """
    Human reviewer decision for soft_flag / hard_stop documents.
    human_decision=True  → approve category (or override)
    human_decision=False → reject → DIC/Miscellaneous/ with classification_notes
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cd.*, dm.file_path, dm.file_name, dm.source_type,
                       dm.document_id
                FROM   classification_docs cd
                JOIN   document_metadata   dm ON dm.document_id = cd.document_id
                WHERE  cd.classification_id = %s
                  AND  cd.classification_status IN ('soft_flag', 'hard_stop')
                """,
                (classification_id,),
            )
            rec = cur.fetchone()

        if not rec:
            raise HTTPException(
                status_code=404,
                detail="Record not found or not in reviewable state (soft_flag/hard_stop).",
            )

        file_bytes = download_source_blob(rec["file_path"])

        if body.human_decision:
            final_category = (
                body.override_category
                if body.override_category and body.override_category in classifier.valid_categories
                else rec["document_type"]
            )
            new_status    = "human_approved"
            review_notes  = None
        else:
            final_category = MISCELLANEOUS
            new_status     = "human_rejected"
            review_notes   = (
                f"Rejected by reviewer '{body.reviewed_by}'. "
                f"System had suggested '{rec['document_type']}'. "
                "Routed to Miscellaneous by reviewer decision."
            )

        output_key = build_dic_blob_key(final_category, rec["file_path"])
        upload_to_dic(file_bytes, output_key)

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE classification_docs
                SET    document_type          = %s,
                       output_document_path   = %s,
                       classification_status  = %s,
                       human_decision         = %s,
                       reviewed_by            = %s,
                       reviewed_at            = (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata'),
                       classification_notes   = %s
                WHERE  classification_id = %s
                """,
                (
                    final_category, output_key, new_status,
                    body.human_decision, body.reviewed_by,
                    review_notes,
                    classification_id,
                ),
            )

        # Scenario 3: generate embeddings after human approval
        # file_bytes already downloaded above — embedding_manager handles extraction
        document_id = str(rec["document_id"])
        log.info(f"  Scenario 3 embedding: '{final_category}' (status={new_status})")
        embedding_result = embed_and_store(
            conn              = conn,
            document_id       = document_id,
            classification_id = classification_id,
            file_bytes        = file_bytes,
            filename          = rec["file_name"],
            final_category    = final_category,
        )

        conn.commit()

        log.info(
            f"HITL #{classification_id}: {new_status} → {final_category} "
            f"({body.reviewed_by}) | embedding={embedding_result['status']}"
        )

        return {
            "classification_id":     classification_id,
            "file_name":             rec["file_name"],
            "final_category":        final_category,
            "classification_status": new_status,
            "output_document_path":  output_key,
            "reviewed_by":           body.reviewed_by,
            "classification_notes":  review_notes,
            "embedding":             embedding_result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@router.get("/classification/results")
def get_results(
    document_type:         Optional[str] = None,
    classification_status: Optional[str] = None,
    classification_tier:   Optional[str] = None,
    source_type:           Optional[str] = None,
    limit:  int = 100,
    offset: int = 0,
):
    """Finalized documents only (auto_classified, auto_miscellaneous, human_approved, human_rejected)."""
    conn = get_connection()
    try:
        where_clauses = [
            "cd.classification_status IN ('auto_classified','auto_miscellaneous','human_approved','human_rejected')"
        ]
        params: list = []

        if document_type:
            where_clauses.append("cd.document_type = %s"); params.append(document_type)
        if classification_status:
            where_clauses.append("cd.classification_status = %s"); params.append(classification_status)
        if classification_tier:
            where_clauses.append("cd.classification_tier = %s"); params.append(classification_tier)
        if source_type:
            where_clauses.append("dm.source_type = %s"); params.append(source_type)

        where_sql = " AND ".join(where_clauses)

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    cd.classification_id, cd.document_id,
                    dm.source_type, dm.file_name, dm.file_path, dm.is_duplicate,
                    cd.rule_confidence_score,
                    cd.ai_predicted_category, cd.ai_confidence_score, cd.ai_reasoning,
                    cd.combined_confidence_score  AS confidence_score,
                    cd.document_type, cd.output_document_path,
                    cd.type_of_classification, cd.classification_status,
                    cd.classification_tier, cd.classification_notes,
                    cd.reviewed_by, cd.reviewed_at, cd.current_timestamp_ist
                FROM   classification_docs cd
                JOIN   document_metadata   dm ON dm.document_id = cd.document_id
                WHERE  {where_sql}
                ORDER  BY cd.current_timestamp_ist DESC
                LIMIT  %s OFFSET %s
                """,
                params + [limit, offset],
            )
            files = [dict(r) for r in cur.fetchall()]

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM classification_docs cd "
                f"JOIN document_metadata dm ON dm.document_id = cd.document_id "
                f"WHERE {where_sql}", params,
            )
            total = cur.fetchone()[0]

        return {"total": total, "limit": limit, "offset": offset, "results": files}
    finally:
        conn.close()


@router.get("/classification/summary")
def get_classification_summary():
    """Dashboard counts — by status, tier, and document type."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)                                                                 AS total_classified,
                    COUNT(*) FILTER (WHERE classification_status = 'auto_classified')        AS auto_classified,
                    COUNT(*) FILTER (WHERE classification_status = 'auto_miscellaneous')     AS auto_miscellaneous,
                    COUNT(*) FILTER (WHERE classification_status = 'soft_flag')              AS soft_flag,
                    COUNT(*) FILTER (WHERE classification_status = 'hard_stop')              AS hard_stop,
                    COUNT(*) FILTER (WHERE classification_status = 'human_approved')         AS human_approved,
                    COUNT(*) FILTER (WHERE classification_status = 'human_rejected')         AS human_rejected,
                    COUNT(*) FILTER (WHERE classification_tier   = 'rule_only')              AS rule_only,
                    COUNT(*) FILTER (WHERE classification_tier   = 'rule_and_ai')            AS ai_assisted,
                    COUNT(*) FILTER (WHERE document_type = 'Contracts')                      AS contracts,
                    COUNT(*) FILTER (WHERE document_type = 'Compliance')                     AS compliance,
                    COUNT(*) FILTER (WHERE document_type = 'Governance')                     AS governance,
                    COUNT(*) FILTER (WHERE document_type = 'Certifications')                 AS certifications,
                    COUNT(*) FILTER (WHERE document_type = 'Invoices')                       AS invoices,
                    COUNT(*) FILTER (WHERE document_type = 'Miscellaneous')                  AS miscellaneous
                FROM classification_docs
                """
            )
            return dict(cur.fetchone())
    finally:
        conn.close()

@router.get("/classification/queue-count")
def get_queue_count():
    """Exact count of original documents that have no classification record yet."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS queue_count
                FROM   document_metadata dm
                LEFT JOIN LATERAL (
                    SELECT 1 FROM classification_docs
                    WHERE  document_id = dm.document_id LIMIT 1
                ) cd ON TRUE
                WHERE  dm.is_duplicate       = FALSE
                  AND  dm.processing_status != 'Deleted'
                  AND  cd IS NULL
                """
            )
            return {"queue_count": cur.fetchone()[0]}
    finally:
        conn.close()