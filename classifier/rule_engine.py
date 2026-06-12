
"""
classifier/rule_engine.py — Rule-based document classifier.

Changes in this version vs previous:
  FIX 1 (word boundaries) — filename patterns now all have \b boundaries.
         Patterns without \b were applied in ruleBasedConditions.json.
         E.g. "(?i)billing" could previously match "prebilling_report.pdf" —
         now only "(?i)\\bbilling\\b" matches files where 'billing' is a
         distinct word.
  FIX 2 (tie-breaking) — when two categories share the same top score,
         the winner is now chosen deterministically:
           1. Prefer category with the most matched signals
           2. If still tied, prefer category with a filename pattern match
           3. If still tied, prefer category with a sentence match
         Previously Python's max() returned the first key in dict order
         (always "Contracts"), which was arbitrary.

Signal weights and confidence formula unchanged:
  keyword   × 2 pts  |  sentence × 5 pts  |  filename × 10 pts
  confidence = min(raw_score / SCORE_BASELINE × 100, 100.0)

Three-tier decision thresholds (from config.py):
  >= AUTO_THRESHOLD (70%)          → auto_classified
  >= SOFT_FLAG_THRESHOLD (40%)     → soft_flag
  < SOFT_FLAG_THRESHOLD            → hard_stop
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

from config import (
    RULES_FILE,
    KEYWORD_WEIGHT,
    SENTENCE_WEIGHT,
    FILENAME_WEIGHT,
    SCORE_BASELINE,
    AUTO_THRESHOLD,
    SOFT_FLAG_THRESHOLD,
)

log = logging.getLogger(__name__)

MISCELLANEOUS = "Miscellaneous"


class DocumentClassifier:
    """
    Scores documents against all categories using keyword, sentence,
    and filename pattern matching. Always runs all three signal types
    even when document text is empty (scanned PDFs can still match
    via filename patterns).
    """

    def __init__(self, rules_path: Optional[str] = None):
        path = Path(rules_path or RULES_FILE)
        if not path.is_absolute():
            path = Path(__file__).parent.parent / path

        with open(path, encoding="utf-8") as f:
            self._rules: dict = json.load(f)

        self.categories: list[str]       = list(self._rules.keys())
        self.valid_categories: list[str] = self.categories + [MISCELLANEOUS]

        log.info(
            f"DocumentClassifier loaded {len(self.categories)} categories "
            f"from {path.name}: {self.categories}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str, filename: str) -> dict:
        """
        Score the document against all categories and return the result.

        Always called even when text is empty — filename patterns run
        regardless of whether text was extracted.

        Args:
            text:     cleaned document text (may be empty for scanned PDFs)
            filename: original filename including extension

        Returns dict with:
            best_category, confidence_score, classification_status,
            all_scores, matched_rules
        """
        text_lower = text.lower()

        # Preprocess filename: replace _ and - with space so that
        # \b word boundaries work correctly.
        # "invoice_10514.pdf" → "invoice 10514.pdf" → \binvoice\b matches
        filename_for_patterns = re.sub(r'[_\-]', ' ', filename)

        all_scores  : dict[str, float] = {}
        all_matched : dict[str, list]  = {}

        for category, rules in self._rules.items():
            score, matched = self._score_category(
                text_lower, filename, filename_for_patterns, rules, category
            )
            all_scores[category]  = score
            all_matched[category] = matched

        # ── Winner selection with tie-breaking (FIX 2) ────────────────────────
        winner = self._pick_winner(all_scores, all_matched)
        confidence = all_scores[winner]
        status     = self._decide_status(confidence)

        if all_matched[winner]:
            log.info(
                f"  ▶ {winner} ({confidence:.1f}%) [{status}] — "
                f"matched: {all_matched[winner]}"
            )
        else:
            log.info(
                f"  ▶ {winner} ({confidence:.1f}%) [{status}] — "
                f"no signals matched for any category"
            )
            log.info(
                f"  Filename preprocessed: '{filename}' → '{filename_for_patterns}'"
            )
            if not text_lower.strip():
                log.warning(
                    "  Text empty — only filename patterns were evaluated. "
                    "Scanned PDF requires OCR for keyword/sentence matching."
                )

        return {
            "best_category":         winner,
            "confidence_score":      confidence,
            "classification_status": status,
            "all_scores":            all_scores,
            "matched_rules":         all_matched[winner],
        }

    # ── Tie-breaking (FIX 2) ─────────────────────────────────────────────────

    def _pick_winner(
        self, all_scores: dict[str, float], all_matched: dict[str, list]
    ) -> str:
        """
        Select the winning category deterministically.

        When scores are tied:
          1. Prefer category with the most matched signals.
          2. If still tied, prefer the one with a filename pattern match
             (highest weight signal — strongest evidence).
          3. If still tied, prefer the one with a sentence match.
          4. If all else is equal, use category order in JSON.
        """
        if not all_scores:
            return self.categories[0] if self.categories else MISCELLANEOUS

        top_score = max(all_scores.values())

        # Categories tied at the top score
        tied = [c for c, s in all_scores.items() if s == top_score]

        if len(tied) == 1:
            return tied[0]

        # All scores are 0 — no signal matched at all
        if top_score == 0:
            # Return first category by JSON order; status will be hard_stop anyway
            return tied[0]

        # Multiple categories tied at a non-zero score — apply tie-breaking
        def tie_key(cat: str) -> tuple:
            m = all_matched[cat]
            return (
                len(m),                                             # more total signals
                sum(1 for x in m if x.startswith("fname:")),       # filename matches
                sum(1 for x in m if x.startswith("sent:")),        # sentence matches
                sum(1 for x in m if x.startswith("kw:")),          # keyword matches
            )

        winner = max(tied, key=tie_key)

        if len(tied) > 1:
            log.info(
                f"  Tie-breaking: {[f'{c}={all_scores[c]}%' for c in tied]} "
                f"→ winner: {winner} (most/highest-quality signals)"
            )

        return winner

    # ── Three-tier decision ────────────────────────────────────────────────────

    def _decide_status(self, confidence: float) -> str:
        """
        Map a confidence score to a three-tier workflow status.

            >= AUTO_THRESHOLD (70%)          → auto_classified
            >= SOFT_FLAG_THRESHOLD (40%)     → soft_flag
            < SOFT_FLAG_THRESHOLD (40%)      → hard_stop
        """
        if confidence >= AUTO_THRESHOLD:
            return "auto_classified"
        elif confidence >= SOFT_FLAG_THRESHOLD:
            return "soft_flag"
        else:
            return "hard_stop"

    # ── Scoring internals ─────────────────────────────────────────────────────

    def _score_category(
        self,
        text_lower:            str,
        filename_raw:          str,
        filename_preprocessed: str,
        rules:                 dict,
        category:              str,
    ) -> tuple[float, list[str]]:
        """
        Score a single category. Returns (confidence 0–100, matched labels).

        filename_raw:          original filename (for log labels)
        filename_preprocessed: underscores/hyphens replaced with spaces
                               so \b word boundaries work correctly
        """
        raw_score = 0
        matched   = []

        # ── Keywords (checked against document text) ─────────────────────────
        for kw in rules.get("keywords", []):
            kw_lower = kw.lower()
            if ' ' in kw_lower:
                # Multi-word phrase: substring match
                if kw_lower in text_lower:
                    raw_score += KEYWORD_WEIGHT
                    matched.append(f"kw:{kw}")
            else:
                # Single word: word boundary to avoid matching inside other words
                # e.g. "audit" should not match inside "auditorium"
                try:
                    if re.search(r'\b' + re.escape(kw_lower) + r'\b', text_lower):
                        raw_score += KEYWORD_WEIGHT
                        matched.append(f"kw:{kw}")
                except re.error:
                    if kw_lower in text_lower:
                        raw_score += KEYWORD_WEIGHT
                        matched.append(f"kw:{kw}")

        # ── Sentences (checked against document text) ─────────────────────────
        for sentence in rules.get("sentences", []):
            if sentence.lower() in text_lower:
                raw_score += SENTENCE_WEIGHT
                matched.append(f"sent:{sentence[:60]}")

        # ── Filename patterns ─────────────────────────────────────────────────
        # ALWAYS runs, even when text is empty.
        # Uses preprocessed filename (underscores → spaces) for correct \b matching.
        for pattern in rules.get("filename_patterns", []):
            try:
                if re.search(pattern, filename_preprocessed, re.IGNORECASE):
                    raw_score += FILENAME_WEIGHT
                    matched.append(f"fname:{pattern}")
            except re.error as exc:
                log.warning(
                    f"  Invalid regex in {category} filename_patterns: "
                    f"'{pattern}' — {exc}"
                )

        confidence = (
            min((raw_score / SCORE_BASELINE) * 100, 100.0)
            if SCORE_BASELINE > 0 else 0.0
        )
        return round(confidence, 2), matched


# Module-level singleton — loaded once at startup
classifier = DocumentClassifier()