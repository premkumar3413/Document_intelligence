"""
ai_classifier/score_combiner.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines rule-based and AI-based classification scores into a
single final confidence score and decision status.

Agreed combination formula (established in architecture sessions):

  Categories AGREE (rule and AI predict the same category):
      combined = rule_score × RULE_WEIGHT  +  ai_score × AI_WEIGHT
             = rule_score × 0.35          +  ai_score × 0.65

      AI gets higher weight (65%) because it has semantic understanding
      beyond keyword matching. Rule engine weight (35%) reflects its
      value as a fast structural filter.

  Categories DISAGREE:
      combined = max(rule_score, ai_score) × DISAGREE_PENALTY
             = max(rule_score, ai_score) × 0.75

      When two independent classifiers point to different categories,
      genuine ambiguity exists. A 25% penalty reflects this uncertainty
      and pushes borderline cases toward human review.

  All weights are configurable in config.py.

Decision thresholds applied to combined score:
  combined >= AUTO_THRESHOLD (70%)          →  auto_classified
  SOFT_FLAG_THRESHOLD (40%) <= combined     →  soft_flag
  combined < SOFT_FLAG_THRESHOLD (40%)      →  hard_stop
"""

import logging
from config import (
    RULE_WEIGHT,
    AI_WEIGHT,
    DISAGREE_PENALTY,
    AUTO_THRESHOLD,
    SOFT_FLAG_THRESHOLD,
)

log = logging.getLogger(__name__)


def combine_scores(
    rule_score:    float,
    rule_category: str,
    ai_score:      float,
    ai_category:   str,
) -> dict:
    """
    Combine rule-based and AI-based scores into a final classification result.

    Args:
        rule_score:    confidence from rule engine (0–100)
        rule_category: category predicted by rule engine
        ai_score:      confidence from GPT-4o-mini (0–100)
        ai_category:   category predicted by GPT-4o-mini

    Returns dict:
        combined_score      float 0–100
        final_category      str
        status              "auto_classified" | "soft_flag" | "hard_stop"
        categories_agreed   bool
    """
    categories_agreed = (rule_category == ai_category)

    if categories_agreed:
        # Both engines agree → weighted combination
        combined   = rule_score * RULE_WEIGHT + ai_score * AI_WEIGHT
        final_cat  = ai_category
        log.info(
            f"  Scores (agree): rule={rule_score:.1f}%×{RULE_WEIGHT} + "
            f"ai={ai_score:.1f}%×{AI_WEIGHT} = {combined:.1f}%"
        )
    else:
        # Categories disagree → 25% penalty on the higher-confidence prediction
        if ai_score >= rule_score:
            combined  = ai_score * DISAGREE_PENALTY
            final_cat = ai_category
        else:
            combined  = rule_score * DISAGREE_PENALTY
            final_cat = rule_category

        log.info(
            f"  Scores (disagree): rule={rule_score:.1f}%/{rule_category} vs "
            f"ai={ai_score:.1f}%/{ai_category} "
            f"→ penalty {DISAGREE_PENALTY}× → {combined:.1f}%/{final_cat}"
        )

    combined = round(min(combined, 100.0), 2)

    if combined >= AUTO_THRESHOLD:
        status = "auto_classified"
    elif combined >= SOFT_FLAG_THRESHOLD:
        status = "soft_flag"
    else:
        status = "hard_stop"

    log.info(f"  Final: {final_cat} ({combined:.1f}%) → {status}")

    return {
        "combined_score":    combined,
        "final_category":    final_cat,
        "status":            status,
        "categories_agreed": categories_agreed,
    }