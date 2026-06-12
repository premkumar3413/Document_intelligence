"""
utils/text_cleaner.py — Basic text cleaning for rule-based classification.

Applied AFTER text extraction and BEFORE rule-based classification.
Scope: basic normalisation only — no OCR character correction,
no spell-checking, no semantic changes.

Cleaning steps (in order):
  1. Decode fix      — replace common encoding artefacts (Ã©→é, â€™→', etc.)
  2. Null/control    — remove null bytes and non-printable control characters
  3. Hyphen repair   — rejoin words split across lines (word-\nword → wordword)
  4. Whitespace      — collapse multiple spaces/tabs to a single space
  5. Line cleanup    — trim each line, collapse 3+ blank lines to 2
  6. Quote/dash norm — curly quotes → straight quotes; em-dash → hyphen
  7. Strip           — remove leading/trailing whitespace from final result

What is deliberately NOT done:
  - OCR character substitution (0→o, 1→l, 3→e) — risks corrupting codes/numbers
  - Spell-checking — risks changing domain-specific legal/financial terms
  - Lowercasing — caller does this for keyword matching, not here
  - Punctuation removal — punctuation is needed for sentence matching
"""

import re
import unicodedata
import logging

log = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Apply all basic cleaning steps to extracted document text.

    Args:
        text: raw text returned by text_extractor.extract_text()

    Returns:
        Cleaned text string ready for rule-based classification.
    """
    if not text:
        return text

    original_len = len(text)

    # ── Step 1: Fix common UTF-8 encoding artefacts ───────────────────────────
    # Handles documents that were incorrectly decoded somewhere in the pipeline.
    text = _fix_encoding_artefacts(text)

    # ── Step 2: Remove null bytes and non-printable control characters ────────
    # Keep: \t (tab), \n (newline), \r (carriage return)
    # Remove: \x00–\x08, \x0b–\x0c, \x0e–\x1f, \x7f
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # ── Step 3: Repair hyphenated word breaks across lines ────────────────────
    # "contrac-\ntual" → "contractual"
    # Only applies when a word character precedes the hyphen at line end.
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)

    # ── Step 4: Normalise horizontal whitespace ───────────────────────────────
    # Collapse multiple spaces/tabs within a line to a single space.
    text = re.sub(r'[ \t]+', ' ', text)

    # ── Step 5: Clean up individual lines ────────────────────────────────────
    lines = text.split('\n')
    lines = [line.strip() for line in lines]

    # Collapse 3+ consecutive blank lines to 2 blank lines (preserve paragraphs)
    cleaned_lines = []
    consecutive_blanks = 0
    for line in lines:
        if line == '':
            consecutive_blanks += 1
            if consecutive_blanks <= 2:
                cleaned_lines.append(line)
        else:
            consecutive_blanks = 0
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    # ── Step 6: Normalise quotes and dashes ───────────────────────────────────
    # Curly/smart quotes → straight quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # '' → '
    text = text.replace('\u201c', '"').replace('\u201d', '"')   # "" → "
    # Em-dash and en-dash → hyphen-minus (preserves meaning, aids keyword match)
    text = text.replace('\u2014', '-').replace('\u2013', '-')   # — – → -
    # Ellipsis character → three dots
    text = text.replace('\u2026', '...')

    # ── Step 7: Final strip ───────────────────────────────────────────────────
    text = text.strip()

    cleaned_len = len(text)
    if original_len > 0:
        reduction = (original_len - cleaned_len) / original_len * 100
        log.debug(
            f"  Text cleaned: {original_len:,} → {cleaned_len:,} chars "
            f"({reduction:.1f}% reduction)"
        )

    return text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fix_encoding_artefacts(text: str) -> str:
    """
    Fix common encoding artefacts produced by misconfigured PDF extractors
    or incorrect charset assumptions.

    Examples of what gets fixed:
        Ã©  → é      (UTF-8 byte 0xC3 0xA9 decoded as Latin-1)
        â€™ → '      (U+2019 RIGHT SINGLE QUOTATION MARK decoded as Latin-1)
        Â   → (space) (spurious Â before non-ASCII characters)
    """
    try:
        # Try to fix Latin-1 mis-decoded UTF-8
        encoded = text.encode('latin-1', errors='ignore')
        decoded = encoded.decode('utf-8', errors='ignore')
        # Only use the re-decoded version if it looks more like normal text
        if decoded and _score_text_quality(decoded) > _score_text_quality(text):
            text = decoded
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass  # original text is fine

    # Remove spurious Â characters that appear before non-ASCII chars
    text = re.sub(r'Â(?=\s|[^\x00-\x7F])', '', text)

    # Normalise to NFC form (composed Unicode — standard for text processing)
    try:
        text = unicodedata.normalize('NFC', text)
    except Exception:
        pass

    return text


def _score_text_quality(text: str) -> float:
    """
    Simple heuristic to estimate text quality.
    Higher score = more printable ASCII = better quality.
    Used to decide whether re-decoded text is an improvement.
    """
    if not text:
        return 0.0
    printable_ascii = sum(1 for c in text if 0x20 <= ord(c) <= 0x7e or c in '\n\r\t')
    return printable_ascii / len(text)