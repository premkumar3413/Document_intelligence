
# """
# utils/text_extractor.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Supported file types: PDF (.pdf) and Word (.docx) ONLY.

#   .doc → ValueError  (python-docx cannot read old binary .doc format)
#   All other formats → ValueError

# Two public functions
# ─────────────────────────────────────────────────────────────────
# extract_text(file_bytes, filename) → str
#     Plain text for the rule engine and AI classifier.
#     Structure is not needed here — just clean readable text.
#     Called in classification.py before rule/AI scoring.

# extract_chunks_for_embedding(file_bytes, filename) → list[dict]
#     Structure-aware chunks for the embedding pipeline.
#     Preserves paragraph boundaries, heading hierarchy, and table
#     formatting so each chunk represents ONE coherent idea.
#     Called inside embedding_manager.embed_and_store().

#     Each chunk dict:
#         {
#             "text":            str,           # original content (stored in DB)
#             "heading_context": str,           # "H1 > H2 > H3" or "" for PDFs
#             "chunk_type":      "text"|"table" # tables handled separately
#         }

# PDF extraction strategy
# ─────────────────────────────────────────────────────────────────
# Uses PyMuPDF 1.27+ block detection (page.get_text("blocks")) to
# preserve paragraph structure. Plain get_text() loses all structure.

#   Step 1 — Header/footer removal
#     Method A: Y-position — any block in the top/bottom 7% of page
#               height is treated as header/footer and skipped.
#     Method B: Cross-page — text blocks at the same Y-position on
#               pages 0 and 1 are repeated elements (company name,
#               document title, page numbers) and are skipped globally.
#     Standalone page numbers are also removed ("1", "Page 3", etc.)

#   Step 2 — Table detection (PyMuPDF 1.23+ find_tables)
#     Tables are formatted as "col1 | col2 | col3" rows.
#     Table bounding boxes are excluded from the text block pass
#     to avoid duplicate content.

#   Step 3 — Paragraph grouping
#     Text blocks are consecutive text within the same Y-position
#     flow. A gap > 8pt between blocks signals a paragraph boundary.
#     Hyphenated line-breaks are repaired (word-\nword → wordword).

# DOCX extraction strategy
# ─────────────────────────────────────────────────────────────────
# Uses python-docx paragraph.style.name to detect heading levels.
# Builds a heading stack and flushes accumulated paragraphs into a
# chunk when the heading changes.

#   Each chunk reads:
#     "Data Retention Policy > Archiving Rules: Personal data must
#      not be retained beyond 7 years..."

#   vs old flat approach:
#     "Personal data must not be retained beyond 7 years..."

#   The heading context dramatically improves embedding precision for
#   legal/compliance/governance documents.
# """

# import io
# import re
# import logging
# from pathlib import Path

# log = logging.getLogger(__name__)

# # Supported extensions
# _SUPPORTED = {".pdf", ".docx"}
# _DOC_ONLY  = {".doc"}


# # ══════════════════════════════════════════════════════════════════
# #  PUBLIC API
# # ══════════════════════════════════════════════════════════════════

# def extract_text(file_bytes: bytes, filename: str) -> str:
#     """
#     Extract plain text for classification (rule engine + AI classifier).

#     Returns a single string. Structure is not preserved — the rule
#     engine needs flat text for keyword/sentence matching.

#     Raises ValueError for .doc or unsupported file types so the caller
#     can return HTTP 400 immediately.
#     """
#     ext = Path(filename).suffix.lower()
#     _validate_extension(filename, ext)

#     try:
#         if ext == ".pdf":
#             return _extract_pdf_text(file_bytes, filename)
#         else:
#             return _extract_docx_text(file_bytes, filename)
#     except ValueError:
#         raise
#     except Exception as exc:
#         log.warning(f"  Text extraction failed ({filename}): {exc}")
#         return ""


# def extract_chunks_for_embedding(file_bytes: bytes, filename: str) -> list:
#     """
#     Extract structure-aware chunks for the embedding pipeline.

#     Returns a list of chunk dicts. Each dict has:
#         text            – chunk content (stored as chunk_text in DB)
#         heading_context – DOCX: "H1 > H2 > H3"; PDF: ""
#         chunk_type      – "text" or "table"

#     Chunks from this function may still exceed CHUNK_SIZE. The
#     embedding_manager._structure_split_chunks() further splits them at
#     paragraph/sentence boundaries and merges tiny ones.

#     Raises ValueError for unsupported file types.
#     """
#     ext = Path(filename).suffix.lower()
#     _validate_extension(filename, ext)

#     try:
#         if ext == ".pdf":
#             return _extract_pdf_chunks(file_bytes, filename)
#         else:
#             return _extract_docx_chunks(file_bytes, filename)
#     except ValueError:
#         raise
#     except Exception as exc:
#         log.warning(f"  Chunk extraction failed ({filename}): {exc}")
#         return []


# # ══════════════════════════════════════════════════════════════════
# #  VALIDATION
# # ══════════════════════════════════════════════════════════════════

# def _validate_extension(filename: str, ext: str) -> None:
#     if ext in _DOC_ONLY:
#         raise ValueError(
#             f"Unsupported file format: '{filename}'. "
#             "The .doc format (Word 97-2003) is not supported. "
#             "Please save the file as .docx (Word 2007 or later) and re-upload."
#         )
#     if ext not in _SUPPORTED:
#         raise ValueError(
#             f"Unsupported file format '{ext}' in '{filename}'. "
#             "Only PDF (.pdf) and Word (.docx) files are accepted."
#         )


# # ══════════════════════════════════════════════════════════════════
# #  PDF — PLAIN TEXT (for classification)
# # ══════════════════════════════════════════════════════════════════

# def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
#     """
#     Extract plain text from a PDF using PyMuPDF block detection.
#     Header/footer zones and standalone page numbers are skipped.
#     """
#     import fitz

#     doc          = fitz.open(stream=file_bytes, filetype="pdf")
#     hf_zones     = _detect_header_footer_zones(doc)
#     repeated     = _detect_repeated_blocks(doc)
#     pages_text   = []

#     for page in doc:
#         page_h   = page.rect.height
#         zone     = hf_zones.get(page.number, {})
#         hf_top   = zone.get("top",    page_h * 0.07)
#         hf_bot   = zone.get("bottom", page_h * 0.93)

#         blocks = page.get_text("blocks", sort=True)
#         lines  = []

#         for block in blocks:
#             x0, y0, x1, y1, text, *_ = block[:7]
#             block_type = block[6] if len(block) > 6 else 0
#             if block_type != 0:               # skip image blocks
#                 continue
#             if y1 <= hf_top or y0 >= hf_bot:  # header/footer zone
#                 continue
#             text = text.strip()
#             if not text:
#                 continue
#             if text in repeated:               # repeated cross-page block
#                 continue
#             if _is_page_number(text):          # standalone page number
#                 continue
#             # Fix hyphenated line breaks and collapse internal whitespace
#             text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
#             text = re.sub(r'\s+', ' ', text).strip()
#             lines.append(text)

#         if lines:
#             pages_text.append("\n".join(lines))

#     doc.close()
#     result = "\n\n".join(pages_text)
#     if not result.strip():
#         log.warning(f"  PDF has no embedded text (scanned?): {filename}")
#     return result


# # ══════════════════════════════════════════════════════════════════
# #  TABLE FORMATTING — Natural Language Sentences
# # ══════════════════════════════════════════════════════════════════

# def _format_table_as_sentences(table_data: list) -> str:
#     """
#     Convert a table (list-of-lists) into natural language sentences
#     for better cosine similarity with natural language queries.

#     WHY NOT PIPE-DELIMITED:
#     "Subtotal | $2050 | Tax (18%) | $369 | Grand Total | $2419"
#     lives in "structured-data symbol space" in the embedding model.
#     Queries like "total invoice amount including tax" live in
#     "conversational language space". Cosine similarity: ~0.44.

#     "Subtotal: $2,050. Tax 18%: $369. Grand Total: $2,419."
#     lives in natural language space — much closer to queries.
#     Expected cosine similarity: 0.62–0.72.

#     THREE TABLE PATTERNS HANDLED:

#     Pattern A — 2-column key-value (each row = label + value):
#       Input : [["Subtotal", "$2050"], ["Tax (18%)", "$369"], ...]
#       Output: "Subtotal: $2050. Tax (18%): $369. Grand Total: $2419."

#     Pattern B — Header row + data rows (≥3 cols, header detected):
#       Input : [["Item", "Qty", "Price", "Total"], ["Cloud Hosting", "2", "$500", "$1000"], ...]
#       Output: "Item: Cloud Hosting, Qty: 2, Price: $500, Total: $1000.
#                Item: AI Document Processing, Qty: 1, Price: $750, Total: $750."

#     Pattern C — Alternating label-value in wide rows (invoice metadata):
#       Input : [["Invoice Number:", "INV-2026-1001", "Invoice Date:", "24-May-2026"]]
#       Output: "Invoice Number: INV-2026-1001, Invoice Date: 24-May-2026."

#     Pattern D — Fallback: comma-joined values.
#     """
#     if not table_data:
#         return ""

#     # Clean cells; remove fully-empty rows
#     cleaned = []
#     for row in table_data:
#         r = [str(c).strip() if c is not None else "" for c in row]
#         if any(c for c in r):
#             cleaned.append(r)
#     if not cleaned:
#         return ""

#     def is_numeric_cell(c: str) -> bool:
#         """Return True if the cell looks like a number or currency."""
#         return bool(re.match(r'^\$?[\d,]+\.?\d*%?$', c.strip()))

#     def is_label_cell(c: str) -> bool:
#         """Return True if the cell is text (not purely numeric)."""
#         return bool(c) and not is_numeric_cell(c)

#     ncols     = max(len(r) for r in cleaned)
#     nrows     = len(cleaned)
#     sentences = []

#     def row_has_numerics(row):
#         return any(is_numeric_cell(c) for c in row if c)

#     def row_all_text(row):
#         non_empty = [c for c in row if c]
#         return bool(non_empty) and all(is_label_cell(c) for c in non_empty)

#     # ── Pattern A: 2-column table → "Label: Value." per row ───────────────────
#     if ncols == 2:
#         for row in cleaned:
#             label = row[0].rstrip(":").strip() if len(row) > 0 else ""
#             value = row[1].strip()              if len(row) > 1 else ""
#             if label and value:
#                 sentences.append(f"{label}: {value}.")
#             elif label:
#                 sentences.append(f"{label}.")
#             elif value:
#                 sentences.append(f"{value}.")

#     # ── Patterns B/C: Multi-column — detect by second-row content ─────────────
#     else:
#         first_all_text    = row_all_text(cleaned[0])
#         second_has_nums   = nrows > 1 and row_has_numerics(cleaned[1])

#         if first_all_text and second_has_nums:
#             # Pattern B: header row + data rows (e.g. line items table)
#             # Row 1 = column labels.  Row 2+ = data values (contain numbers).
#             headers   = cleaned[0]
#             data_rows = cleaned[1:]
#             for row in data_rows:
#                 pairs = []
#                 for i, value in enumerate(row):
#                     if not value:
#                         continue
#                     header = headers[i].rstrip(":").strip() if i < len(headers) else ""
#                     if header:
#                         pairs.append(f"{header}: {value}")
#                     else:
#                         pairs.append(value)
#                 if pairs:
#                     sentences.append(", ".join(pairs) + ".")

#         elif first_all_text and ncols % 2 == 0 and ncols >= 4:
#             # Pattern C: alternating label-value in each row
#             # e.g. ["Invoice Number:", "INV-001", "Invoice Date:", "24-May-2026"]
#             for row in cleaned:
#                 pairs = []
#                 for i in range(0, len(row) - 1, 2):
#                     lbl = row[i].rstrip(":").strip()
#                     val = row[i + 1].strip() if (i + 1) < len(row) else ""
#                     if lbl and val:
#                         pairs.append(f"{lbl}: {val}")
#                     elif val:
#                         pairs.append(val)
#                 if pairs:
#                     sentences.append(", ".join(pairs) + ".")

#         else:
#             # Pattern D: fallback — comma-joined values per row
#             for row in cleaned:
#                 vals = [c for c in row if c]
#                 if vals:
#                     sentences.append(", ".join(vals) + ".")

#     return " ".join(sentences) if sentences else ""


# # ══════════════════════════════════════════════════════════════════
# #  PDF — STRUCTURED CHUNKS (for embedding)
# # ══════════════════════════════════════════════════════════════════

# def _extract_pdf_chunks(file_bytes: bytes, filename: str) -> list:
#     """
#     Structure-aware PDF extraction returning paragraph and table chunks.

#     Pipeline per page:
#       1. Skip header/footer zones (Y-position + cross-page detection)
#       2. Detect tables → format as natural language sentences
#       3. Group remaining text blocks into paragraphs by Y-gap
#       4. Return all chunks ordered by page/position
#     """
#     import fitz

#     doc      = fitz.open(stream=file_bytes, filetype="pdf")
#     hf_zones = _detect_header_footer_zones(doc)
#     repeated = _detect_repeated_blocks(doc)
#     chunks   = []

#     for page in doc:
#         page_h = page.rect.height
#         zone   = hf_zones.get(page.number, {})
#         hf_top = zone.get("top",    page_h * 0.07)
#         hf_bot = zone.get("bottom", page_h * 0.93)

#         # ── Step 1: Detect tables ──────────────────────────────────────────────
#         table_chunks = []
#         table_bboxes = []
#         try:
#             tabs = page.find_tables()
#             for tab in tabs.tables:
#                 table_data = tab.extract()
#                 if table_data:
#                     formatted = _format_table_as_sentences(table_data)
#                     if formatted:
#                         table_chunks.append({
#                             "text":            formatted,
#                             "heading_context": "",
#                             "chunk_type":      "table",
#                             "_bbox":           tab.bbox,
#                         })
#                         table_bboxes.append(tab.bbox)
#         except Exception as exc:
#             log.debug(f"  Table detection on page {page.number}: {exc}")

#         # ── Step 2: Extract text blocks, skip header/footer and table areas ────
#         blocks          = page.get_text("blocks", sort=True)
#         current_para    = []
#         prev_y1         = None
#         PARA_GAP_PT     = 8     # gap in points that signals a new paragraph

#         def flush_para():
#             """Emit accumulated lines as one paragraph chunk."""
#             if current_para:
#                 para_text = " ".join(current_para).strip()
#                 if para_text and not _is_page_number(para_text):
#                     chunks.append({
#                         "text":            para_text,
#                         "heading_context": "",
#                         "chunk_type":      "text",
#                     })
#             current_para.clear()

#         for block in blocks:
#             if len(block) < 7:
#                 continue
#             x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
#             text            = block[4].strip()
#             block_type      = block[6]

#             if block_type != 0:
#                 continue                                 # image block
#             if y1 <= hf_top or y0 >= hf_bot:
#                 continue                                 # header/footer zone
#             if not text or text in repeated:
#                 continue                                 # empty or cross-page repeat
#             if _is_page_number(text):
#                 continue                                 # standalone page number
#             if any(_bbox_overlap((x0, y0, x1, y1), tb) for tb in table_bboxes):
#                 continue                                 # inside a detected table

#             # Paragraph boundary: large Y-gap from previous block
#             if prev_y1 is not None and (y0 - prev_y1) > PARA_GAP_PT:
#                 flush_para()

#             # Fix hyphenated line breaks; collapse internal whitespace
#             text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
#             text = re.sub(r'\n', ' ', text)
#             text = re.sub(r'\s+', ' ', text).strip()
#             current_para.append(text)
#             prev_y1 = y1

#         flush_para()

#         # ── Step 3: Add table chunks for this page ─────────────────────────────
#         for tc in table_chunks:
#             chunks.append({k: v for k, v in tc.items() if k != "_bbox"})

#     doc.close()
#     log.info(
#         f"  PDF '{filename}': {len(chunks)} raw chunks "
#         f"({sum(1 for c in chunks if c['chunk_type']=='table')} tables)"
#     )
#     return chunks


# # ══════════════════════════════════════════════════════════════════
# #  PDF HELPERS
# # ══════════════════════════════════════════════════════════════════

# def _detect_header_footer_zones(doc) -> dict:
#     """
#     Return per-page header/footer Y-zones as {page_num: {"top": float, "bottom": float}}.
#     Uses the fixed-percentage method (7% of page height).
#     """
#     zones = {}
#     for page in doc:
#         h = page.rect.height
#         zones[page.number] = {
#             "top":    h * 0.07,
#             "bottom": h * 0.93,
#         }
#     return zones


# def _detect_repeated_blocks(doc) -> set:
#     """
#     Detect text blocks that appear at the same Y-position on pages 0 and 1.
#     These are company names, document titles, or running headers/footers.
#     Returns a set of text strings to skip on all pages.
#     """
#     repeated = set()
#     if len(doc) < 2:
#         return repeated

#     def blocks_by_y(page_num):
#         page   = doc[page_num]
#         result = {}
#         for b in page.get_text("blocks"):
#             y0, text = round(b[1], -1), b[4].strip()
#             if text and not _is_page_number(text):
#                 result[y0] = text
#         return result

#     try:
#         p0 = blocks_by_y(0)
#         p1 = blocks_by_y(1)
#         for y_key, text in p0.items():
#             if p1.get(y_key) == text:
#                 repeated.add(text)
#     except Exception:
#         pass

#     return repeated


# def _is_page_number(text: str) -> bool:
#     """Return True if text is a standalone page number."""
#     text = text.strip()
#     patterns = [
#         r'^\d+$',                  # "1"  "23"
#         r'^[Pp]age\s+\d+$',       # "Page 1"
#         r'^[-–]\s*\d+\s*[-–]$',   # "- 3 -"
#         r'^\d+\s*/\s*\d+$',       # "1/10"
#         r'^[Pp]g\.?\s*\d+$',      # "Pg. 5"
#     ]
#     return any(re.match(p, text) for p in patterns)


# def _bbox_overlap(bbox1: tuple, bbox2) -> bool:
#     """Return True if two bounding boxes overlap."""
#     try:
#         x0a, y0a, x1a, y1a = bbox1
#         x0b = float(bbox2[0]); y0b = float(bbox2[1])
#         x1b = float(bbox2[2]); y1b = float(bbox2[3])
#         return not (x1a <= x0b or x0a >= x1b or y1a <= y0b or y0a >= y1b)
#     except Exception:
#         return False


# # ══════════════════════════════════════════════════════════════════
# #  DOCX — PLAIN TEXT (for classification)
# # ══════════════════════════════════════════════════════════════════

# def _extract_docx_text(file_bytes: bytes, filename: str) -> str:
#     """
#     Extract plain text from a DOCX document for classification.
#     Headings and paragraphs are included as flat text.
#     """
#     from docx import Document

#     doc   = Document(io.BytesIO(file_bytes))
#     lines = []
#     for para in doc.paragraphs:
#         text = para.text.strip()
#         if text:
#             lines.append(text)
#     return "\n".join(lines)


# # ══════════════════════════════════════════════════════════════════
# #  DOCX — STRUCTURED CHUNKS (for embedding)
# # ══════════════════════════════════════════════════════════════════

# def _extract_docx_chunks(file_bytes: bytes, filename: str) -> list:
#     """
#     Section-aware DOCX extraction with heading hierarchy context.

#     Algorithm:
#       1. Iterate paragraphs in document order
#       2. Track heading stack as headings are encountered
#       3. Accumulate normal paragraphs under the current heading
#       4. When a new heading is encountered, flush the accumulated
#          paragraphs as one chunk (with heading context prepended)
#       5. Flush final section at end of document

#     Result: each chunk represents one coherent document section.
#     The heading context ("Policy > Section 3 > Sub-clause") is
#     stored separately and prepended only at embedding time.
#     """
#     from docx import Document

#     doc           = Document(io.BytesIO(file_bytes))
#     chunks        = []
#     heading_stack = []      # current [H1_text, H2_text, H3_text, ...]
#     current_paras = []      # body paragraphs accumulated under current heading

#     def flush_section():
#         if current_paras:
#             body    = "\n".join(current_paras).strip()
#             context = _build_heading_context(heading_stack)
#             if body:
#                 chunks.append({
#                     "text":            body,
#                     "heading_context": context,
#                     "chunk_type":      "text",
#                 })

#     for para in doc.paragraphs:
#         raw   = para.text.strip()
#         style = para.style.name if para.style else ""
#         level = _get_heading_level(style)

#         if level > 0:
#             # New heading — flush what came before
#             flush_section()
#             current_paras = []
#             # Trim heading stack to parent level, then push new heading
#             heading_stack = heading_stack[: level - 1]
#             heading_stack.append(raw)
#         else:
#             # Normal paragraph, list item, or body text
#             if raw:
#                 current_paras.append(raw)

#     flush_section()  # flush final section

#     log.info(f"  DOCX '{filename}': {len(chunks)} section chunks")
#     return chunks


# # ══════════════════════════════════════════════════════════════════
# #  DOCX HELPERS
# # ══════════════════════════════════════════════════════════════════

# def _get_heading_level(style_name: str) -> int:
#     """
#     Return heading level (1-6) for heading styles, 0 otherwise.

#     Handles: "Heading 1", "heading 2", "Title", "Subtitle", custom
#     styles that begin with "Heading ".
#     """
#     if not style_name:
#         return 0
#     s = style_name.lower().strip()

#     # Standard heading styles
#     m = re.match(r'heading\s+(\d+)', s)
#     if m:
#         return int(m.group(1))

#     # Title and subtitle treated as top-level headings
#     if s in ("title", "subtitle"):
#         return 1

#     return 0


# def _build_heading_context(stack: list) -> str:
#     """
#     Build a "H1 > H2 > H3" context string from the heading stack.
#     Limited to 3 levels and 120 characters total.

#     Examples:
#         ["Data Retention Policy", "Section 3 — Personnel Records"]
#         → "Data Retention Policy > Section 3 — Personnel Records"

#         ["Master Services Agreement", "Termination", "Clause 8.2 Notice Period"]
#         → "Master Services Agreement > Termination > Clause 8.2 Notice Period"
#     """
#     context = " > ".join(h for h in stack[:3] if h)
#     if len(context) > 120:
#         context = context[:117] + "..."
#     return context














"""
utils/text_extractor.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Supported file types: PDF (.pdf) and Word (.docx) ONLY.

  .doc → ValueError  (python-docx cannot read old binary .doc format)
  All other formats → ValueError

Two public functions
─────────────────────────────────────────────────────────────────
extract_text(file_bytes, filename) → str
    Plain text for the rule engine and AI classifier.
    Structure is not needed here — just clean readable text.
    Called in classification.py before rule/AI scoring.

extract_chunks_for_embedding(file_bytes, filename) → list[dict]
    Structure-aware chunks for the embedding pipeline.
    Preserves paragraph boundaries, heading hierarchy, and table
    formatting so each chunk represents ONE coherent idea.
    Called inside embedding_manager.embed_and_store().

    Each chunk dict:
        {
            "text":            str,           # original content (stored in DB)
            "heading_context": str,           # "H1 > H2 > H3" or "" for PDFs
            "chunk_type":      "text"|"table" # tables handled separately
        }

WHAT CHANGED vs previous version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX — _extract_docx_chunks() now processes doc.tables in addition
to doc.paragraphs.

OLD: _extract_docx_chunks() only iterated doc.paragraphs.
     Any DOCX document that stores content in Word tables (invoices,
     contracts with signature blocks, forms, metadata tables) would
     produce ZERO useful chunks — only free-text paragraphs between
     tables were extracted.

     Impact on the APX invoice (Invoice_APX-INV-2026-016044…docx):
     • The document contains 8 Word tables and exactly 1 free-text
       paragraph (the tax note footer).
     • The old extractor produced 1 chunk: the tax note.
     • Invoice number, date, due date, line items, and total were
       NEVER embedded.
     • Every retrieval query scored ~0.30 on the APX invoice because
       the only embedded chunk (tax note) contained no answer.

NEW: _extract_docx_chunks() uses _iter_docx_body_order() to walk
     paragraphs AND tables in their original document order.

     Table cell handling — the KEY fix for invoice-style tables:
     ─────────────────────────────────────────────────────────────
     Word invoice tables use a "label \\n value" pattern in each cell:
         | INVOICE NUMBER   | INVOICE DATE  | DUE DATE      |
         | APX-INV-2026-016 | 27 May 2026   | 26 Jun 2026   |

     python-docx reports this as a single cell with text
     "INVOICE NUMBER\\nAPX-INV-2026-016044".

     _normalize_table_for_formatting() detects this pattern and
     splits each such cell into [label, value] rows before passing
     to _format_table_as_sentences(), producing:

         "INVOICE NUMBER: APX-INV-2026-016044.
          INVOICE DATE: 27 May 2026.
          DUE DATE: 26 Jun 2026."

     This lives in natural-language embedding space and scores
     0.65–0.78 on queries like "What is the invoice number?" —
     compared to 0.30 for the unembedded tax note.

     Duplicate cell deduplication:
     ─────────────────────────────
     Word tables with merged cells repeat the same text in adjacent
     cells. _deduplicate_table_rows() removes consecutive duplicate
     rows before formatting to prevent "Invoice Number: INV-001.
     Invoice Number: INV-001." repetition.

PDF extraction strategy (unchanged)
─────────────────────────────────────────────────────────────────
Uses PyMuPDF 1.27+ block detection (page.get_text("blocks")) to
preserve paragraph structure. Plain get_text() loses all structure.

  Step 1 — Header/footer removal
    Method A: Y-position — any block in the top/bottom 7% of page
              height is treated as header/footer and skipped.
    Method B: Cross-page — text blocks at the same Y-position on
              pages 0 and 1 are repeated elements (company name,
              document title, page numbers) and are skipped globally.
    Standalone page numbers are also removed ("1", "Page 3", etc.)

  Step 2 — Table detection (PyMuPDF 1.23+ find_tables)
    Tables are formatted as "col1 | col2 | col3" rows.
    Table bounding boxes are excluded from the text block pass
    to avoid duplicate content.

  Step 3 — Paragraph grouping
    Text blocks are consecutive text within the same Y-position
    flow. A gap > 8pt between blocks signals a paragraph boundary.
    Hyphenated line-breaks are repaired (word-\\nword → wordword).

DOCX extraction strategy (updated)
─────────────────────────────────────────────────────────────────
Walks paragraphs AND tables in document body order.

  Paragraphs: heading detection unchanged — heading stack tracks
    H1/H2/H3 context; body paragraphs accumulate under current
    heading and are flushed as one chunk per section.

  Tables: each table is formatted via _format_table_as_sentences()
    and emitted as a "table" chunk with the current heading context.
    The table formatter handles four table patterns:

      Pattern A — 2-column key-value (label | value per row)
      Pattern B — Header row + data rows (line items)
      Pattern C — Alternating label-value in wide rows
      Pattern D — Fallback: comma-joined values

    For invoice-style "stacked" cells (label\\nvalue in one cell),
    _normalize_table_for_formatting() splits them first so Pattern A
    applies correctly.
"""

import io
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Supported extensions
_SUPPORTED = {".pdf", ".docx"}
_DOC_ONLY  = {".doc"}


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text for classification (rule engine + AI classifier).

    Returns a single string. Structure is not preserved — the rule
    engine needs flat text for keyword/sentence matching.

    Raises ValueError for .doc or unsupported file types so the caller
    can return HTTP 400 immediately.
    """
    ext = Path(filename).suffix.lower()
    _validate_extension(filename, ext)

    try:
        if ext == ".pdf":
            return _extract_pdf_text(file_bytes, filename)
        else:
            return _extract_docx_text(file_bytes, filename)
    except ValueError:
        raise
    except Exception as exc:
        log.warning(f"  Text extraction failed ({filename}): {exc}")
        return ""


def extract_chunks_for_embedding(file_bytes: bytes, filename: str) -> list:
    """
    Extract structure-aware chunks for the embedding pipeline.

    Returns a list of chunk dicts. Each dict has:
        text            – chunk content (stored as chunk_text in DB)
        heading_context – DOCX: "H1 > H2 > H3"; PDF: ""
        chunk_type      – "text" or "table"

    Chunks from this function may still exceed CHUNK_SIZE. The
    embedding_manager._structure_split_chunks() further splits them at
    paragraph/sentence boundaries and merges tiny ones.

    Raises ValueError for unsupported file types.
    """
    ext = Path(filename).suffix.lower()
    _validate_extension(filename, ext)

    try:
        if ext == ".pdf":
            return _extract_pdf_chunks(file_bytes, filename)
        else:
            return _extract_docx_chunks(file_bytes, filename)
    except ValueError:
        raise
    except Exception as exc:
        log.warning(f"  Chunk extraction failed ({filename}): {exc}")
        return []


# ══════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════

def _validate_extension(filename: str, ext: str) -> None:
    if ext in _DOC_ONLY:
        raise ValueError(
            f"Unsupported file format: '{filename}'. "
            "The .doc format (Word 97-2003) is not supported. "
            "Please save the file as .docx (Word 2007 or later) and re-upload."
        )
    if ext not in _SUPPORTED:
        raise ValueError(
            f"Unsupported file format '{ext}' in '{filename}'. "
            "Only PDF (.pdf) and Word (.docx) files are accepted."
        )


# ══════════════════════════════════════════════════════════════════
#  TABLE FORMATTING — Natural Language Sentences
# ══════════════════════════════════════════════════════════════════

def _format_table_as_sentences(table_data: list) -> str:
    """
    Convert a table (list-of-lists) into natural language sentences
    for better cosine similarity with natural language queries.

    WHY NOT PIPE-DELIMITED:
    "Subtotal | $2050 | Tax (18%) | $369 | Grand Total | $2419"
    lives in "structured-data symbol space" in the embedding model.
    Queries like "total invoice amount including tax" live in
    "conversational language space". Cosine similarity: ~0.44.

    "Subtotal: $2,050. Tax 18%: $369. Grand Total: $2,419."
    lives in natural language space — much closer to queries.
    Expected cosine similarity: 0.62–0.72.

    THREE TABLE PATTERNS HANDLED:

    Pattern A — 2-column key-value (each row = label + value):
      Input : [["Subtotal", "$2050"], ["Tax (18%)", "$369"], ...]
      Output: "Subtotal: $2050. Tax (18%): $369. Grand Total: $2419."

    Pattern B — Header row + data rows (≥3 cols, header detected):
      Input : [["Item", "Qty", "Price", "Total"], ["Cloud Hosting", "2", "$500", "$1000"], ...]
      Output: "Item: Cloud Hosting, Qty: 2, Price: $500, Total: $1000.
               Item: AI Document Processing, Qty: 1, Price: $750, Total: $750."

    Pattern C — Alternating label-value in wide rows (invoice metadata):
      Input : [["Invoice Number:", "INV-2026-1001", "Invoice Date:", "24-May-2026"]]
      Output: "Invoice Number: INV-2026-1001, Invoice Date: 24-May-2026."

    Pattern D — Fallback: comma-joined values.
    """
    if not table_data:
        return ""

    # Clean cells; remove fully-empty rows
    cleaned = []
    for row in table_data:
        r = [str(c).strip() if c is not None else "" for c in row]
        if any(c for c in r):
            cleaned.append(r)
    if not cleaned:
        return ""

    def is_numeric_cell(c: str) -> bool:
        """Return True if the cell looks like a number or currency."""
        return bool(re.match(r'^\$?[\d,]+\.?\d*%?$', c.strip()))

    def is_label_cell(c: str) -> bool:
        """Return True if the cell is text (not purely numeric)."""
        return bool(c) and not is_numeric_cell(c)

    ncols     = max(len(r) for r in cleaned)
    nrows     = len(cleaned)
    sentences = []

    def row_has_numerics(row):
        return any(is_numeric_cell(c) for c in row if c)

    def row_all_text(row):
        non_empty = [c for c in row if c]
        return bool(non_empty) and all(is_label_cell(c) for c in non_empty)

    # ── Pattern A: 2-column table → "Label: Value." per row ───────────────────
    if ncols == 2:
        for row in cleaned:
            label = row[0].rstrip(":").strip() if len(row) > 0 else ""
            value = row[1].strip()              if len(row) > 1 else ""
            if label and value:
                sentences.append(f"{label}: {value}.")
            elif label:
                sentences.append(f"{label}.")
            elif value:
                sentences.append(f"{value}.")

    # ── Patterns B/C: Multi-column — detect by second-row content ─────────────
    else:
        first_all_text    = row_all_text(cleaned[0])
        second_has_nums   = nrows > 1 and row_has_numerics(cleaned[1])

        if first_all_text and second_has_nums:
            # Pattern B: header row + data rows (e.g. line items table)
            headers   = cleaned[0]
            data_rows = cleaned[1:]
            for row in data_rows:
                pairs = []
                for i, value in enumerate(row):
                    if not value:
                        continue
                    header = headers[i].rstrip(":").strip() if i < len(headers) else ""
                    if header:
                        pairs.append(f"{header}: {value}")
                    else:
                        pairs.append(value)
                if pairs:
                    sentences.append(", ".join(pairs) + ".")

        elif first_all_text and ncols % 2 == 0 and ncols >= 4:
            # Pattern C: alternating label-value in each row
            for row in cleaned:
                pairs = []
                for i in range(0, len(row) - 1, 2):
                    lbl = row[i].rstrip(":").strip()
                    val = row[i + 1].strip() if (i + 1) < len(row) else ""
                    if lbl and val:
                        pairs.append(f"{lbl}: {val}")
                    elif val:
                        pairs.append(val)
                if pairs:
                    sentences.append(", ".join(pairs) + ".")

        else:
            # Pattern D: fallback — comma-joined values per row
            for row in cleaned:
                vals = [c for c in row if c]
                if vals:
                    sentences.append(", ".join(vals) + ".")

    return " ".join(sentences) if sentences else ""


# ══════════════════════════════════════════════════════════════════
#  PDF — PLAIN TEXT (for classification)
# ══════════════════════════════════════════════════════════════════

def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from a PDF using PyMuPDF block detection.
    Header/footer zones and standalone page numbers are skipped.
    """
    import fitz

    doc          = fitz.open(stream=file_bytes, filetype="pdf")
    hf_zones     = _detect_header_footer_zones(doc)
    repeated     = _detect_repeated_blocks(doc)
    pages_text   = []

    for page in doc:
        page_h   = page.rect.height
        zone     = hf_zones.get(page.number, {})
        hf_top   = zone.get("top",    page_h * 0.07)
        hf_bot   = zone.get("bottom", page_h * 0.93)

        blocks = page.get_text("blocks", sort=True)
        lines  = []

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block[:7]
            block_type = block[6] if len(block) > 6 else 0
            if block_type != 0:               # skip image blocks
                continue
            if y1 <= hf_top or y0 >= hf_bot:  # header/footer zone
                continue
            text = text.strip()
            if not text:
                continue
            if text in repeated:               # repeated cross-page block
                continue
            if _is_page_number(text):          # standalone page number
                continue
            # Fix hyphenated line breaks and collapse internal whitespace
            text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
            text = re.sub(r'\s+', ' ', text).strip()
            lines.append(text)

        if lines:
            pages_text.append("\n".join(lines))

    doc.close()
    result = "\n\n".join(pages_text)
    if not result.strip():
        log.warning(f"  PDF has no embedded text (scanned?): {filename}")
    return result


# ══════════════════════════════════════════════════════════════════
#  PDF — STRUCTURED CHUNKS (for embedding)
# ══════════════════════════════════════════════════════════════════

def _extract_pdf_chunks(file_bytes: bytes, filename: str) -> list:
    """
    Structure-aware PDF extraction returning paragraph and table chunks.

    Pipeline per page:
      1. Skip header/footer zones (Y-position + cross-page detection)
      2. Detect tables → format as natural language sentences
      3. Group remaining text blocks into paragraphs by Y-gap
      4. Return all chunks ordered by page/position
    """
    import fitz

    doc      = fitz.open(stream=file_bytes, filetype="pdf")
    hf_zones = _detect_header_footer_zones(doc)
    repeated = _detect_repeated_blocks(doc)
    chunks   = []

    for page in doc:
        page_h = page.rect.height
        zone   = hf_zones.get(page.number, {})
        hf_top = zone.get("top",    page_h * 0.07)
        hf_bot = zone.get("bottom", page_h * 0.93)

        # ── Step 1: Detect tables ──────────────────────────────────────────────
        table_chunks = []
        table_bboxes = []
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                table_data = tab.extract()
                if table_data:
                    formatted = _format_table_as_sentences(table_data)
                    if formatted:
                        table_chunks.append({
                            "text":            formatted,
                            "heading_context": "",
                            "chunk_type":      "table",
                            "_bbox":           tab.bbox,
                        })
                        table_bboxes.append(tab.bbox)
        except Exception as exc:
            log.debug(f"  Table detection on page {page.number}: {exc}")

        # ── Step 2: Extract text blocks, skip header/footer and table areas ────
        blocks          = page.get_text("blocks", sort=True)
        current_para    = []
        prev_y1         = None
        PARA_GAP_PT     = 8     # gap in points that signals a new paragraph

        def flush_para():
            """Emit accumulated lines as one paragraph chunk."""
            if current_para:
                para_text = " ".join(current_para).strip()
                if para_text and not _is_page_number(para_text):
                    chunks.append({
                        "text":            para_text,
                        "heading_context": "",
                        "chunk_type":      "text",
                    })
            current_para.clear()

        for block in blocks:
            if len(block) < 7:
                continue
            x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
            text            = block[4].strip()
            block_type      = block[6]

            if block_type != 0:
                continue                                 # image block
            if y1 <= hf_top or y0 >= hf_bot:
                continue                                 # header/footer zone
            if not text or text in repeated:
                continue                                 # empty or cross-page repeat
            if _is_page_number(text):
                continue                                 # standalone page number
            if any(_bbox_overlap((x0, y0, x1, y1), tb) for tb in table_bboxes):
                continue                                 # inside a detected table

            # Paragraph boundary: large Y-gap from previous block
            if prev_y1 is not None and (y0 - prev_y1) > PARA_GAP_PT:
                flush_para()

            # Fix hyphenated line breaks; collapse internal whitespace
            text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            current_para.append(text)
            prev_y1 = y1

        flush_para()

        # ── Step 3: Add table chunks for this page ─────────────────────────────
        for tc in table_chunks:
            chunks.append({k: v for k, v in tc.items() if k != "_bbox"})

    doc.close()
    log.info(
        f"  PDF '{filename}': {len(chunks)} raw chunks "
        f"({sum(1 for c in chunks if c['chunk_type']=='table')} tables)"
    )
    return chunks


# ══════════════════════════════════════════════════════════════════
#  PDF HELPERS
# ══════════════════════════════════════════════════════════════════

def _detect_header_footer_zones(doc) -> dict:
    """
    Return per-page header/footer Y-zones as {page_num: {"top": float, "bottom": float}}.
    Uses the fixed-percentage method (7% of page height).
    """
    zones = {}
    for page in doc:
        h = page.rect.height
        zones[page.number] = {
            "top":    h * 0.07,
            "bottom": h * 0.93,
        }
    return zones


def _detect_repeated_blocks(doc) -> set:
    """
    Detect text blocks that appear at the same Y-position on pages 0 and 1.
    These are company names, document titles, or running headers/footers.
    Returns a set of text strings to skip on all pages.
    """
    repeated = set()
    if len(doc) < 2:
        return repeated

    def blocks_by_y(page_num):
        page   = doc[page_num]
        result = {}
        for b in page.get_text("blocks"):
            y0, text = round(b[1], -1), b[4].strip()
            if text and not _is_page_number(text):
                result[y0] = text
        return result

    try:
        p0 = blocks_by_y(0)
        p1 = blocks_by_y(1)
        for y_key, text in p0.items():
            if p1.get(y_key) == text:
                repeated.add(text)
    except Exception:
        pass

    return repeated


def _is_page_number(text: str) -> bool:
    """Return True if text is a standalone page number."""
    text = text.strip()
    patterns = [
        r'^\d+$',                  # "1"  "23"
        r'^[Pp]age\s+\d+$',       # "Page 1"
        r'^[-–]\s*\d+\s*[-–]$',   # "- 3 -"
        r'^\d+\s*/\s*\d+$',       # "1/10"
        r'^[Pp]g\.?\s*\d+$',      # "Pg. 5"
    ]
    return any(re.match(p, text) for p in patterns)


def _bbox_overlap(bbox1: tuple, bbox2) -> bool:
    """Return True if two bounding boxes overlap."""
    try:
        x0a, y0a, x1a, y1a = bbox1
        x0b = float(bbox2[0]); y0b = float(bbox2[1])
        x1b = float(bbox2[2]); y1b = float(bbox2[3])
        return not (x1a <= x0b or x0a >= x1b or y1a <= y0b or y0a >= y1b)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
#  DOCX — PLAIN TEXT (for classification)
# ══════════════════════════════════════════════════════════════════

def _extract_docx_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from a DOCX document for classification.
    Includes both paragraph text AND table cell text as flat text.
    """
    from docx import Document

    doc   = Document(io.BytesIO(file_bytes))
    lines = []

    # Paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)

    # Tables — join all cell text as flat lines
    for table in doc.tables:
        seen_rows = set()
        for row in table.rows:
            cells = tuple(c.text.strip() for c in row.cells)
            if cells in seen_rows:
                continue
            seen_rows.add(cells)
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                lines.append(row_text)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  DOCX — STRUCTURED CHUNKS (for embedding)
# ══════════════════════════════════════════════════════════════════

def _iter_docx_body_order(doc):
    """
    Yield (element_type, element) in document body order, where
    element_type is "paragraph" or "table".

    python-docx's doc.paragraphs and doc.tables iterate their
    respective element types in isolation. This function walks the
    raw XML body so paragraphs and tables are yielded in the order
    they actually appear in the document — critical for preserving
    the correct heading context for table chunks.
    """
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    body = doc.element.body
    for child in body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield "paragraph", Paragraph(child, doc)
        elif tag == "tbl":
            yield "table", Table(child, doc)


def _normalize_table_for_formatting(table) -> list:
    """
    Convert a python-docx Table object into a list-of-lists suitable
    for _format_table_as_sentences().

    KEY FIX — handles "stacked" invoice cells (label\\nvalue in one cell):
    ─────────────────────────────────────────────────────────────────────
    Word invoice tables use this layout:
        Cell text = "INVOICE NUMBER\\nAPX-INV-2026-016044"

    This is a single cell containing both the label and value stacked
    vertically. The old code passed this as-is and _format_table_as_sentences
    treated "INVOICE NUMBER" as a label and the rest as undefined — losing
    the value.

    Detection: if ALL non-empty cells in a row contain exactly one "\\n",
    we treat it as a stacked-cell row and split each cell into
    [label, value] rows.

    Example transformation:
      Input row (1 row × 5 cols):
        ["INVOICE NUMBER\\nAPX-INV-2026-016044",
         "INVOICE DATE\\n27 May 2026",
         "DUE DATE\\n26 Jun 2026",
         "CURRENCY\\nUSD",
         "TERMS\\nNet 30"]

      Output (5 rows × 2 cols):
        [["INVOICE NUMBER", "APX-INV-2026-016044"],
         ["INVOICE DATE",   "27 May 2026"],
         ["DUE DATE",       "26 Jun 2026"],
         ["CURRENCY",       "USD"],
         ["TERMS",          "Net 30"]]

    This is then formatted by Pattern A as:
        "INVOICE NUMBER: APX-INV-2026-016044.
         INVOICE DATE: 27 May 2026.
         DUE DATE: 26 Jun 2026.
         CURRENCY: USD.
         TERMS: Net 30."
    """
    rows = []
    seen = set()

    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]

        # Deduplicate merged cells (Word repeats merged cell text)
        deduped = []
        prev = object()
        for c in cells:
            if c != prev:
                deduped.append(c)
            prev = c
        cells = deduped

        key = tuple(cells)
        if key in seen:
            continue
        seen.add(key)

        non_empty = [c for c in cells if c]
        if not non_empty:
            continue

        # Detect stacked-cell pattern: every non-empty cell has exactly one \n
        is_stacked = all("\n" in c for c in non_empty)

        if is_stacked:
            # Split each cell into [label, value] and emit as separate 2-col rows
            for cell_text in non_empty:
                parts = [p.strip() for p in cell_text.split("\n") if p.strip()]
                if len(parts) >= 2:
                    rows.append([parts[0], " ".join(parts[1:])])
                elif len(parts) == 1:
                    rows.append([parts[0], ""])
        else:
            # Normal row — clean internal newlines within each cell
            cleaned = [re.sub(r'\s*\n\s*', ' ', c).strip() for c in cells]
            rows.append(cleaned)

    return rows


def _extract_docx_chunks(file_bytes: bytes, filename: str) -> list:
    """
    Section-aware DOCX extraction with heading hierarchy context.

    UPDATED: now processes doc.tables in addition to doc.paragraphs,
    yielding both in their original document order via _iter_docx_body_order().

    Algorithm:
      1. Walk all body elements (paragraphs + tables) in document order
      2. Track heading stack as headings are encountered
      3. Accumulate normal paragraphs under the current heading
      4. When a table is encountered:
           a. Flush any accumulated paragraphs as a text chunk
           b. Format the table via _normalize_table_for_formatting()
              + _format_table_as_sentences()
           c. Emit the formatted table as a "table" chunk with the
              current heading context
      5. When a new heading is encountered, flush accumulated paragraphs
      6. Flush final accumulated paragraphs at end of document

    Result: each chunk represents one coherent document section or table.
    The heading context ("Invoice Header > Line Items") is stored separately
    and prepended only at embedding time.
    """
    from docx import Document

    doc           = Document(io.BytesIO(file_bytes))
    chunks        = []\

    heading_stack = []      # current [H1_text, H2_text, H3_text, ...]
    current_paras = []      # body paragraphs accumulated under current heading

    def flush_section():
        if current_paras:
            body    = "\n".join(current_paras).strip()
            context = _build_heading_context(heading_stack)
            if body:
                chunks.append({
                    "text":            body,
                    "heading_context": context,
                    "chunk_type":      "text",
                })

    for element_type, element in _iter_docx_body_order(doc):

        if element_type == "paragraph":
            raw   = element.text.strip()
            style = element.style.name if element.style else ""
            level = _get_heading_level(style)

            if level > 0:
                # New heading — flush accumulated paragraphs first
                flush_section()
                current_paras = []
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(raw)
            else:
                if raw:
                    current_paras.append(raw)

        elif element_type == "table":
            # Flush any paragraphs accumulated before this table
            flush_section()
            current_paras = []

            # Normalize and format the table
            table_data = _normalize_table_for_formatting(element)
            if table_data:
                formatted = _format_table_as_sentences(table_data)
                if formatted:
                    context = _build_heading_context(heading_stack)
                    chunks.append({
                        "text":            formatted,
                        "heading_context": context,
                        "chunk_type":      "table",
                    })
                    log.debug(
                        f"  DOCX table chunk ({len(formatted)} chars): "
                        f"'{formatted[:60]}...'"
                    )

    flush_section()  # flush final section

    log.info(
        f"  DOCX '{filename}': {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c['chunk_type']=='table')} tables, "
        f"{sum(1 for c in chunks if c['chunk_type']=='text')} text)"
    )
    return chunks


# ══════════════════════════════════════════════════════════════════
#  DOCX HELPERS
# ══════════════════════════════════════════════════════════════════

def _get_heading_level(style_name: str) -> int:
    """
    Return heading level (1-6) for heading styles, 0 otherwise.

    Handles: "Heading 1", "heading 2", "Title", "Subtitle", custom
    styles that begin with "Heading ".
    """
    if not style_name:
        return 0
    s = style_name.lower().strip()

    # Standard heading styles
    m = re.match(r'heading\s+(\d+)', s)
    if m:
        return int(m.group(1))

    # Title and subtitle treated as top-level headings
    if s in ("title", "subtitle"):
        return 1

    return 0


def _build_heading_context(stack: list) -> str:
    """
    Build a "H1 > H2 > H3" context string from the heading stack.
    Limited to 3 levels and 120 characters total.

    Examples:
        ["Data Retention Policy", "Section 3 — Personnel Records"]
        → "Data Retention Policy > Section 3 — Personnel Records"

        ["Master Services Agreement", "Termination", "Clause 8.2 Notice Period"]
        → "Master Services Agreement > Termination > Clause 8.2 Notice Period"
    """
    context = " > ".join(h for h in stack[:3] if h)
    if len(context) > 120:
        context = context[:117] + "..."
    return context