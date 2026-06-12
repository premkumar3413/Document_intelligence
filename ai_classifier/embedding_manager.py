
# """
# ai_classifier/embedding_manager.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embedding pipeline for the Document Intelligence platform.

# What changed vs previous version
# ─────────────────────────────────
# OLD: embed_and_store(conn, document_id, classification_id, text, final_category)
#      — took cleaned_text (flat string, structure lost)
#      — fixed 4,000-char character-slice chunking
#      — no document-level context per chunk
#      — retrieval cosine scores: ~0.44

# NEW: embed_and_store(conn, document_id, classification_id,
#                      file_bytes, filename, final_category)
#      — takes raw file bytes, does its own structure-aware extraction
#      — paragraph/sentence/character chunking (800 char max)
#      — Contextual Retrieval: GPT-4o-mini generates 2-sentence context per chunk
#      — retrieval cosine scores expected: 0.75–0.88

# Embedding pipeline (per document)
# ───────────────────────────────────
# Step 1  extract_chunks_for_embedding(file_bytes, filename)
#         → raw structural chunks (paragraphs/sections/tables)

# Step 2  _structure_split_chunks(raw_chunks)
#         → all chunks between MIN_CHUNK_CHARS and CHUNK_SIZE
#         Split order: paragraph boundary → sentence boundary → character
#         Merge strategy: chunks < MIN_CHUNK_CHARS appended to previous

# Step 3  _build_full_doc_text(sized_chunks)
#         → first CONTEXT_MAX_DOC_CHARS chars of the document
#         Used as document-level context for Step 4

# Step 4  For each chunk:
#         _generate_context(full_doc_text, chunk_text, heading_ctx, category)
#         → 2-sentence context situating chunk in the document

# Step 5  embed_text = f"{context}\n\n{chunk_text}"
#         _call_embedding_api(embed_text)
#         → 3,072-dim vector

# Step 6  _store_chunk(conn, document_id, idx, chunk_text, vector, "generated")
#         chunk_text = original (for search result display)
#         embedding  = vector of (context + chunk) — enriched for retrieval

# Calling scenarios
# ──────────────────
# Scenario 1 — Rule-only auto_classified   → called in _process_record()
# Scenario 2 — Rule+AI auto_classified     → called in _process_record()
# Scenario 3 — Human approved              → called in human_review()
# NOT called for soft_flag, hard_stop, or human_rejected.

# Embedding rules
# ────────────────
# ✓ Embed:  Contracts, Compliance, Governance, Certifications, Invoices
# ✗ Skip:   Miscellaneous — log skip record (NULL vector)
# ✗ Skip:   Empty text (scanned PDF) — log failed record

# DB operations
# ─────────────
# All chunk INSERTs use SAVEPOINTs so a single chunk failure cannot
# abort the outer transaction that also contains the classification_docs
# INSERT. This preserves existing FIX 1 (savepoint pattern).
# """

# import logging
# import re
# from typing import Optional

# from openai import AzureOpenAI

# from config import (
#     OPENAI_ENDPOINT,
#     OPENAI_KEY,
#     OPENAI_API_VER,
#     EMBEDDING_MODEL,
#     EMBEDDING_DIMS,
#     CHUNK_SIZE,
#     CHUNK_OVERLAP,
#     MIN_CHUNK_CHARS,
#     ACTIVE_CATEGORIES,
#     CONTEXTUAL_RETRIEVAL_ENABLED,
#     CONTEXT_MAX_DOC_CHARS,
#     CONTEXT_MODEL,
# )
# from utils.text_extractor import extract_chunks_for_embedding

# log = logging.getLogger(__name__)

# EMBEDDABLE_CATEGORIES = ACTIVE_CATEGORIES
# MISCELLANEOUS         = "Miscellaneous"

# # Sentence-ending pattern for splitting
# _SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


# # ══════════════════════════════════════════════════════════════════
# #  PUBLIC API
# # ══════════════════════════════════════════════════════════════════

# def embed_and_store(
#     conn,
#     document_id:       str,
#     classification_id: str,
#     file_bytes:        bytes,
#     filename:          str,
#     final_category:    str,
# ) -> dict:
#     """
#     Generate contextual embeddings for a finalized document and store them.

#     Signature changed from text: str to file_bytes + filename so the
#     embedding manager can perform structure-aware extraction internally
#     (DOCX heading hierarchy, PDF block detection, table formatting).
#     The caller no longer needs to pass cleaned_text.

#     Args:
#         conn              : active psycopg2 connection (caller commits)
#         document_id       : UUID from document_metadata
#         classification_id : UUID from classification_docs
#         file_bytes        : raw file content downloaded from ADLS
#         filename          : original filename (e.g. "contract_v2.pdf")
#         final_category    : confirmed category after classification/review

#     Returns:
#         {
#             "status":             "generated" | "skipped",
#             "category":           str,
#             "chunks_total":       int,
#             "chunks_stored":      int,
#             "chunks_failed":      int,
#             "first_embedding_id": str | None,
#         }
#     """
#     # ── Rule 1: Only embed the 5 active categories ────────────────────────────
#     if final_category not in EMBEDDABLE_CATEGORIES:
#         return _log_skipped(conn, document_id, final_category, "skipped_miscellaneous")

#     # ── Step 1: Structure-aware extraction → raw chunks ───────────────────────
#     try:
#         raw_chunks = extract_chunks_for_embedding(file_bytes, filename)
#     except ValueError as exc:
#         log.warning(f"  Extraction failed for {filename}: {exc}")
#         return _log_skipped(conn, document_id, final_category, "failed")
#     except Exception as exc:
#         log.warning(f"  Unexpected extraction error for {filename}: {exc}")
#         return _log_skipped(conn, document_id, final_category, "failed")

#     if not raw_chunks:
#         log.warning(
#             f"  No chunks extracted from '{filename}'. "
#             "Scanned PDF? OCR required for embedding."
#         )
#         return _log_skipped(conn, document_id, final_category, "failed")

#     # ── Step 2: Size-split → all chunks within [MIN_CHUNK_CHARS, CHUNK_SIZE] ──
#     sized_chunks = _structure_split_chunks(raw_chunks)
#     if not sized_chunks:
#         log.warning(f"  No usable chunks after size-split for '{filename}'")
#         return _log_skipped(conn, document_id, final_category, "failed")

#     log.info(
#         f"  Embedding '{final_category}' — '{filename}': "
#         f"{len(raw_chunks)} raw → {len(sized_chunks)} sized chunks"
#     )

#     # ── Step 3: Build full document text for context generation ───────────────
#     full_doc_text = _build_full_doc_text(sized_chunks)

#     # ── Steps 4-6: Context → embed → store (per chunk) ────────────────────────
#     first_embedding_id = None
#     stored = 0
#     failed = 0

#     for idx, chunk in enumerate(sized_chunks):
#         chunk_text      = chunk["text"]
#         heading_context = chunk.get("heading_context", "")

#         # Step 4: Contextual Retrieval — generate 2-sentence context
#         if CONTEXTUAL_RETRIEVAL_ENABLED:
#             context = _generate_context(
#                 full_doc_text, chunk_text, heading_context, final_category
#             )
#         else:
#             context = ""

#         # Step 5: Build text to embed (context + chunk)
#         if context:
#             embed_text = f"{context}\n\n{chunk_text}"
#         else:
#             embed_text = chunk_text

#         # Step 6: Embed and store
#         vector = _call_embedding_api(embed_text)

#         if vector:
#             emb_id = _store_chunk(
#                 conn, document_id, idx, chunk_text, vector, "generated"
#             )
#             if emb_id:
#                 stored += 1
#                 if idx == 0:
#                     first_embedding_id = emb_id
#                 log.debug(
#                     f"    Chunk {idx}: stored "
#                     f"({len(chunk_text):,} chars, context={bool(context)})"
#                 )
#             else:
#                 failed += 1
#                 log.warning(f"    Chunk {idx}: DB insert failed")
#         else:
#             _store_chunk(conn, document_id, idx, chunk_text, None, "failed")
#             failed += 1
#             log.warning(f"    Chunk {idx}: embedding API failed")

#     # ── Link first chunk's embedding_id → classification_docs ─────────────────
#     if first_embedding_id:
#         _update_classification_embedding_id(conn, classification_id, first_embedding_id)

#     log.info(
#         f"  Embedding complete: {stored} stored, {failed} failed "
#         f"out of {len(sized_chunks)} chunks"
#     )

#     return {
#         "status":              "generated",
#         "category":            final_category,
#         "chunks_total":        len(sized_chunks),
#         "chunks_stored":       stored,
#         "chunks_failed":       failed,
#         "first_embedding_id":  first_embedding_id,
#     }


# # ══════════════════════════════════════════════════════════════════
# #  CHUNKING
# # ══════════════════════════════════════════════════════════════════

# def _structure_split_chunks(raw_chunks: list) -> list:
#     """
#     Normalise raw structural chunks into embedding-ready chunks.

#     Rules:
#       - chunk > CHUNK_SIZE  → split at paragraph → sentence → character boundary
#       - chunk < MIN_CHUNK_CHARS AND not a table → merge into previous chunk
#       - tables are never merged with text chunks (kept as-is)

#     Returns a list of chunk dicts where each has:
#         text, heading_context, chunk_type
#     """
#     result: list = []

#     for raw in raw_chunks:
#         text     = raw["text"].strip()
#         context  = raw.get("heading_context", "")
#         ctype    = raw.get("chunk_type", "text")

#         if not text:
#             continue

#         if len(text) <= CHUNK_SIZE:
#             sub_chunks = [{"text": text, "heading_context": context, "chunk_type": ctype}]
#         else:
#             # Split at structural boundaries
#             sub_texts  = _split_at_boundary(text)
#             sub_chunks = [
#                 {"text": t, "heading_context": context, "chunk_type": ctype}
#                 for t in sub_texts if t.strip()
#             ]

#         for chunk in sub_chunks:
#             t = chunk["text"].strip()
#             if not t:
#                 continue

#             # Merge tiny non-table chunks into the previous chunk
#             if (
#                 len(t) < MIN_CHUNK_CHARS
#                 and chunk["chunk_type"] != "table"
#                 and result
#                 and result[-1]["chunk_type"] != "table"
#             ):
#                 result[-1]["text"] = result[-1]["text"].rstrip() + "\n" + t
#             else:
#                 result.append(chunk)

#     return result


# def _split_at_boundary(text: str) -> list:
#     """
#     Split text that exceeds CHUNK_SIZE into smaller pieces.

#     Priority order:
#       1. Paragraph boundary (\n\n) — preserves semantic units
#       2. Sentence boundary (. ! ?) — keeps complete sentences
#       3. Character boundary (last resort, adds CHUNK_OVERLAP)

#     Returns a list of strings each <= CHUNK_SIZE characters.
#     """
#     if len(text) <= CHUNK_SIZE:
#         return [text]

#     # ── Try paragraph split ────────────────────────────────────────────────────
#     paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
#     if len(paragraphs) > 1:
#         return _pack_into_chunks(paragraphs, separator="\n\n")

#     # ── Try sentence split ─────────────────────────────────────────────────────
#     sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
#     if len(sentences) > 1:
#         return _pack_into_chunks(sentences, separator=" ")

#     # ── Character split (last resort) ─────────────────────────────────────────
#     # Overlap of CHUNK_OVERLAP chars carried over to next chunk so context
#     # at boundaries is not lost when no natural boundary exists.
#     chunks = []
#     start  = 0
#     while start < len(text):
#         end = min(start + CHUNK_SIZE, len(text))
#         chunks.append(text[start:end])
#         if end == len(text):
#             break
#         start = end - CHUNK_OVERLAP
#     return chunks


# def _pack_into_chunks(pieces: list, separator: str) -> list:
#     """
#     Greedily pack pieces (paragraphs or sentences) into chunks of at most
#     CHUNK_SIZE characters, using the given separator between pieces.
#     If a single piece itself exceeds CHUNK_SIZE, it is split recursively.
#     """
#     chunks  = []
#     current = ""

#     for piece in pieces:
#         if not piece:
#             continue

#         if len(piece) > CHUNK_SIZE:
#             # Single piece is oversize — recurse
#             if current:
#                 chunks.append(current)
#                 current = ""
#             chunks.extend(_split_at_boundary(piece))
#             continue

#         candidate = (current + separator + piece).lstrip(separator) if current else piece

#         if len(candidate) <= CHUNK_SIZE:
#             current = candidate
#         else:
#             if current:
#                 chunks.append(current)
#             current = piece

#     if current:
#         chunks.append(current)

#     return chunks


# def _build_full_doc_text(chunks: list) -> str:
#     """
#     Concatenate all chunk texts to form a document-level context string.
#     Truncated to CONTEXT_MAX_DOC_CHARS for the GPT-4o-mini prompt.
#     """
#     full = "\n\n".join(c["text"] for c in chunks)
#     return full[:CONTEXT_MAX_DOC_CHARS]


# # ══════════════════════════════════════════════════════════════════
# #  CONTEXTUAL RETRIEVAL
# # ══════════════════════════════════════════════════════════════════

# def _generate_context(
#     full_doc_text:   str,
#     chunk_text:      str,
#     heading_context: str,
#     category:        str,
# ) -> str:
#     """
#     Generate a 2-sentence context that situates this chunk within the document.

#     Anthropic Contextual Retrieval technique (Sept 2024):
#       Prepending context to each chunk before embedding reduces retrieval
#       failures by ~49% compared to embedding raw chunks.

#     Example output:
#       "This chunk is from an Invoices document issued by ABC Technologies
#        dated May 2026. It describes the line items and grand total including
#        18% GST for cloud hosting and AI document processing services."

#     On failure: returns "" so the chunk is embedded without context
#     rather than skipping the chunk entirely.
#     """
#     heading_hint = (
#         f" The chunk is located under section: '{heading_context}'."
#         if heading_context else ""
#     )

#     system_msg = (
#         "You are a document indexing assistant. "
#         "Given a business document and a specific chunk from it, "
#         "write exactly 2 sentences that describe: "
#         "(1) what type of document this is and its main purpose, "
#         "(2) what specific topic or section this chunk covers. "
#         "Be concise and factual. Do not quote or paraphrase the chunk. "
#         "Output only the 2-sentence context, nothing else."
#     )

#     user_msg = (
#         f"Document category: {category}\n\n"
#         f"Full document text (excerpt):\n{full_doc_text}\n\n"
#         f"---\n"
#         f"Chunk to contextualize:{heading_hint}\n{chunk_text}\n\n"
#         f"---\n"
#         "Generate the 2-sentence context for this chunk:"
#     )

#     try:
#         client = AzureOpenAI(
#             azure_endpoint=OPENAI_ENDPOINT,
#             api_key=OPENAI_KEY,
#             api_version=OPENAI_API_VER,
#         )
#         response = client.chat.completions.create(
#             model=CONTEXT_MODEL,
#             messages=[
#                 {"role": "system", "content": system_msg},
#                 {"role": "user",   "content": user_msg},
#             ],
#             temperature=0.1,
#             max_tokens=120,
#         )
#         context = response.choices[0].message.content.strip()
#         return context
#     except Exception as exc:
#         log.warning(
#             f"    Context generation failed (chunk will be embedded without context): {exc}"
#         )
#         return ""


# # ══════════════════════════════════════════════════════════════════
# #  EMBEDDING API
# # ══════════════════════════════════════════════════════════════════

# def _call_embedding_api(text: str) -> Optional[list]:
#     """
#     Call text-embedding-3-large for one text string.
#     Returns the 3,072-dim vector or None on failure.
#     """
#     try:
#         client = AzureOpenAI(
#             azure_endpoint=OPENAI_ENDPOINT,
#             api_key=OPENAI_KEY,
#             api_version=OPENAI_API_VER,
#         )
#         response = client.embeddings.create(
#             input=text,
#             model=EMBEDDING_MODEL,
#         )
#         vec = response.data[0].embedding
#         assert len(vec) == EMBEDDING_DIMS, (
#             f"Dimension mismatch: got {len(vec)}, expected {EMBEDDING_DIMS}"
#         )
#         return vec
#     except Exception as exc:
#         log.warning(f"    Embedding API error: {exc}")
#         return None


# # ══════════════════════════════════════════════════════════════════
# #  DATABASE OPERATIONS  (savepoint pattern preserved from v2)
# # ══════════════════════════════════════════════════════════════════

# def _store_chunk(
#     conn,
#     document_id:  str,
#     chunk_index:  int,
#     chunk_text:   str,
#     vector:       Optional[list],
#     status:       str,
# ) -> Optional[str]:
#     """
#     Insert or upsert one chunk row in document_embeddings.

#     Uses a SAVEPOINT so that if this INSERT fails (e.g. constraint
#     violation), only this chunk is rolled back — the outer transaction
#     (classification_docs INSERT, other chunks) stays intact.

#     The embedding column stores the vector of (context + chunk_text)
#     but chunk_text stores only the original chunk for user display.

#     Returns the embedding_id UUID on success, None on failure.
#     """
#     vec_str   = "[" + ",".join(repr(float(v)) for v in vector) + "]" if vector else None
#     savepoint = f"sp_chunk_{chunk_index}"
#     cur       = conn.cursor()

#     try:
#         cur.execute(f"SAVEPOINT {savepoint}")
#         cur.execute(
#             """
#             INSERT INTO document_embeddings (
#                 document_id, chunk_index, chunk_text,
#                 embedding, embedding_status, model_name
#             )
#             VALUES (%s, %s, %s, %s::halfvec, %s, %s)
#             ON CONFLICT (document_id, chunk_index) DO UPDATE SET
#                 chunk_text            = EXCLUDED.chunk_text,
#                 embedding             = EXCLUDED.embedding,
#                 embedding_status      = EXCLUDED.embedding_status,
#                 model_name            = EXCLUDED.model_name,
#                 current_timestamp_ist = (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata')
#             RETURNING embedding_id
#             """,
#             (document_id, chunk_index, chunk_text, vec_str, status, EMBEDDING_MODEL),
#         )
#         row = cur.fetchone()
#         cur.execute(f"RELEASE SAVEPOINT {savepoint}")
#         cur.close()
#         return str(row[0]) if row else None

#     except Exception as exc:
#         try:
#             cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
#         except Exception:
#             pass
#         cur.close()
#         log.error(
#             f"    Failed to store chunk {chunk_index} for {document_id}: {exc}"
#         )
#         return None


# def _log_skipped(
#     conn,
#     document_id:    str,
#     final_category: str,
#     status:         str,
# ) -> dict:
#     """
#     Record a skip/failed row in document_embeddings.
#     Uses a savepoint so this cannot abort the outer transaction.
#     """
#     savepoint = "sp_skip_log"
#     cur       = conn.cursor()
#     try:
#         cur.execute(f"SAVEPOINT {savepoint}")
#         cur.execute(
#             """
#             INSERT INTO document_embeddings (
#                 document_id, chunk_index, chunk_text,
#                 embedding, embedding_status, model_name
#             )
#             VALUES (%s, 0, '', NULL, %s, %s)
#             ON CONFLICT (document_id, chunk_index) DO UPDATE SET
#                 embedding_status      = EXCLUDED.embedding_status,
#                 current_timestamp_ist = (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata')
#             """,
#             (document_id, status, EMBEDDING_MODEL if status == "failed" else None),
#         )
#         cur.execute(f"RELEASE SAVEPOINT {savepoint}")
#         log.info(
#             f"  Embedding skip recorded: {document_id} — "
#             f"category='{final_category}', status='{status}'"
#         )
#     except Exception as exc:
#         try:
#             cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
#         except Exception:
#             pass
#         log.warning(f"  Failed to log skipped embedding: {exc}")
#     finally:
#         cur.close()

#     return {
#         "status":              "skipped",
#         "category":            final_category,
#         "skip_reason":         status,
#         "chunks_total":        0,
#         "chunks_stored":       0,
#         "chunks_failed":       0,
#         "first_embedding_id":  None,
#     }


# def _update_classification_embedding_id(
#     conn,
#     classification_id: str,
#     embedding_id:      str,
# ) -> None:
#     """
#     Link the first chunk's embedding_id to classification_docs.
#     Savepoint ensures this cannot abort the outer transaction.
#     """
#     savepoint = "sp_emb_id_update"
#     cur       = conn.cursor()
#     try:
#         cur.execute(f"SAVEPOINT {savepoint}")
#         cur.execute(
#             """
#             UPDATE classification_docs
#             SET    embedding_id = %s
#             WHERE  classification_id = %s
#             """,
#             (embedding_id, classification_id),
#         )
#         cur.execute(f"RELEASE SAVEPOINT {savepoint}")
#     except Exception as exc:
#         try:
#             cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
#         except Exception:
#             pass
#         log.warning(f"  Failed to update classification_docs.embedding_id: {exc}")
#     finally:
#         cur.close()





















"""
ai_classifier/embedding_manager.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Embedding pipeline for the Document Intelligence platform.

WHAT CHANGED vs previous version — TWO FIXES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX 1 — Batch embedding (the main speedup)
──────────────────────────────────────────
OLD: _call_embedding_api(text: str) was called ONCE PER CHUNK inside a
     for-loop. For a 130-chunk document = 130 sequential HTTP calls.
     Each call takes ~500ms → ~65 seconds minimum just for network time.

     LOG EVIDENCE from your logs:
       18:24:53 → 18:26:12 = 79 seconds for ~130 chunks
       One HTTP 200 response every ~500ms, perfectly sequential.

NEW: _call_embedding_api_batch(texts: list[str]) sends ALL chunks in
     a single API call. The Azure OpenAI embeddings endpoint accepts
     up to 2048 inputs per request. 130 chunks = 2 API calls not 130.

     SPEEDUP:
       Before: 130 calls × 500ms = ~65s
       After : 2 calls  × 800ms  = ~1.6s   (~40x faster)

     BATCH_SIZE = 96 (safe margin below the 2048 hard limit).

FIX 2 — Module-level AzureOpenAI singleton
───────────────────────────────────────────
OLD: new AzureOpenAI() constructed inside every _call_embedding_api()
     call — rebuilds the HTTP connection pool each time.
NEW: _openai_client created once at module import, reused for all calls.

Everything else is unchanged:
  structure-aware chunking, SAVEPOINT pattern, CONTEXTUAL_RETRIEVAL flag,
  embed_and_store() signature, _log_skipped/_store_chunk/_update_*.
"""

import logging
import math
import re
from typing import Optional

from openai import AzureOpenAI

from config import (
    OPENAI_ENDPOINT,
    OPENAI_KEY,
    OPENAI_API_VER,
    EMBEDDING_MODEL,
    EMBEDDING_DIMS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    ACTIVE_CATEGORIES,
    CONTEXTUAL_RETRIEVAL_ENABLED,
    CONTEXT_MAX_DOC_CHARS,
    CONTEXT_MODEL,
)
from utils.text_extractor import extract_chunks_for_embedding

log = logging.getLogger(__name__)

EMBEDDABLE_CATEGORIES = ACTIVE_CATEGORIES
MISCELLANEOUS         = "Miscellaneous"
_SENTENCE_END         = re.compile(r'(?<=[.!?])\s+')

# ── FIX 2: Module-level client singleton ──────────────────────────────────────
_openai_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version=OPENAI_API_VER,
)

# ── FIX 1: Batch size ─────────────────────────────────────────────────────────
# Azure OpenAI accepts up to 2048 inputs per embeddings request.
# 96 is a safe batch size for text-embedding-3-large with long chunk texts.
EMBEDDING_BATCH_SIZE = 96


# ══════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════

def embed_and_store(
    conn,
    document_id:       str,
    classification_id: str,
    file_bytes:        bytes,
    filename:          str,
    final_category:    str,
) -> dict:
    """
    Generate embeddings for a finalized document and store them.

    Returns:
        { status, category, chunks_total, chunks_stored, chunks_failed,
          first_embedding_id }
    """
    if final_category not in EMBEDDABLE_CATEGORIES:
        return _log_skipped(conn, document_id, final_category, "skipped_miscellaneous")

    # Step 1 — structure-aware extraction
    try:
        raw_chunks = extract_chunks_for_embedding(file_bytes, filename)
    except ValueError as exc:
        log.warning(f"  Extraction failed for {filename}: {exc}")
        return _log_skipped(conn, document_id, final_category, "failed")
    except Exception as exc:
        log.warning(f"  Unexpected extraction error for {filename}: {exc}")
        return _log_skipped(conn, document_id, final_category, "failed")

    if not raw_chunks:
        log.warning(f"  No chunks extracted from '{filename}'. Scanned PDF?")
        return _log_skipped(conn, document_id, final_category, "failed")

    # Step 2 — size normalisation
    sized_chunks = _structure_split_chunks(raw_chunks)
    if not sized_chunks:
        log.warning(f"  No usable chunks after size-split for '{filename}'")
        return _log_skipped(conn, document_id, final_category, "failed")

    log.info(
        f"  Embedding '{final_category}' — '{filename}': "
        f"{len(raw_chunks)} raw → {len(sized_chunks)} sized chunks"
    )

    # Step 3 — full document text for context generation
    full_doc_text = _build_full_doc_text(sized_chunks)

    # Step 4 — optional contextual prefixes (one LLM call per chunk)
    if CONTEXTUAL_RETRIEVAL_ENABLED:
        log.info(f"  Contextual Retrieval ON: generating {len(sized_chunks)} prefixes...")
        contexts = [
            _generate_context(
                full_doc_text,
                chunk["text"],
                chunk.get("heading_context", ""),
                final_category,
            )
            for chunk in sized_chunks
        ]
    else:
        contexts = [""] * len(sized_chunks)

    # Step 5 — build embed texts and call API IN BATCHES  ← THE FIX
    embed_texts = [
        f"{ctx}\n\n{chunk['text']}" if ctx else chunk["text"]
        for ctx, chunk in zip(contexts, sized_chunks)
    ]

    n_batches = math.ceil(len(embed_texts) / EMBEDDING_BATCH_SIZE)
    log.info(
        f"  Embedding API: {len(embed_texts)} chunks → "
        f"{n_batches} batch(es) of ≤{EMBEDDING_BATCH_SIZE}"
    )

    vectors = _call_embedding_api_batch(embed_texts)

    # Step 6 — store each (chunk, vector) with SAVEPOINT
    first_embedding_id = None
    stored = 0
    failed = 0

    for idx, (chunk, vector) in enumerate(zip(sized_chunks, vectors)):
        chunk_text = chunk["text"]

        if vector:
            emb_id = _store_chunk(conn, document_id, idx, chunk_text, vector, "generated")
            if emb_id:
                stored += 1
                if idx == 0:
                    first_embedding_id = emb_id
                log.debug(f"    Chunk {idx}: stored ({len(chunk_text):,} chars)")
            else:
                failed += 1
                log.warning(f"    Chunk {idx}: DB insert failed")
        else:
            _store_chunk(conn, document_id, idx, chunk_text, None, "failed")
            failed += 1
            log.warning(f"    Chunk {idx}: embedding was None (API failure)")

    if first_embedding_id:
        _update_classification_embedding_id(conn, classification_id, first_embedding_id)

    log.info(
        f"  Embedding complete: {stored} stored, {failed} failed "
        f"out of {len(sized_chunks)} chunks"
    )

    return {
        "status":              "generated",
        "category":            final_category,
        "chunks_total":        len(sized_chunks),
        "chunks_stored":       stored,
        "chunks_failed":       failed,
        "first_embedding_id":  first_embedding_id,
    }


# ══════════════════════════════════════════════════════════════════
#  EMBEDDING API — BATCH  (core fix)
# ══════════════════════════════════════════════════════════════════

def _call_embedding_api_batch(texts: list) -> list:
    """
    Embed a list of texts in batches using the module-level singleton client.

    WHY THIS IS FAST:
      The Azure OpenAI embeddings API accepts input as a list[str] and
      returns all vectors in a single HTTP round-trip. Previously the
      code called the API once per chunk (sequential, one round-trip each).

      Before: 130 chunks × 1 call = 130 HTTP round-trips ~ 65s
      After : 130 chunks / 96     = 2 HTTP round-trips   ~ 1.6s

    The API guarantees response.data is ordered the same as the input
    list, so index alignment between texts and returned vectors is exact.

    Returns a list of vectors in the same order as input.
    Chunks whose embedding failed are represented as None (stored as
    status="failed" rather than crashing the whole document).
    """
    if not texts:
        return []

    all_vectors: list = [None] * len(texts)
    total_batches = math.ceil(len(texts) / EMBEDDING_BATCH_SIZE)

    for batch_num, batch_start in enumerate(
        range(0, len(texts), EMBEDDING_BATCH_SIZE), start=1
    ):
        batch_texts = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]

        try:
            response = _openai_client.embeddings.create(
                input=batch_texts,
                model=EMBEDDING_MODEL,
            )

            for i, embedding_obj in enumerate(response.data):
                vec = embedding_obj.embedding
                if len(vec) != EMBEDDING_DIMS:
                    log.warning(
                        f"    Batch {batch_num}: chunk {batch_start + i} "
                        f"dimension mismatch: got {len(vec)}, expected {EMBEDDING_DIMS}"
                    )
                    continue
                all_vectors[batch_start + i] = vec

            log.info(
                f"  Embedding batch {batch_num}/{total_batches}: "
                f"{len(batch_texts)} chunks OK"
            )

        except Exception as exc:
            log.error(
                f"  Embedding batch {batch_num}/{total_batches} FAILED "
                f"(chunks {batch_start}–{batch_start + len(batch_texts) - 1} → None): {exc}"
            )

    return all_vectors


# ══════════════════════════════════════════════════════════════════
#  CHUNKING  (unchanged)
# ══════════════════════════════════════════════════════════════════

def _structure_split_chunks(raw_chunks: list) -> list:
    result: list = []
    for raw in raw_chunks:
        text    = raw["text"].strip()
        context = raw.get("heading_context", "")
        ctype   = raw.get("chunk_type", "text")
        if not text:
            continue
        if len(text) <= CHUNK_SIZE:
            sub_chunks = [{"text": text, "heading_context": context, "chunk_type": ctype}]
        else:
            sub_texts  = _split_at_boundary(text)
            sub_chunks = [
                {"text": t, "heading_context": context, "chunk_type": ctype}
                for t in sub_texts if t.strip()
            ]
        for chunk in sub_chunks:
            t = chunk["text"].strip()
            if not t:
                continue
            if (
                len(t) < MIN_CHUNK_CHARS
                and chunk["chunk_type"] != "table"
                and result
                and result[-1]["chunk_type"] != "table"
            ):
                result[-1]["text"] = result[-1]["text"].rstrip() + "\n" + t
            else:
                result.append(chunk)
    return result


def _split_at_boundary(text: str) -> list:
    if len(text) <= CHUNK_SIZE:
        return [text]
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if len(paragraphs) > 1:
        return _pack_into_chunks(paragraphs, separator="\n\n")
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if len(sentences) > 1:
        return _pack_into_chunks(sentences, separator=" ")
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def _pack_into_chunks(pieces: list, separator: str) -> list:
    chunks  = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_at_boundary(piece))
            continue
        candidate = (current + separator + piece).lstrip(separator) if current else piece
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _build_full_doc_text(chunks: list) -> str:
    return "\n\n".join(c["text"] for c in chunks)[:CONTEXT_MAX_DOC_CHARS]


# ══════════════════════════════════════════════════════════════════
#  CONTEXTUAL RETRIEVAL  (unchanged — only used when enabled)
# ══════════════════════════════════════════════════════════════════

def _generate_context(
    full_doc_text: str,
    chunk_text: str,
    heading_context: str,
    category: str,
) -> str:
    heading_hint = (
        f" The chunk is located under section: '{heading_context}'."
        if heading_context else ""
    )
    system_msg = (
        "You are a document indexing assistant. "
        "Given a business document and a specific chunk from it, "
        "write exactly 2 sentences that describe: "
        "(1) what type of document this is and its main purpose, "
        "(2) what specific topic or section this chunk covers. "
        "Be concise and factual. Do not quote or paraphrase the chunk. "
        "Output only the 2-sentence context, nothing else."
    )
    user_msg = (
        f"Document category: {category}\n\n"
        f"Full document text (excerpt):\n{full_doc_text}\n\n"
        f"---\n"
        f"Chunk to contextualize:{heading_hint}\n{chunk_text}\n\n"
        f"---\n"
        "Generate the 2-sentence context for this chunk:"
    )
    try:
        response = _openai_client.chat.completions.create(
            model=CONTEXT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        log.warning(f"    Context generation failed (chunk embedded without context): {exc}")
        return ""


# ══════════════════════════════════════════════════════════════════
#  DATABASE OPERATIONS  (savepoint pattern — unchanged)
# ══════════════════════════════════════════════════════════════════

def _store_chunk(
    conn,
    document_id:  str,
    chunk_index:  int,
    chunk_text:   str,
    vector:       Optional[list],
    status:       str,
) -> Optional[str]:
    vec_str   = "[" + ",".join(repr(float(v)) for v in vector) + "]" if vector else None
    savepoint = f"sp_chunk_{chunk_index}"
    cur       = conn.cursor()
    try:
        cur.execute(f"SAVEPOINT {savepoint}")
        cur.execute(
            """
            INSERT INTO document_embeddings (
                document_id, chunk_index, chunk_text,
                embedding, embedding_status, model_name
            )
            VALUES (%s, %s, %s, %s::halfvec, %s, %s)
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                chunk_text            = EXCLUDED.chunk_text,
                embedding             = EXCLUDED.embedding,
                embedding_status      = EXCLUDED.embedding_status,
                model_name            = EXCLUDED.model_name,
                current_timestamp_ist = (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata')
            RETURNING embedding_id
            """,
            (document_id, chunk_index, chunk_text, vec_str, status, EMBEDDING_MODEL),
        )
        row = cur.fetchone()
        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
        cur.close()
        return str(row[0]) if row else None
    except Exception as exc:
        try:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        except Exception:
            pass
        cur.close()
        log.error(f"    Failed to store chunk {chunk_index} for {document_id}: {exc}")
        return None


def _log_skipped(conn, document_id, final_category, status) -> dict:
    savepoint = "sp_skip_log"
    cur       = conn.cursor()
    try:
        cur.execute(f"SAVEPOINT {savepoint}")
        cur.execute(
            """
            INSERT INTO document_embeddings (
                document_id, chunk_index, chunk_text,
                embedding, embedding_status, model_name
            )
            VALUES (%s, 0, '', NULL, %s, %s)
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                embedding_status      = EXCLUDED.embedding_status,
                current_timestamp_ist = (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata')
            """,
            (document_id, status, EMBEDDING_MODEL if status == "failed" else None),
        )
        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
        log.info(
            f"  Embedding skip recorded: {document_id} — "
            f"category='{final_category}', status='{status}'"
        )
    except Exception as exc:
        try:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        except Exception:
            pass
        log.warning(f"  Failed to log skipped embedding: {exc}")
    finally:
        cur.close()
    return {
        "status":              "skipped",
        "category":            final_category,
        "skip_reason":         status,
        "chunks_total":        0,
        "chunks_stored":       0,
        "chunks_failed":       0,
        "first_embedding_id":  None,
    }


def _update_classification_embedding_id(conn, classification_id, embedding_id) -> None:
    savepoint = "sp_emb_id_update"
    cur       = conn.cursor()
    try:
        cur.execute(f"SAVEPOINT {savepoint}")
        cur.execute(
            "UPDATE classification_docs SET embedding_id = %s WHERE classification_id = %s",
            (embedding_id, classification_id),
        )
        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception as exc:
        try:
            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        except Exception:
            pass
        log.warning(f"  Failed to update classification_docs.embedding_id: {exc}")
    finally:
        cur.close()