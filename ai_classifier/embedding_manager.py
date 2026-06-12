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
from config import AI_TEMPERATURE

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
            temperature=AI_TEMPERATURE,
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