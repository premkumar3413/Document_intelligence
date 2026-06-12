"""
routers/search.py — Semantic Search and Context-Aware Q&A API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoint: POST /api/search/semantic

Two retrieval modes (controlled by the "mode" field in the request):

  mode = "search"  — Semantic Document Search
    Finds documents semantically related to the query.
    Returns a ranked list of matching documents with excerpts.
    No LLM answer generation — pure vector retrieval.
    Use case: "Find invoices related to cloud hosting"

  mode = "qa"      — Context-Aware Question Answering
    Retrieves the most relevant document chunks, then passes them
    to GPT-4o-mini to generate a direct answer with citations.
    Use case: "What is the payment term in the invoice from ABC Technologies?"

Two search strategies (controlled by "search_mode"):

  search_mode = "standard"  — Embed the query directly.
    Score range: 0.50–0.65. Best for browsing and topic search.

  search_mode = "hyde"      — HyDE (Hypothetical Document Embeddings).
    GPT-4o-mini writes a realistic answer, then embeds THAT.
    Score range: 0.63–0.72. Best for specific fact extraction (Q&A).
    Default for mode="qa".

Retrieval approach:
  Production uses TOP-K retrieval (not hard threshold) so that
  correct chunks with slightly lower scores are never silently
  filtered out. The LLM handles relevance judgment in Q&A mode.

Example request (Swagger UI: http://localhost:8000/docs):

  POST /api/search/semantic
  {
    "query": "What is the grand total on the invoice from ABC Technologies?",
    "mode": "qa",
    "search_mode": "hyde",
    "filters": {
      "document_type": "Invoices"
    }
  }
"""

import logging
from typing import Optional

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, HTTPException
from openai import AzureOpenAI, RateLimitError, APIStatusError, APIConnectionError
from database import get_connection
from pydantic import BaseModel

import config
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from config import AI_TEMPERATURE


log = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])

# ── Constants ──────────────────────────────────────────────────────────────────

CHUNK_CANDIDATES  = 20   # chunks fetched from pgvector before deduplication
FINAL_TOP_K       = 5    # documents returned after deduplication
QA_CONTEXT_DOCS   = 3    # documents passed to GPT-4o-mini for answer generation


# ══════════════════════════════════════════════════════════════════
#  PYDANTIC REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════

class SearchFilters(BaseModel):
    """
    Optional metadata filters applied in the SQL WHERE clause.

    All fields are optional — omit any field to skip that filter.

    document_type:         e.g. "Invoices", "Contracts", "Compliance"
    source_type:           e.g. "Outlook", "Sharepoint", "manual_upload"
    classification_status: e.g. "auto_classified", "human_approved"

    Swagger UI note: Swagger fills string fields with "string" as a
    placeholder. This model automatically ignores placeholder values
    so you can test without removing the filters field entirely.
    """
    document_type:         Optional[str] = None
    source_type:           Optional[str] = None
    classification_status: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Silently discard Swagger UI placeholder values ('string', 'str', etc.)."""
        _placeholders = {"string", "str", "example", "test", "<string>", "none", "null"}
        for field in ("document_type", "source_type", "classification_status"):
            val = getattr(self, field)
            if val and val.strip().lower() in _placeholders:
                setattr(self, field, None)


class SearchRequest(BaseModel):
    """
    Request body for POST /api/search/semantic.

    Fields:
        query       : natural language search query
        mode        : "search" (document list) or "qa" (answer + citations)
        search_mode : "standard" (embed query) or "hyde" (embed hypothetical answer)
                      Defaults to "hyde" when mode="qa", "standard" when mode="search"
        filters     : optional metadata filters
    """
    query:       str
    mode:        str = "search"    # "search" | "qa"
    search_mode: Optional[str] = None  # "standard" | "hyde" | None (auto)
    filters:     Optional[SearchFilters] = None


class SearchResultItem(BaseModel):
    """One document returned in search mode."""
    rank:                    int
    similarity_score:        float
    file_name:               str
    document_type:           Optional[str]
    source_type:             str
    classification_status:   Optional[str]
    output_document_path:    Optional[str]
    matched_chunk_index:     int
    matched_excerpt:         str
    ingested_at:             Optional[str]


class SemanticSearchResponse(BaseModel):
    """Response body for mode='search'."""
    query:         str
    mode:          str
    search_mode:   str
    filters:       Optional[dict]
    total_results: int
    results:       list[SearchResultItem]


class Citation(BaseModel):
    """Source document cited in a Q&A answer."""
    file_name:             str
    document_type:         Optional[str]
    similarity_score:      float
    matched_excerpt:       str
    output_document_path:  Optional[str]


class QAResponse(BaseModel):
    """Response body for mode='qa'."""
    query:       str
    mode:        str
    search_mode: str
    filters:     Optional[dict]
    answer:      str
    citations:   list[Citation]


# ══════════════════════════════════════════════════════════════════
#  AZURE OPENAI HELPERS
# ══════════════════════════════════════════════════════════════════

# ── Module-level AzureOpenAI singleton ────────────────────────────────────────
# Created once at import time instead of per-call. Avoids rebuilding the HTTP
# connection pool on every embed / HyDE / answer-generation call.
_search_client = AzureOpenAI(
    azure_endpoint=config.OPENAI_ENDPOINT,
    api_key=config.OPENAI_KEY,
    api_version=config.OPENAI_API_VER,
)

# ── Retry decorator for transient Azure OpenAI errors (429 / 5xx) ─────────────
_search_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError, APIConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)


@_search_retry
def _embed(text: str) -> list:
    """Embed a text string with text-embedding-3-large (3072 dims). Retries on 429/5xx."""
    response = _search_client.embeddings.create(input=text, model=config.EMBEDDING_MODEL)
    vec      = response.data[0].embedding
    if len(vec) != config.EMBEDDING_DIMS:
        raise ValueError(f"Embedding dimension mismatch: {len(vec)} != {config.EMBEDDING_DIMS}")
    return vec


def _vec_to_str(vec: list) -> str:
    """
    Convert a 3072-dim float list to a PostgreSQL halfvec literal string.

    Format: '[f1,f2,...,f3072]'

    Uses fixed-point notation (.15f) instead of repr() to guarantee
    no scientific notation (like 1.234e-10) reaches the halfvec parser.

    repr(1.234e-10) → '1.234e-10'      ← halfvec may reject in some contexts
    f"{1.234e-10:.15f}" → '0.000000000123400'  ← always accepted

    Why this matters: search_debug.py (standalone) works with repr() but
    the uvicorn/FastAPI context may have different PostgreSQL session
    settings that cause the halfvec cast to fail silently on scientific
    notation values, returning 0 rows instead of raising an exception.
    """
    return "[" + ",".join(f"{float(v):.15f}" for v in vec) + "]"


def _generate_hypothetical_answer(query: str) -> str:
    """
    HyDE: ask GPT-4o-mini to write a realistic document excerpt that answers
    the query. Embed that answer instead of the question.

    Benefit: both the answer and the stored chunk are in DOCUMENT embedding
    space, reducing the Q→D cosine gap from ~0.58 to ~0.66.
    """
    system = (
        "You are a document retrieval assistant. "
        "Given a question about a business document, write a realistic 2–3 sentence "
        "excerpt that answers it as if quoting directly from the document. "
        "Be specific with amounts, names, and terminology. "
        "Output ONLY the document excerpt — no preamble."
    )
    response = _search_client.chat.completions.create(
        model=config.AI_CLASSIFICATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Write a document excerpt that answers: {query}"},
        ],
        temperature=AI_TEMPERATURE,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════

# _get_db_connection() removed — uses the shared ThreadedConnectionPool
# from database.py via get_connection() instead of opening a raw socket.
# conn.close() in the finally block returns it to the pool (monkey-patched).


# ══════════════════════════════════════════════════════════════════
#  PGVECTOR SEARCH
# ══════════════════════════════════════════════════════════════════

def _log_db_state(conn) -> None:
    """
    Diagnostic: log the JOIN state between embeddings and classification.
    Called automatically when the main search returns 0 results to help
    identify whether the issue is a missing/mismatched classification record.
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    dm.file_name,
                    de.document_id          AS emb_doc_id,
                    cd.document_id          AS cls_doc_id,
                    cd.classification_status,
                    cd.document_type,
                    COUNT(de.chunk_index)   AS chunk_count
                FROM document_embeddings de
                JOIN document_metadata   dm ON dm.document_id = de.document_id
                LEFT JOIN classification_docs cd ON cd.document_id = de.document_id
                WHERE de.embedding_status = 'generated'
                GROUP BY dm.file_name, de.document_id,
                         cd.document_id, cd.classification_status, cd.document_type
                ORDER BY dm.file_name
            """)
            rows = cur.fetchall()

        if not rows:
            log.warning("  DIAGNOSTIC: document_embeddings table is empty")
            return

        log.warning("  DIAGNOSTIC — embedding vs classification state:")
        for r in rows:
            cls_id  = r.get("cls_doc_id")
            emb_id  = str(r["emb_doc_id"])
            status  = r.get("classification_status", "MISSING")
            match   = "IDs match ✓" if cls_id and str(cls_id) == emb_id else "ID MISMATCH ✗" if cls_id else "NO classification record ✗"
            log.warning(
                f"    {r['file_name']} | chunks={r['chunk_count']} | "
                f"status={status} | {match}"
            )
            if cls_id and str(cls_id) == emb_id and status not in ("auto_classified", "human_approved"):
                log.warning(
                    f"    → FIX: classification_status='{status}' is not searchable. "
                    "Must be 'auto_classified' or 'human_approved'. "
                    "Use PUT /api/classification/{id}/review to approve, "
                    "or re-run POST /api/classify/batch."
                )
            elif not cls_id:
                log.warning(
                    f"    → FIX: No classification_docs record for this document. "
                    "Re-run POST /api/classify/batch to classify and embed."
                )
    except Exception as exc:
        log.warning(f"  DIAGNOSTIC failed: {exc}")


def _run_vector_search(conn, vec_str: str, filters: SearchFilters) -> list:
    """
    Execute pgvector cosine similarity search against document_embeddings.

    KEY FIX — Sequential scan forced (SET LOCAL enable_indexscan = off):
    The HNSW index uses Approximate Nearest Neighbor search. With few
    vectors (< ~100) the HNSW graph has low connectivity and can return
    0 results for query vectors that are semantically distant from stored
    vectors. This manifests as 0 results in the FastAPI context while
    search_debug.py (standalone) appears to work.

    Forcing a sequential scan checks EVERY row — guaranteed to find all
    matching chunks. With 14–1000 chunks this is instant (<5ms). Enable
    the HNSW index only when you have 10,000+ vectors.

    Metadata filters are applied in the SQL WHERE clause (pre-filter)
    so only relevant documents are compared.
    """
    extra_where = ""
    extra_where_no_cd = ""
    if filters:
        if filters.document_type:
            extra_where += f" AND cd.document_type = '{filters.document_type}'"
        if filters.source_type:
            extra_where += f" AND dm.source_type = '{filters.source_type}'"
            extra_where_no_cd += f" AND dm.source_type = '{filters.source_type}'"
        if filters.classification_status:
            extra_where += f" AND cd.classification_status = '{filters.classification_status}'"

    # ── Stage 1: Full join with classification_docs (production) ──────────────
    sql_full = f"""
        SELECT
            de.document_id,
            de.chunk_index,
            de.chunk_text,
            dm.file_name,
            dm.source_type,
            dm.file_path,
            1 - (de.embedding <=> '{vec_str}'::halfvec(3072)) AS similarity_score,
            cd.document_type,
            cd.classification_status,
            cd.output_document_path,
            dm.current_timestamp_ist::text  AS ingested_at
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        JOIN classification_docs cd ON cd.document_id = de.document_id
        WHERE de.embedding IS NOT NULL
          AND de.embedding_status = 'generated'
          AND cd.classification_status IN ('auto_classified', 'human_approved')
          {extra_where}
        ORDER BY similarity_score DESC
        LIMIT {CHUNK_CANDIDATES}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Force sequential scan — bypasses the HNSW index which can return 0
        # results when the vector count is low (<1000) or when the query vector
        # is far from stored vectors in the ANN graph traversal.
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute(sql_full)
        results = [dict(r) for r in cur.fetchall()]

    if results:
        log.info(f"  Stage 1 (with classification join) returned {len(results)} chunk(s)")
        return results

    # ── Stage 2: No classification join (fallback for dev/testing) ────────────
    # Stage 1 returned 0 — classification_docs either has no matching record
    # or the status is not auto_classified/human_approved.
    # Log diagnostic to explain why, then run without the classification join
    # so the endpoint is still usable during development.
    log.warning(
        "  SEARCH: Stage 1 (with classification join) returned 0 results. "
        "Running diagnostic and falling back to Stage 2 (embeddings only)..."
    )
    _log_db_state(conn)

    sql_fallback = f"""
        SELECT
            de.document_id,
            de.chunk_index,
            de.chunk_text,
            dm.file_name,
            dm.source_type,
            dm.file_path,
            1 - (de.embedding <=> '{vec_str}'::halfvec(3072)) AS similarity_score,
            NULL::text  AS document_type,
            dm.processing_status AS classification_status,
            NULL::text  AS output_document_path,
            dm.current_timestamp_ist::text  AS ingested_at
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        WHERE de.embedding IS NOT NULL
          AND de.embedding_status = 'generated'
          {extra_where_no_cd}
        ORDER BY similarity_score DESC
        LIMIT {CHUNK_CANDIDATES}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute(sql_fallback)
        fallback = [dict(r) for r in cur.fetchall()]

    if fallback:
        log.warning(
            f"  SEARCH: Stage 2 fallback returned {len(fallback)} chunk(s). "
            "Results are from embeddings without classification validation. "
            "Fix: re-run POST /api/classify/batch so documents are auto_classified."
        )
    return fallback


# ══════════════════════════════════════════════════════════════════
#  DEDUPLICATION
# ══════════════════════════════════════════════════════════════════

def _deduplicate_to_documents(chunks: list) -> list:
    """
    Group chunks by document. Keep the highest-scoring chunk per document.
    That chunk's text becomes the displayed excerpt in the response.
    Returns top FINAL_TOP_K documents sorted by similarity score descending.
    """
    best: dict = {}
    for row in chunks:
        doc_id = str(row["document_id"])
        score  = float(row["similarity_score"])
        if doc_id not in best or score > float(best[doc_id]["similarity_score"]):
            best[doc_id] = row
    return sorted(
        best.values(),
        key=lambda r: float(r["similarity_score"]),
        reverse=True,
    )[:FINAL_TOP_K]


# ══════════════════════════════════════════════════════════════════
#  Q&A ANSWER GENERATION
# ══════════════════════════════════════════════════════════════════

def _generate_answer(query: str, top_docs: list) -> str:
    """
    Build context from top matched documents and ask GPT-4o-mini
    to generate a direct answer with source citations.

    Context structure:
      [Source: invoice_001.pdf | Category: Invoices]
      Subtotal: $2050. Tax (18%): $369. Grand Total: $2419.

      ---

      [Source: invoice_002.pdf | Category: Invoices]
      Cloud Hosting Service: 2 units at $500 each...

    GPT is instructed to cite source documents for every claim.
    """
    context_docs = top_docs[:QA_CONTEXT_DOCS]

    context_parts = []
    for doc in context_docs:
        source   = doc.get("file_name", "Unknown")
        category = doc.get("document_type", "Unknown")
        excerpt  = (doc.get("chunk_text") or "").strip()
        context_parts.append(
            f"[Source: {source} | Category: {category}]\n{excerpt}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a Document Intelligence assistant. "
        "Answer questions based ONLY on the provided source documents. "
        "For every specific fact, number, or claim, cite the source document "
        "in parentheses — e.g. (sample_invoice_document.pdf). "
        "If the documents do not contain enough information to answer, "
        "say: 'The provided documents do not contain enough information to answer this question.' "
        "Be concise and factual."
    )

    user = f"Source documents:\n\n{context}\n\nQuestion: {query}"

    response = _search_client.chat.completions.create(
        model=config.AI_CLASSIFICATION_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════
#  FASTAPI ENDPOINT
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/search/semantic",
    summary="Semantic Search and Context-Aware Q&A",
    description=(
        "Two modes:\n\n"
        "**mode='search'** — Find documents semantically related to the query. "
        "Returns a ranked list of matching documents with excerpts.\n\n"
        "**mode='qa'** — Ask a specific question about documents. "
        "Retrieves the most relevant passages and generates a direct answer with citations."
    ),
)
def semantic_search(request: SearchRequest):
    """
    POST /api/search/semantic

    Accepts a natural language query and returns either:
      - A ranked list of matching documents (mode='search')
      - A GPT-generated answer with source citations (mode='qa')

    Internally:
      1. Embeds the query (standard) or a hypothetical answer (hyde)
      2. Searches pgvector with halfvec cosine similarity (top-K)
      3. Deduplicates chunks to document level
      4. Returns document list (search) or LLM answer (qa)
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query field must not be empty.")

    # Validate mode
    mode = request.mode.lower()
    if mode not in ("search", "qa"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be 'search' or 'qa'."
        )

    # Auto-select search_mode if not specified
    # search → standard (user is browsing, not asking a specific question)
    # qa     → hyde (best for extracting specific facts from documents)
    if request.search_mode:
        search_mode = request.search_mode.lower()
        if search_mode not in ("standard", "hyde"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search_mode '{search_mode}'. Must be 'standard' or 'hyde'."
            )
    else:
        search_mode = "hyde" if mode == "qa" else "standard"

    filters = request.filters

    log.info(f"Search: mode={mode}, search_mode={search_mode}, query='{query[:60]}'")

    try:
        # ── Step 1: Embed ──────────────────────────────────────────────────────
        if search_mode == "hyde":
            log.info("  Generating HyDE hypothetical answer...")
            text_to_embed = _generate_hypothetical_answer(query)
            log.info(f"  HyDE: '{text_to_embed[:80]}...'")
        else:
            text_to_embed = query

        vec     = _embed(text_to_embed)
        vec_str = _vec_to_str(vec)

        # ── Step 2: pgvector search ────────────────────────────────────────────
        conn = get_connection()
        try:
            log.info(f"  Running pgvector search (top-{CHUNK_CANDIDATES})...")
            chunks = _run_vector_search(conn, vec_str, filters)
            log.info(f"  pgvector returned {len(chunks)} chunk(s)")
        finally:
            conn.close()

        # ── Step 3: Deduplication ──────────────────────────────────────────────
        top_docs = _deduplicate_to_documents(chunks)
        log.info(f"  Deduplicated to {len(top_docs)} document(s)")

        if not top_docs:
            log.info("  No matching documents found after deduplication.")

        # ── Step 4a: Semantic search response ─────────────────────────────────
        if mode == "search":
            results = []
            for rank, doc in enumerate(top_docs, 1):
                excerpt = (doc.get("chunk_text") or "").strip().replace("\n", " ")
                results.append(SearchResultItem(
                    rank                  = rank,
                    similarity_score      = round(float(doc["similarity_score"]), 4),
                    file_name             = doc["file_name"],
                    document_type         = doc.get("document_type"),
                    source_type           = doc["source_type"],
                    classification_status = doc.get("classification_status"),
                    output_document_path  = doc.get("output_document_path"),
                    matched_chunk_index   = doc["chunk_index"],
                    matched_excerpt       = excerpt[:500],
                    ingested_at           = doc.get("ingested_at"),
                ))
            return SemanticSearchResponse(
                query         = query,
                mode          = mode,
                search_mode   = search_mode,
                filters       = filters.model_dump() if filters else None,
                total_results = len(results),
                results       = results,
            )

        # ── Step 4b: Q&A response ──────────────────────────────────────────────
        else:
            if not top_docs:
                answer = "No relevant documents were found for your query."
            else:
                log.info(f"  Generating answer from top {min(QA_CONTEXT_DOCS, len(top_docs))} doc(s)...")
                answer = _generate_answer(query, top_docs)

            citations = []
            for doc in top_docs[:QA_CONTEXT_DOCS]:
                excerpt = (doc.get("chunk_text") or "").strip().replace("\n", " ")
                citations.append(Citation(
                    file_name            = doc["file_name"],
                    document_type        = doc.get("document_type"),
                    similarity_score     = round(float(doc["similarity_score"]), 4),
                    matched_excerpt      = excerpt[:500],
                    output_document_path = doc.get("output_document_path"),
                ))

            return QAResponse(
                query       = query,
                mode        = mode,
                search_mode = search_mode,
                filters     = filters.model_dump() if filters else None,
                answer      = answer,
                citations   = citations,
            )

    except HTTPException:
        raise
    except Exception as exc:
        log.exception(f"Search failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}"
        )