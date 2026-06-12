"""
search_debug.py  — Development and debug script for vector search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Location: root of the project (same level as embed_query.py, main.py)

PURPOSE
───────
This is a STANDALONE command-line debug script. It is NOT part of the
FastAPI app. Use it to verify that your vector embeddings and pgvector
search are working correctly before testing via the /api/search/semantic
endpoint.

For the production API endpoint → see routers/search.py

WHAT IT DOES
────────────
1. Embeds a query (standard or HyDE mode)
2. Searches pgvector directly with the embedded vector
3. Deduplicates chunks → documents
4. Prints ranked results with similarity scores and matched excerpts

SEARCH_MODE settings
─────────────────────
  "standard" → embed the query directly
               score range: 0.50–0.65 for good Q→D matches
               best for: semantic document search (topic browsing)

  "hyde"     → GPT-4o-mini writes a hypothetical answer → embed that
               score range: 0.63–0.72 for good Q→D matches
               best for: context-aware Q&A (extracting specific facts)

RETRIEVAL_MODE settings
────────────────────────
  "threshold" → only return chunks with score >= SIMILARITY_THRESHOLD
                risk: correct chunks near threshold may be missed

  "topk"      → always return top CHUNK_CANDIDATES chunks regardless of score
                recommended: no relevant results are silently filtered out

DEBUG_MODE settings
────────────────────
  True  → search ALL embedded docs (no classification_docs filter)
          use while testing with few documents
  False → search only auto_classified / human_approved docs
          use for production-like testing

HOW TO RUN
──────────
  cd /mnt/d/Azure-DIC-updated/document_intelligence
  python search_debug.py

  Then edit SEARCH_MODE / RETRIEVAL_MODE / QUERY at the bottom and re-run.

SCORE INTERPRETATION
─────────────────────
  text-embedding-3-large bi-encoder Q→D cosine similarity ranges:
    0.65+  Very good match (HyDE mode)
    0.55+  Good match (standard mode)
    0.45+  Moderate match
    < 0.45 Weak match (may not be relevant)

  The ranking matters more than the absolute score.
  If chunk #1 (the correct chunk) always scores highest, retrieval is working.

SISTER SCRIPTS
──────────────
  embed_query.py  → embed a query and print raw vector + pgAdmin SQL
  search_debug.py → this file (test full search pipeline)
"""

import psycopg2
import psycopg2.extras
from openai import AzureOpenAI
import os

# ── Config (matches config.py) ─────────────────────────────────────────────────
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
OPENAI_API_VER  = "2024-02-01"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL      = "gpt-4o-mini"
EMBEDDING_DIMS  = 3072

PG_HOST = "docint-pgserver.postgres.database.azure.com"
PG_DB   = "int-doc-class"
PG_USER = "docintadmin"
PG_PASS = os.getenv("PG_PASS")
PG_PORT = "5432"


# ══════════════════════════════════════════════════════════════════
#  AZURE OPENAI HELPERS
# ══════════════════════════════════════════════════════════════════

def _client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        api_version=OPENAI_API_VER,
    )


def embed_text(text: str, label: str = "text") -> list:
    """Embed any text with text-embedding-3-large (3072 dims)."""
    response = _client().embeddings.create(input=text, model=EMBEDDING_MODEL)
    vec = response.data[0].embedding
    assert len(vec) == EMBEDDING_DIMS, f"Dim mismatch: {len(vec)} != {EMBEDDING_DIMS}"
    print(f"  Embedded {label} ({len(vec)} dims)  first 3: {[round(v, 6) for v in vec[:3]]}")
    return vec


def vec_to_str(vec: list) -> str:
    """
    Convert Python float list to PostgreSQL halfvec literal.
    repr() avoids scientific notation (e.g. 1.234e-07) that halfvec rejects.
    """
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


# ══════════════════════════════════════════════════════════════════
#  HYDE — Hypothetical Document Embeddings
# ══════════════════════════════════════════════════════════════════

def generate_hypothetical_answer(query: str) -> str:
    """
    HyDE technique: instead of embedding the QUESTION, ask GPT-4o-mini
    to write a short realistic ANSWER, then embed that answer.

    Why it works:
      Question space:  "What is the grand total on the invoice?"
      Document space:  "Grand Total: $2,419."
      Gap:             cosine ~0.58 (bi-encoder Q→D)

      HyDE answer:     "The grand total on the invoice from ABC Technologies
                        is $2,419 including 18% GST of $369."
      Both answer and chunk are in DOCUMENT space → cosine ~0.66

    The answer does not need to be factually correct.
    Its embedding just needs to land near the real chunk in vector space.
    """
    system = (
        "You are a document retrieval assistant. "
        "Given a question about a business document (invoice, contract, compliance, etc.), "
        "write a realistic 2–3 sentence excerpt that answers it as if quoting directly "
        "from the document. Be specific with amounts, names, and dates when you can. "
        "Output ONLY the document excerpt — no preamble, no 'the answer is'."
    )
    response = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Write a document excerpt that answers: {query}"},
        ],
        temperature=0.1,
        max_tokens=150,
    )
    answer = response.choices[0].message.content.strip()
    print(f"  HyDE answer: \"{answer[:110]}{'...' if len(answer) > 110 else ''}\"")
    return answer


# ══════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════

def get_connection():
    return psycopg2.connect(
        host=PG_HOST, dbname=PG_DB, user=PG_USER,
        password=PG_PASS, port=PG_PORT,
        sslmode="require",
    )


# ══════════════════════════════════════════════════════════════════
#  DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════

def show_embedded_documents(conn) -> None:
    """Print a table of all documents currently in document_embeddings."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT dm.file_name, dm.source_type, de.embedding_status,
                   COUNT(de.chunk_index) AS chunk_count
            FROM document_embeddings de
            JOIN document_metadata   dm ON dm.document_id = de.document_id
            GROUP BY dm.file_name, dm.source_type, de.embedding_status
            ORDER BY chunk_count DESC
        """)
        rows = cur.fetchall()

    print(f"\n  {'File name':<45} {'Source':<15} {'Status':<25} Chunks")
    print("  " + "─" * 100)
    for r in rows:
        print(
            f"  {r['file_name'][:44]:<45}"
            f" {r['source_type']:<15}"
            f" {r['embedding_status']:<25}"
            f" {r['chunk_count']}"
        )


def show_raw_scores(conn, vec_str: str) -> None:
    """
    Show cosine similarity scores for ALL embedded chunks with NO threshold filter.

    Check: does the correct chunk rank #1?
    If yes, retrieval logic is working — only the threshold may need adjusting.
    If no, there is a deeper chunking or embedding quality issue.
    """
    sql = f"""
        SELECT
            dm.file_name,
            de.chunk_index,
            LEFT(de.chunk_text, 60)  AS preview,
            ROUND(
                (1 - (de.embedding <=> '{vec_str}'::halfvec(3072)))::numeric, 4
            ) AS score
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        WHERE de.embedding IS NOT NULL
          AND de.embedding_status = 'generated'
        ORDER BY de.embedding <=> '{vec_str}'::halfvec(3072) ASC
        LIMIT 15
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    print(f"\n  RAW SCORES — no threshold (check that correct chunk is ranked #1):")
    print(f"  {'File':<32} {'Chunk':>6} {'Score':>8}  {'Preview'}")
    print("  " + "─" * 95)

    if not rows:
        print("  No rows returned — halfvec comparison failed.")
        print("  Check that the pgvector extension is installed and enabled.")
        return

    for i, r in enumerate(rows):
        score   = float(r["score"])
        bar     = "█" * int(score * 20)
        preview = (r["preview"] or "").replace("\n", " ")
        rank    = "← #1" if i == 0 else ""
        print(f"  {r['file_name'][:31]:<32} #{r['chunk_index']:>4} {score:>8.4f}"
              f"  {bar:<14} {preview}  {rank}")


# ══════════════════════════════════════════════════════════════════
#  PGVECTOR SEARCH
# ══════════════════════════════════════════════════════════════════

def _threshold_clause(vec_str: str, threshold: float) -> str:
    return f"AND 1 - (de.embedding <=> '{vec_str}'::halfvec(3072)) >= {threshold}"


def search_all_docs(conn, vec_str: str, retrieval_mode: str,
                    threshold: float, top_n: int) -> list:
    """
    DEBUG mode: search ALL embedded documents.
    No join to classification_docs — includes all embedding_status='generated'.
    Use while testing with a small number of documents.
    """
    threshold_sql = _threshold_clause(vec_str, threshold) if retrieval_mode == "threshold" else ""
    sql = f"""
        SELECT
            de.document_id,
            de.chunk_index,
            de.chunk_text,
            dm.file_name,
            dm.source_type,
            dm.file_path,
            1 - (de.embedding <=> '{vec_str}'::halfvec(3072)) AS similarity_score
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        WHERE de.embedding IS NOT NULL
          AND de.embedding_status = 'generated'
          {threshold_sql}
        ORDER BY de.embedding <=> '{vec_str}'::halfvec(3072) ASC
        LIMIT {top_n}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


def search_finalized_docs(conn, vec_str: str, retrieval_mode: str,
                          threshold: float, top_n: int, filters: dict) -> list:
    """
    PRODUCTION mode: search only auto_classified and human_approved documents.
    Applies optional metadata filters (document_type, source_type, classification_status).
    """
    extra = ""
    if filters.get("document_type"):
        extra += f" AND cd.document_type = '{filters['document_type']}'"
    if filters.get("source_type"):
        extra += f" AND dm.source_type = '{filters['source_type']}'"
    if filters.get("classification_status"):
        extra += f" AND cd.classification_status = '{filters['classification_status']}'"

    threshold_sql = _threshold_clause(vec_str, threshold) if retrieval_mode == "threshold" else ""

    sql = f"""
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
            cd.combined_confidence_score   AS classification_confidence,
            dm.current_timestamp_ist       AS ingested_at
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        JOIN classification_docs cd ON cd.document_id = de.document_id
        WHERE de.embedding IS NOT NULL
          AND de.embedding_status = 'generated'
          AND cd.classification_status IN ('auto_classified', 'human_approved')
          {threshold_sql}
          {extra}
        ORDER BY de.embedding <=> '{vec_str}'::halfvec(3072) ASC
        LIMIT {top_n}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════
#  DEDUPLICATION
# ══════════════════════════════════════════════════════════════════

def deduplicate_to_documents(chunk_results: list, final_top_k: int) -> list:
    """
    Group chunks by document. Keep the highest-scoring chunk per document.
    That chunk's text becomes the displayed excerpt.
    Return top final_top_k documents sorted by score descending.
    """
    best: dict = {}
    for row in chunk_results:
        doc_id = str(row["document_id"])
        score  = float(row["similarity_score"])
        if doc_id not in best or score > float(best[doc_id]["similarity_score"]):
            best[doc_id] = row
    return sorted(best.values(), key=lambda r: float(r["similarity_score"]),
                  reverse=True)[:final_top_k]


# ══════════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ══════════════════════════════════════════════════════════════════

def print_results(query: str, filters: dict, results: list,
                  search_mode: str, retrieval_mode: str,
                  debug_mode: bool, threshold: float, top_n: int) -> None:
    print()
    print("═" * 70)
    print(f"  QUERY         : {query}")
    print(f"  SEARCH_MODE   : {search_mode.upper()}")
    if retrieval_mode == "threshold":
        print(f"  RETRIEVAL     : THRESHOLD (score >= {threshold})")
    else:
        print(f"  RETRIEVAL     : TOP-K ({top_n} candidates)")
    print(f"  SCOPE         : {'DEBUG — all docs' if debug_mode else 'PRODUCTION — finalized only'}")
    if filters:
        print(f"  FILTERS       : {filters}")
    print(f"  RESULTS       : {len(results)} document(s)")
    print("═" * 70)

    if not results:
        print("\n  No results found.")
        if retrieval_mode == "threshold":
            print(f"  → Switch RETRIEVAL_MODE = 'topk' to bypass the threshold")
        print(f"  → Try SEARCH_MODE = 'hyde' for higher similarity scores")
        return

    for i, r in enumerate(results, 1):
        score = float(r["similarity_score"])
        bar   = "█" * int(score * 20)
        print(f"\n  [{i}] {r['file_name']}")
        print(f"       Score    : {score:.4f}  [{bar}]")
        print(f"       Source   : {r['source_type']}")
        if "document_type" in r:
            print(f"       Category : {r.get('document_type', 'N/A')}")
            print(f"       Status   : {r.get('classification_status', 'N/A')}")
            if r.get("output_document_path"):
                print(f"       DIC Path : {r['output_document_path']}")
        print(f"       Chunk #  : {r['chunk_index']}")
        excerpt = (r.get("chunk_text") or "").strip().replace("\n", " ")
        if len(excerpt) > 400:
            excerpt = excerpt[:400] + "..."
        print(f"       Excerpt  : \"{excerpt}\"")
    print()


# ══════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_search(
    query:          str,
    search_mode:    str  = "standard",
    retrieval_mode: str  = "topk",
    debug_mode:     bool = True,
    threshold:      float = 0.50,
    chunk_candidates: int = 20,
    final_top_k:    int  = 5,
    filters:        dict = None,
) -> None:
    """
    Full vector search pipeline.

    Args:
        query           : natural language search query
        search_mode     : "standard" (embed query) or "hyde" (embed hypothetical answer)
        retrieval_mode  : "topk" (best N) or "threshold" (score >= threshold)
        debug_mode      : True = all docs, False = finalized only
        threshold       : cosine threshold (only for retrieval_mode="threshold")
        chunk_candidates: number of chunks to fetch from pgvector before dedup
        final_top_k     : number of documents to return after deduplication
        filters         : optional dict with document_type, source_type, classification_status
    """
    if filters is None:
        filters = {}

    print(f"\nQuery: \"{query}\"")
    print(f"Mode : {search_mode.upper()}  |  {retrieval_mode.upper()}"
          f"  |  DEBUG={debug_mode}")

    # ── Step 1: Embed ──────────────────────────────────────────────────────────
    if search_mode == "hyde":
        print("\n  Generating hypothetical answer (HyDE)...")
        text_to_embed = generate_hypothetical_answer(query)
        embed_label   = "hypothetical answer"
    else:
        text_to_embed = query
        embed_label   = "query"

    vec     = embed_text(text_to_embed, label=embed_label)
    vec_str = vec_to_str(vec)

    # ── Step 2: Connect + diagnostics ─────────────────────────────────────────
    conn = get_connection()
    try:
        print("\n  ── Documents in embedding table ──")
        show_embedded_documents(conn)
        show_raw_scores(conn, vec_str)

        # ── Step 3: pgvector search ────────────────────────────────────────────
        print(f"\n  Running {retrieval_mode.upper()} search...")
        if debug_mode:
            chunks = search_all_docs(conn, vec_str, retrieval_mode,
                                     threshold, chunk_candidates)
        else:
            chunks = search_finalized_docs(conn, vec_str, retrieval_mode,
                                           threshold, chunk_candidates, filters)

        print(f"  pgvector returned : {len(chunks)} chunk(s)")

        # ── Step 4: Deduplication ──────────────────────────────────────────────
        docs = deduplicate_to_documents(chunks, final_top_k)
        print(f"  After dedup       : {len(docs)} document(s)")

        # ── Step 5: Print ──────────────────────────────────────────────────────
        print_results(query, filters, docs, search_mode, retrieval_mode,
                      debug_mode, threshold, chunk_candidates)

    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
#  EDIT THESE SETTINGS AND RUN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Query ─────────────────────────────────────────────────────────────────
    QUERY = "What is the grand total on the invoice from ABC Technologies?"

    # ── Search mode ───────────────────────────────────────────────────────────
    # "standard" → embed query directly        (score range 0.50–0.65)
    # "hyde"     → embed GPT-generated answer  (score range 0.63–0.72)
    SEARCH_MODE = "standard"

    # ── Retrieval mode ────────────────────────────────────────────────────────
    # "topk"      → always return top CHUNK_CANDIDATES results (recommended)
    # "threshold" → only return results with score >= SIMILARITY_THRESHOLD
    RETRIEVAL_MODE = "topk"

    # ── Scope ─────────────────────────────────────────────────────────────────
    # True  → search all embedded docs (use while testing)
    # False → search only auto_classified / human_approved (production-like)
    DEBUG_MODE = True

    # ── Retrieval parameters ──────────────────────────────────────────────────
    SIMILARITY_THRESHOLD = 0.50   # only used when RETRIEVAL_MODE="threshold"
    CHUNK_CANDIDATES     = 20     # chunks fetched from pgvector before dedup
    FINAL_TOP_K          = 5      # documents returned after deduplication

    # ── Filters (only applied when DEBUG_MODE=False) ──────────────────────────
    FILTERS = {}
    # Example: FILTERS = {"document_type": "Invoices", "source_type": "manual_upload"}

    run_search(
        query           = QUERY,
        search_mode     = SEARCH_MODE,
        retrieval_mode  = RETRIEVAL_MODE,
        debug_mode      = DEBUG_MODE,
        threshold       = SIMILARITY_THRESHOLD,
        chunk_candidates = CHUNK_CANDIDATES,
        final_top_k     = FINAL_TOP_K,
        filters         = FILTERS,
    )