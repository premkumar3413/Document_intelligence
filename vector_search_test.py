
"""
vector_search_test.py  (v4 — SEARCH_MODE: standard | hyde)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THE SCORE IS 0.58 AND NOT 0.80+ (important to understand)
──────────────────────────────────────────────────────────────
text-embedding-3-large is a BI-ENCODER. Bi-encoders produce different
score ranges depending on WHAT you are comparing:

  D→D  Document vs Document   0.75–0.92  ← 0.70 threshold is for THIS
  Q→D  Question vs Document   0.50–0.65  ← you are here

Score 0.58 for a correct Q→D match IS a good score. The right chunk
ranked #1 out of 7. The retrieval is working correctly.

WHAT IS SEARCH_MODE
──────────────────────────────────────────────────────────────
SEARCH_MODE = "standard"
  Embed the query directly.  Score range: 0.50–0.65.
  Use for: Semantic Search (find relevant documents by topic).

SEARCH_MODE = "hyde"
  HyDE — Hypothetical Document Embeddings (Gao et al. 2022).
  Ask GPT-4o-mini to write a realistic ANSWER to the query.
  Embed that answer instead of the question.
  Because the answer is in document space (not question space),
  cosine similarity jumps to 0.65–0.78 for correct matches.
  Use for: Context-Aware Q&A (extract specific facts from documents).

WHAT IS RETRIEVAL_MODE
──────────────────────────────────────────────────────────────
RETRIEVAL_MODE = "threshold"
  Only return chunks where score >= SIMILARITY_THRESHOLD.
  Risk: a correct chunk with score 0.49 gets filtered out.

RETRIEVAL_MODE = "topk"
  Return the top CHUNK_CANDIDATES chunks regardless of score.
  Deduplicate to FINAL_TOP_K documents.
  Recommended for production — no relevant results are filtered out.

DEBUG_MODE = True   → search all embedded docs (no classification filter)
DEBUG_MODE = False  → production: only auto_classified / human_approved
"""

import psycopg2
import psycopg2.extras
from openai import AzureOpenAI
import os
from config import AI_TEMPERATURE

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

SEARCH_MODE          = "hyde"   # "standard" or "hyde"
RETRIEVAL_MODE       = "topk"  # "threshold" or "topk"
SIMILARITY_THRESHOLD = 0.50         # used when RETRIEVAL_MODE="threshold"
CHUNK_CANDIDATES     = 20           # chunks fetched from pgvector (for both modes)
FINAL_TOP_K          = 5            # final documents returned after dedup
DEBUG_MODE           = True         # True=no classification filter, False=prod


def _client() -> AzureOpenAI:
    return AzureOpenAI(azure_endpoint=OPENAI_ENDPOINT, api_key=OPENAI_KEY,
                       api_version=OPENAI_API_VER)

def get_connection():
    return psycopg2.connect(host=PG_HOST, dbname=PG_DB, user=PG_USER,
                            password=PG_PASS, port=PG_PORT, sslmode="require")

def embed_text(text: str, label: str = "text") -> list:
    response = _client().embeddings.create(input=text, model=EMBEDDING_MODEL)
    vec = response.data[0].embedding
    assert len(vec) == EMBEDDING_DIMS
    print(f"  Embedded {label} ({len(vec)} dims)  first 3: {[round(v,6) for v in vec[:3]]}")
    return vec

def vec_to_str(vec: list) -> str:
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


# ── HyDE ──────────────────────────────────────────────────────────────────────

def generate_hypothetical_answer(query: str) -> str:
    """
    Hypothetical Document Embeddings:
      Problem:  query is in QUESTION space → score 0.50-0.65 vs document space
      Solution: generate a plausible ANSWER → embed that → now in document space
                → score 0.65-0.78

    The answer does not need to be factually correct. Its embedding just needs
    to land near the real answer chunk in the vector space.
    """
    system = (
        "You are a document retrieval assistant. "
        "Given a question about a business document, write a realistic 2-3 sentence "
        "excerpt that answers it as if quoting from the document. "
        "Be specific with numbers and terminology. "
        "Output ONLY the excerpt — no preamble."
    )
    response = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Write an excerpt answering: {query}"}],
        temperature=AI_TEMPERATURE, max_tokens=120,
    )
    answer = response.choices[0].message.content.strip()
    print(f'  HyDE answer: "{answer[:100]}{"..." if len(answer) > 100 else ""}')
    return answer


# ── Diagnostics ────────────────────────────────────────────────────────────────

def show_embedded_documents(conn) -> None:
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
    print("  " + "-" * 100)
    for r in rows:
        print(f"  {r['file_name'][:44]:<45} {r['source_type']:<15}"
              f" {r['embedding_status']:<25} {r['chunk_count']}")


def show_raw_scores(conn, vec_str: str) -> None:
    sql = f"""
        SELECT dm.file_name, de.chunk_index,
               LEFT(de.chunk_text, 55) AS preview,
               ROUND((1-(de.embedding <=> '{vec_str}'::halfvec(3072)))::numeric,4) AS score
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        WHERE de.embedding IS NOT NULL AND de.embedding_status = 'generated'
        ORDER BY de.embedding <=> '{vec_str}'::halfvec(3072) ASC LIMIT 15
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql); rows = cur.fetchall()
    print(f"\n  RAW SCORES (no threshold — correct chunk should rank #1):")
    print(f"  {'File':<32} {'Chunk':>6} {'Score':>8}  Preview")
    print("  " + "-" * 90)
    if not rows:
        print("  No rows — halfvec comparison failed.")
        return
    for r in rows:
        score = float(r["score"])
        bar   = "X" * int(score * 20)
        print(f"  {r['file_name'][:31]:<32} #{r['chunk_index']:>4} {score:>8.4f}"
              f"  {bar:<14} {(r['preview'] or '').replace(chr(10),' ')}")


# ── pgvector search ────────────────────────────────────────────────────────────

def _threshold_clause(vec_str: str) -> str:
    if RETRIEVAL_MODE == "threshold":
        return f"AND 1-(de.embedding <=> '{vec_str}'::halfvec(3072)) >= {SIMILARITY_THRESHOLD}"
    return ""


def search_debug(conn, vec_str: str) -> list:
    sql = f"""
        SELECT de.document_id, de.chunk_index, de.chunk_text,
               dm.file_name, dm.source_type, dm.file_path,
               1-(de.embedding <=> '{vec_str}'::halfvec(3072)) AS similarity_score
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        WHERE de.embedding IS NOT NULL AND de.embedding_status = 'generated'
          {_threshold_clause(vec_str)}
        ORDER BY de.embedding <=> '{vec_str}'::halfvec(3072) ASC LIMIT {CHUNK_CANDIDATES}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql); return [dict(r) for r in cur.fetchall()]


def search_production(conn, vec_str: str, filters: dict) -> list:
    extra = ""
    if filters.get("document_type"):
        extra += f" AND cd.document_type = '{filters['document_type']}'"
    if filters.get("source_type"):
        extra += f" AND dm.source_type = '{filters['source_type']}'"
    if filters.get("classification_status"):
        extra += f" AND cd.classification_status = '{filters['classification_status']}'"

    sql = f"""
        SELECT de.document_id, de.chunk_index, de.chunk_text,
               dm.file_name, dm.source_type, dm.file_path,
               1-(de.embedding <=> '{vec_str}'::halfvec(3072)) AS similarity_score,
               cd.document_type, cd.classification_status, cd.output_document_path,
               cd.combined_confidence_score AS classification_confidence,
               cd.classification_tier, dm.current_timestamp_ist AS ingested_at
        FROM document_embeddings de
        JOIN document_metadata   dm ON dm.document_id = de.document_id
        JOIN classification_docs cd ON cd.document_id = de.document_id
        WHERE de.embedding IS NOT NULL AND de.embedding_status = 'generated'
          AND cd.classification_status IN ('auto_classified', 'human_approved')
          {_threshold_clause(vec_str)} {extra}
        ORDER BY de.embedding <=> '{vec_str}'::halfvec(3072) ASC LIMIT {CHUNK_CANDIDATES}
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql); return [dict(r) for r in cur.fetchall()]


def deduplicate(chunks: list) -> list:
    best = {}
    for row in chunks:
        doc_id = str(row["document_id"])
        if doc_id not in best or float(row["similarity_score"]) > float(best[doc_id]["similarity_score"]):
            best[doc_id] = row
    return sorted(best.values(), key=lambda r: float(r["similarity_score"]), reverse=True)[:FINAL_TOP_K]


def print_results(query, filters, results, filter_label, search_mode):
    print()
    print("=" * 70)
    print(f"  QUERY        : {query}")
    print(f"  SEARCH_MODE  : {search_mode.upper()}")
    retrieval_info = (f"THRESHOLD={SIMILARITY_THRESHOLD}" if RETRIEVAL_MODE=="threshold"
                      else f"TOP-{CHUNK_CANDIDATES}")
    print(f"  RETRIEVAL    : {RETRIEVAL_MODE.upper()} ({retrieval_info})")
    print(f"  MODE         : {filter_label}")
    if filters: print(f"  FILTERS      : {filters}")
    print(f"  RESULTS      : {len(results)} document(s)")
    print("=" * 70)

    if not results:
        print("\n  No results found.")
        if RETRIEVAL_MODE == "threshold":
            print(f"  → Lower SIMILARITY_THRESHOLD (currently {SIMILARITY_THRESHOLD})")
            print(f"  → Or switch RETRIEVAL_MODE = \'topk\'")
        print(f"  → Or try SEARCH_MODE = \'hyde\'")
        return

    for i, r in enumerate(results, 1):
        score = float(r["similarity_score"])
        bar   = "X" * int(score * 20)
        print(f"\n  [{i}] {r['file_name']}")
        print(f"       Score   : {score:.4f}  [{bar}]")
        print(f"       Source  : {r['source_type']}")
        if "document_type" in r:
            print(f"       Category: {r.get('document_type', 'N/A')}")
            print(f"       Status  : {r.get('classification_status', 'N/A')}")
            print(f"       Path    : {r.get('output_document_path', 'N/A')}")
        print(f"       Chunk # : {r['chunk_index']}")
        excerpt = (r.get("chunk_text") or "").strip().replace("\n", " ")
        if len(excerpt) > 400: excerpt = excerpt[:400] + "..."
        print(f"       Excerpt : \"{excerpt}\"")
    print()


def run_search(query: str, filters: dict = None) -> None:
    if filters is None: filters = {}
    print(f"\nQuery: \"{query}\"")
    print(f"Mode : {SEARCH_MODE.upper()}  |  DEBUG_MODE={DEBUG_MODE}  |  {RETRIEVAL_MODE.upper()}")

    if SEARCH_MODE == "hyde":
        print("\n  Generating hypothetical answer (HyDE)...")
        hypothetical = generate_hypothetical_answer(query)
        vec          = embed_text(hypothetical, label="hypothetical answer")
        search_label = "HyDE"
    else:
        vec          = embed_text(query, label="query")
        search_label = "Standard"

    vec_str = vec_to_str(vec)
    conn    = get_connection()
    try:
        print("\n  --- Documents in embedding table ---")
        show_embedded_documents(conn)
        show_raw_scores(conn, vec_str)

        print(f"\n  Running search...")
        chunks = search_debug(conn, vec_str) if DEBUG_MODE else search_production(conn, vec_str, filters)
        print(f"  pgvector returned : {len(chunks)} chunk(s)")
        docs = deduplicate(chunks)
        print(f"  After dedup       : {len(docs)} document(s)")
        filter_label = "DEBUG (all docs)" if DEBUG_MODE else "PRODUCTION (finalized)"
        print_results(query, filters, docs, filter_label, search_label)
    finally:
        conn.close()


if __name__ == "__main__":

    # ── SETTINGS — edit these ──────────────────────────────────────────────────
    SEARCH_MODE    = "qa"   # "standard" or "hyde"
    RETRIEVAL_MODE = "topk"  # "threshold" or "topk"
    DEBUG_MODE     = True         # True=all docs, False=auto_classified/human_approved only

    QUERY   = "What is the grand total on the invoice from ABC Technologies?"
    FILTERS = {}
    # Examples: FILTERS = {"document_type": "Invoices", "source_type": "manual_upload"}

    run_search(query=QUERY, filters=FILTERS)