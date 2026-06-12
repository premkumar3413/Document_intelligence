"""
embed_query.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Script 1: Embed a query and print the vector to terminal.

Use this when you want to:
  1. Convert a query to a vector
  2. Copy that vector from the terminal
  3. Paste it manually into a pgAdmin SQL query

The vector is printed in PostgreSQL halfvec format:
  [0.012, -0.034, 0.056, ...]

Then copy it and run this SQL in pgAdmin:
  SELECT
      dm.file_name,
      de.chunk_index,
      de.chunk_text,
      1 - (de.embedding <=> '[0.012, -0.034, ...]'::halfvec) AS score
  FROM document_embeddings de
  JOIN document_metadata dm ON dm.document_id = de.document_id
  WHERE de.embedding IS NOT NULL
    AND de.embedding_status = 'generated'
  ORDER BY de.embedding <=> '[0.012, -0.034, ...]'::halfvec ASC
  LIMIT 10;

Usage: python embed_query.py
"""

from openai import AzureOpenAI
import os

# ── Config ─────────────────────────────────────────────────────────────────────
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY      = os.getenv("AZURE_OPENAI_KEY")
OPENAI_API_VER  = "2024-02-01"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS  = 3072

# ════════════════════════════════════════════════════════
#  EDIT THIS QUERY BEFORE RUNNING
# ════════════════════════════════════════════════════════
QUERY = "What is the total invoice amount including tax?"
# ════════════════════════════════════════════════════════


def embed_query(query: str) -> list[float]:
    """Convert natural language query to 3072-dim vector."""
    client = AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        api_version=OPENAI_API_VER,
    )
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    )
    vec = response.data[0].embedding
    assert len(vec) == EMBEDDING_DIMS, f"Dim mismatch: {len(vec)} != {EMBEDDING_DIMS}"
    return vec


if __name__ == "__main__":
    print(f"\nEmbedding query: \"{QUERY}\"")
    print("Calling Azure OpenAI text-embedding-3-large...")
    vec = embed_query(QUERY)

    # Format as PostgreSQL halfvec string
    vec_str = "[" + ",".join(str(v) for v in vec) + "]"

    print(f"\n{'='*70}")
    print(f"Query     : {QUERY}")
    print(f"Dims      : {len(vec)}")
    print(f"First 5   : {vec[:5]}")
    print(f"Last  5   : {vec[-5:]}")
    print(f"{'='*70}")
    print("\n✓ COPY THE VECTOR BELOW — paste it into your pgAdmin SQL query:\n")
    print(vec_str)
    print(f"\n{'='*70}")
    print("\n✓ READY-TO-RUN SQL — paste this entire block into pgAdmin:\n")
    print(f"""-- Step 1: Check what documents are embedded
SELECT
    dm.file_name,
    de.document_id,
    COUNT(de.chunk_index)  AS chunk_count,
    de.embedding_status
FROM document_embeddings de
JOIN document_metadata dm ON dm.document_id = de.document_id
GROUP BY de.document_id, dm.file_name, de.embedding_status
ORDER BY chunk_count DESC;

-- Step 2: Run vector similarity search (replace the vector below)
SELECT
    dm.file_name,
    dm.source_type,
    de.chunk_index,
    1 - (de.embedding <=> '{vec_str}'::halfvec) AS similarity_score,
    LEFT(de.chunk_text, 300)                     AS excerpt
FROM document_embeddings de
JOIN document_metadata   dm ON dm.document_id = de.document_id
WHERE de.embedding IS NOT NULL
  AND de.embedding_status = 'generated'
ORDER BY de.embedding <=> '{vec_str}'::halfvec ASC
LIMIT 10;""")
    print()