"""
config.py — Central configuration for the Document Intelligence platform.

Classification flow:
  rule_score == 0.0        → auto_miscellaneous (no AI, no HITL)
  rule_score >= 70%        → auto_classified    (rule only, no AI call)
  rule_score < 70%         → GPT-4o-mini → combine scores
  combined >= 70%          → auto_classified
  40% <= combined < 70%    → soft_flag  (human review recommended)
  combined < 40%           → hard_stop  (mandatory human review)

Embedding pipeline (after classification):
  Structure-aware extraction (PDF blocks / DOCX sections)
  → Paragraph/sentence/character chunking (max 800 chars)
  → Contextual Retrieval: GPT-4o-mini generates 2-sentence context per chunk
  → Embed (context + chunk) via text-embedding-3-large
  → Store original chunk_text + embedding in document_embeddings

─────────────────────────────────────────────────────────────────────────────
SECRET MANAGEMENT
─────────────────────────────────────────────────────────────────────────────
All secrets are read from environment variables — never hardcoded here.

For local development:
  1. Copy .env.example to .env in the project root.
  2. Fill in your real values in .env.
  3. .env is listed in .gitignore — it must never be committed.
  python-dotenv loads it automatically when the app starts (see load_dotenv()
  call below).

For production (Azure App Service / VM / Docker):
  Set the same variable names as OS environment variables through your
  hosting platform's configuration panel. python-dotenv's load_dotenv()
  is a no-op when the variables are already in the environment, so the
  same code works in both contexts without any changes.

Required environment variables (see .env.example):
  AZURE_BLOB_CONN_STR        — Azure Blob Storage connection string
  AZURE_OPENAI_KEY           — Azure OpenAI API key
  PG_PASS                    — PostgreSQL password
─────────────────────────────────────────────────────────────────────────────
"""
import os
import logging
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

log = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )
    return value


# =============================================================================
# Azure Blob Storage
# =============================================================================

AZURE_BLOB_CONN_STR = _require_env("AZURE_BLOB_CONN_STR")

SOURCE_CONN_STR = AZURE_BLOB_CONN_STR
SOURCE_CONTAINER = "csg-tfm-test"

DIC_CONN_STR = AZURE_BLOB_CONN_STR
DIC_CONTAINER = "csg-tfm-test"
DIC_ROOT_FOLDER = "DIC"


# =============================================================================
# Azure OpenAI
# =============================================================================

OPENAI_ENDPOINT = _require_env("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = _require_env("AZURE_OPENAI_KEY")

OPENAI_API_VER = os.getenv(
    "AZURE_OPENAI_API_VER",
    "2024-02-01"
)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMS = 3072


# =============================================================================
# Azure OpenAI LLM
# =============================================================================

LLM_AZURE_ENDPOINT = OPENAI_ENDPOINT
LLM_API_KEY = OPENAI_KEY

LLM_DEPLOYMENT = "gpt-4o-mini"

LLM_API_VERSION = os.getenv(
    "AZURE_OPENAI_LLM_API_VER",
    "2025-01-01-preview"
)


# =============================================================================
# Chunking
# =============================================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 80


# =============================================================================
# Contextual Retrieval
# =============================================================================

CONTEXTUAL_RETRIEVAL_ENABLED = False
CONTEXT_MAX_DOC_CHARS = 25000
CONTEXT_MODEL = "gpt-4o-mini"


# =============================================================================
# PDF Structure Detection
# =============================================================================

HEADER_FOOTER_ZONE = 0.07


# =============================================================================
# AI Classification
# =============================================================================

AI_CLASSIFICATION_MODEL = "gpt-4o-mini"
AI_MAX_INPUT_CHARS = 6000
AI_TEMPERATURE = 0.1


# =============================================================================
# PostgreSQL
# =============================================================================

PG_HOST = _require_env("PG_HOST")
PG_DB = _require_env("PG_DB")
PG_USER = _require_env("PG_USER")
PG_PASS = _require_env("PG_PASS")
PG_PORT = os.getenv("PG_PORT", "5432")


# =============================================================================
# Active Categories
# =============================================================================

ACTIVE_CATEGORIES = [
    "Contracts",
    "Compliance",
    "Governance",
    "Certifications",
    "Invoices",
]


# =============================================================================
# Rule Engine
# =============================================================================

RULES_FILE = "rules/ruleBasedConditions.json"

AUTO_THRESHOLD = 70.0

KEYWORD_WEIGHT = 2
SENTENCE_WEIGHT = 5
FILENAME_WEIGHT = 10
SCORE_BASELINE = 50


# =============================================================================
# Combined Scoring
# =============================================================================

RULE_WEIGHT = 0.35
AI_WEIGHT = 0.65

DISAGREE_PENALTY = 0.75
SOFT_FLAG_THRESHOLD = 40.0


# =============================================================================
# Logging
# =============================================================================

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_LEVEL = "INFO"


# =============================================================================
# Vector Search
# =============================================================================

SIMILARITY_THRESHOLD = 0.50