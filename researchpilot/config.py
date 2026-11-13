"""Central configuration. All values come from environment variables / .env."""
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Tried in order if the primary model stays unavailable (comma-separated). Empty = no fallback.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite").split(",") if m.strip()
]

CROSSREF_URL = "https://api.crossref.org/works"
# Crossref asks polite clients to identify themselves. Optional but recommended.
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---- LLM provider (default: Groq free tier). Gemini remains available with LLM_PROVIDER=gemini ----
SUPPORTED_PROVIDERS = ("groq", "gemini", "openrouter", "ollama", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Presets for OpenAI-compatible chat APIs. Every value can be overridden with LLM_BASE_URL / LLM_API_KEY /
# LLM_MODEL / LLM_FALLBACK_MODELS. Free-tier model names change; verify yours with scripts/check_llm.py.
_PRESETS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model": "llama-3.3-70b-versatile", "fallbacks": "llama-3.1-8b-instant"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY",
                   "model": "meta-llama/llama-3.3-70b-instruct:free", "fallbacks": ""},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": "",
               "model": "llama3.1:8b", "fallbacks": ""},
    "openai": {"base_url": "", "key_env": "OPENAI_API_KEY", "model": "", "fallbacks": ""},
}


def resolve_llm_settings(provider: str, env) -> dict:
    """Settings for an OpenAI-compatible provider from an environment mapping (pure, unit-tested)."""
    preset = _PRESETS.get(provider, {})
    key_env = preset.get("key_env", "")
    fallbacks = env.get("LLM_FALLBACK_MODELS", preset.get("fallbacks", "")) or ""
    return {
        "base_url": (env.get("LLM_BASE_URL") or preset.get("base_url", "")).strip(),
        "api_key": (env.get("LLM_API_KEY") or (env.get(key_env) if key_env else "") or "").strip(),
        "model": (env.get("LLM_MODEL") or preset.get("model", "")).strip(),
        "fallbacks": [m.strip() for m in fallbacks.split(",") if m.strip()],
    }


_llm = resolve_llm_settings(LLM_PROVIDER, os.environ)
LLM_BASE_URL, LLM_API_KEY = _llm["base_url"], _llm["api_key"]
LLM_MODEL, LLM_FALLBACK_MODELS = _llm["model"], _llm["fallbacks"]
LLM_TIMEOUT_SECONDS = _int_env("LLM_TIMEOUT_SECONDS", 90)


# Retry / limit settings (bounded on purpose: guidelines require step/budget limits)
LLM_MAX_ATTEMPTS = _int_env("LLM_MAX_ATTEMPTS", 4)
HTTP_MAX_ATTEMPTS = _int_env("HTTP_MAX_ATTEMPTS", 3)
HTTP_TIMEOUT_SECONDS = _int_env("HTTP_TIMEOUT_SECONDS", 15)
MAX_PAPERS = _int_env("MAX_PAPERS", 50)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")

# ---- Stage 2: discovery, retrieval, memory (all free / open-source) ----
# OpenAlex now requires a FREE key for meaningful use. Optional: without it the
# tiny keyless allowance is used and the system falls back to other sources.
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
OPENALEX_URL = "https://api.openalex.org/works"
# Hard in-process ceiling on OpenAlex calls so we can never run up usage.
OPENALEX_MAX_CALLS = _int_env("OPENALEX_MAX_CALLS", 100)
ARXIV_URL = "https://export.arxiv.org/api/query"

# Vector store backend: "auto" (Chroma if it works, else built-in BM25), "chroma", "lexical"
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "auto").lower()
CHROMA_DIR = os.getenv("CHROMA_DIR", "")  # empty = in-memory per session
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3")
SEARCH_CACHE_DAYS = _int_env("SEARCH_CACHE_DAYS", 7)

# ---- Stage 3: analysis agents and orchestration ----
DISCOVERY_LIMIT = _int_env("DISCOVERY_LIMIT", 10)        # papers requested per discovery
ANALYSIS_MAX_PAPERS = _int_env("ANALYSIS_MAX_PAPERS", 8)  # max papers sent to the LLM per run
ANALYSIS_BATCH_SIZE = _int_env("ANALYSIS_BATCH_SIZE", 4)  # papers per LLM call
MIN_ABSTRACT_PAPERS = _int_env("MIN_ABSTRACT_PAPERS", 3)  # below this, the orchestrator broadens the search once
MAX_STEPS = _int_env("MAX_STEPS", 12)                     # hard step budget per workflow run (topic run + questionnaire + evaluation)

# ---- Stage 4: questionnaire and evaluation ----
MAX_GAP_QUESTIONS = _int_env("MAX_GAP_QUESTIONS", 5)   # gap-specific follow-up questions
MAX_ANSWER_CHARS = _int_env("MAX_ANSWER_CHARS", 3000)  # longest accepted answer

# ---- resilience ----
HTTP_MAX_RETRY_WAIT = _int_env("HTTP_MAX_RETRY_WAIT", 10)  # never sleep longer than this on a 429; give up on that source instead

# ---- Stage 5: report generation ----
REPORT_USE_LLM = os.getenv("REPORT_USE_LLM", "1").strip().lower() not in ("0", "false", "no", "off")

# ---- Stage 6: uploaded-paper (Option 2) pipeline ----
MAX_PDF_MB = _int_env("MAX_PDF_MB", 15)                    # reject larger uploads
MAX_PDF_PAGES = _int_env("MAX_PDF_PAGES", 60)              # analyse at most this many pages (a warning is shown)
MAX_PDF_SECONDS = _int_env("MAX_PDF_SECONDS", 45)          # time budget for text + table extraction
MIN_CHARS_PER_PAGE = _int_env("MIN_CHARS_PER_PAGE", 150)   # below this on average the PDF is treated as scanned
MAX_TEXT_CHARS = _int_env("MAX_TEXT_CHARS", 300000)        # extracted text kept in memory
MAX_STORED_TEXT = _int_env("MAX_STORED_TEXT", 300000)      # text kept in the saved workflow state
PARSE_CHAR_BUDGET = _int_env("PARSE_CHAR_BUDGET", 9000)    # paper text sent to the LLM per extraction call (token-limit friendly)
RELATED_QUERIES = _int_env("RELATED_QUERIES", 3)           # search queries built from the uploaded paper
