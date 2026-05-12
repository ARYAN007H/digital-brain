"""
Central configuration for the Digital Brain.

All paths, environment variables, constants, and hardware constraints
are defined here. Every other module imports from this file.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")



def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Paths:
    """All filesystem paths used by the brain."""

    ROOT = _PROJECT_ROOT
    VAULT = Path(os.getenv("VAULT_ROOT", _PROJECT_ROOT / "vault"))
    DATA = Path(os.getenv("DATA_DIR", _PROJECT_ROOT / "data"))
    BIN = Path(os.getenv("BIN_DIR", _PROJECT_ROOT / "bin"))

    # Cortex regions
    PREFRONTAL = VAULT / "prefrontal"
    HIPPOCAMPUS = VAULT / "hippocampus"
    CREATIVE = VAULT / "creative"
    PREDICTIVE = VAULT / "predictive"
    AMYGDALA = VAULT / "amygdala"
    EXECUTIVE = VAULT / "executive"

    # Special directories
    RAW_LOGS = VAULT / "_raw-logs"
    INBOX = RAW_LOGS / "inbox"
    META = VAULT / "_meta"

    # Data stores
    CHROMADB = DATA / "chromadb"
    POCKETBASE = DATA / "pocketbase"
    QUEUE_DB = DATA / "queue.db"
    BRAIN_DB = DATA / "brain.db"

    # Lock file (prevents concurrent Ollama use)
    BRAIN_LOCK = DATA / ".brain_lock"

    # Piper TTS
    PIPER_BIN = BIN / "piper"
    PIPER_MODEL = BIN / "en_US-lessac-medium.onnx"

    # Templates
    TEMPLATES = ROOT / "templates"

    # All cortex region paths for iteration
    REGIONS = {
        "prefrontal": PREFRONTAL,
        "hippocampus": HIPPOCAMPUS,
        "creative": CREATIVE,
        "predictive": PREDICTIVE,
        "amygdala": AMYGDALA,
        "executive": EXECUTIVE,
    }

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories if they don't exist."""
        for path in [
            cls.VAULT, cls.DATA, cls.BIN,
            cls.PREFRONTAL, cls.HIPPOCAMPUS, cls.CREATIVE,
            cls.PREDICTIVE, cls.AMYGDALA, cls.EXECUTIVE,
            cls.RAW_LOGS, cls.INBOX, cls.META,
            cls.CHROMADB, cls.POCKETBASE,
            cls.TEMPLATES,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class API:
    """API keys and endpoints."""

    # Cloud AI (free tier, no card)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")

    # Supabase (optional cloud mirror)
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

    # Local services
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    POCKETBASE_URL = os.getenv("POCKETBASE_URL", "http://localhost:8090")
    ALLOW_REMOTE_ENDPOINTS = os.getenv("ALLOW_REMOTE_ENDPOINTS", "false").lower() in {
        "1", "true", "yes", "on"
    }
    OLLAMA_AUTH_TOKEN = os.getenv("OLLAMA_AUTH_TOKEN", "")
    POCKETBASE_AUTH_TOKEN = os.getenv("POCKETBASE_AUTH_TOKEN", "")

    @classmethod
    def has_groq(cls) -> bool:
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here")

    @classmethod
    def has_gemini(cls) -> bool:
        return bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here")

    @classmethod
    def has_xai(cls) -> bool:
        return bool(cls.XAI_API_KEY and cls.XAI_API_KEY != "your_xai_api_key_here")

    @classmethod
    def has_supabase(cls) -> bool:
        return bool(cls.SUPABASE_URL and cls.SUPABASE_URL != "https://your-project.supabase.co")


class Hardware:
    """Hardware constraints — every script must respect these."""

    # CPU only — no CUDA, no GPU flags, no ROCm
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"  # for faster-whisper

    # RAM budget (MB)
    TOTAL_RAM_MB = 8192
    USABLE_RAM_MB = 5120  # ~5GB after OS + iGPU
    OLLAMA_RAM_MB = 2800  # qwen2.5:3b footprint
    WHISPER_RAM_MB = 300  # faster-whisper base
    CHROMADB_RAM_MB = 150
    POCKETBASE_RAM_MB = 50

    # Ollama config
    OLLAMA_MODEL = "qwen2.5:3b"
    OLLAMA_KEEP_ALIVE = "0"  # always unload after use

    # Groq model
    GROQ_MODEL = "llama-3.1-70b-versatile"

    # Gemini model
    GEMINI_MODEL = "gemini-2.0-flash"

    # xAI model
    XAI_MODEL = "grok-beta"


class Brain:
    """Brain-level constants."""

    # Synapse strength scoring
    SYNAPSE_CO_ACCESS = 1       # two notes opened within 10 min
    SYNAPSE_WIKILINK = 2        # one note wikilinks another
    SYNAPSE_SEMANTIC_SIM = 3    # local LLM: similarity > 0.85
    SYNAPSE_AI_CONFIRMED = 3    # Groq/Gemini confirms relationship
    SYNAPSE_USER_CONFIRMED = 5  # user manually confirms

    # Semantic similarity threshold
    SIMILARITY_THRESHOLD = 0.85

    # ChromaDB collection names (one per region, amygdala is metadata-only)
    CHROMA_COLLECTIONS = {
        "prefrontal": "prefrontal_neurons",
        "hippocampus": "hippocampus_neurons",
        "creative": "creative_neurons",
        "predictive": "predictive_neurons",
        "executive": "executive_neurons",
    }

    # Query modes
    LOCAL_MODES = {"RECALL", "CONNECT", "DO"}
    CLOUD_MODES = {"DECIDE", "PREDICT", "CREATE"}

    # Sync interval (seconds)
    SYNC_INTERVAL = 300  # 5 minutes

    # Conversation history privacy controls
    HISTORY_ENABLED = _env_bool("HISTORY_ENABLED", True)
    HISTORY_RETENTION_DAYS = max(1, _env_int("HISTORY_RETENTION_DAYS", 30))
    HISTORY_MAX_CONTENT_CHARS = max(100, _env_int("HISTORY_MAX_CONTENT_CHARS", 1000))
    # plain | hash | redact
    HISTORY_PRIVACY_MODE = os.getenv("HISTORY_PRIVACY_MODE", "plain").strip().lower()

    @classmethod
    def history_privacy_mode(cls) -> str:
        mode = cls.HISTORY_PRIVACY_MODE
        return mode if mode in {"plain", "hash", "redact"} else "plain"

    # Surfacing schedule
    NIGHTLY_HOUR = 2  # 2am


class HTTP:
    """Centralized hardened HTTP helpers for local service calls."""

    DEFAULT_TIMEOUT = float(os.getenv("LOCAL_HTTP_TIMEOUT_SEC", "10"))
    _LOCAL_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}

    @classmethod
    def _mask_error(cls, error: Exception) -> str:
        text = str(error)
        for secret in (API.OLLAMA_AUTH_TOKEN, API.POCKETBASE_AUTH_TOKEN):
            if secret:
                text = text.replace(secret, "***REDACTED***")
        return text

    @classmethod
    def validate_local_endpoint(cls, endpoint: str, service_name: str):
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"{service_name} endpoint must use http/https")
        host = (parsed.hostname or "").lower()
        if not host:
            raise ValueError(f"{service_name} endpoint missing host")

        if not API.ALLOW_REMOTE_ENDPOINTS and host not in cls._LOCAL_ALLOWED_HOSTS:
            raise ValueError(
                f"{service_name} endpoint host '{host}' is blocked. "
                "Set ALLOW_REMOTE_ENDPOINTS=true only when explicitly needed."
            )
        return host, parsed.port

    @classmethod
    def sanitized_origin(cls, endpoint: str, service_name: str) -> str:
        host, port = cls.validate_local_endpoint(endpoint, service_name)
        return f"{host}:{port}" if port else host

    @classmethod
    def build_headers(cls, auth_token: str = "", extra_headers: dict | None = None) -> dict:
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    @classmethod
    def request(
        cls,
        method: str,
        url: str,
        *,
        service_name: str,
        timeout: float | None = None,
        headers: dict | None = None,
        **kwargs,
    ) -> requests.Response:
        cls.validate_local_endpoint(url, service_name)
        timeout = cls.DEFAULT_TIMEOUT if timeout is None else timeout
        try:
            response = requests.request(method, url, timeout=timeout, headers=headers, **kwargs)
            return response
        except Exception as error:
            raise RuntimeError(f"{service_name} request failed: {cls._mask_error(error)}") from error


# ── Initialize directories on import ─────────────────────
Paths.ensure_dirs()
