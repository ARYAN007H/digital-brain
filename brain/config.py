"""
Central configuration for the Digital Brain.

All paths, environment variables, constants, and hardware constraints
are defined here. Every other module imports from this file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


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

    # Supabase (optional cloud mirror)
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

    # Local services
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    POCKETBASE_URL = os.getenv("POCKETBASE_URL", "http://localhost:8090")

    @classmethod
    def has_groq(cls) -> bool:
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here")

    @classmethod
    def has_gemini(cls) -> bool:
        return bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here")

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

    # Surfacing schedule
    NIGHTLY_HOUR = 2  # 2am


# ── Initialize directories on import ─────────────────────
Paths.ensure_dirs()
