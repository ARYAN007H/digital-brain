"""
AI query routing engine.

Routes queries to the appropriate AI based on mode tags:
- RECALL / CONNECT / DO → qwen2.5:3b local (offline, instant)
- DECIDE / PREDICT / CREATE → Groq → Gemini → local fallback

Includes RAM safety: file-based lock prevents concurrent Ollama use.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from brain.config import API, Brain, Hardware, Paths

logger = logging.getLogger(__name__)


class BrainLock:
    """File-based lock to prevent concurrent Ollama usage.

    Only one process can use Ollama at a time (RAM constraint).
    """

    def __init__(self):
        self._lock_path = Paths.BRAIN_LOCK
        self._lock_file = None

    def acquire(self) -> bool:
        """Try to acquire the brain lock. Returns True if acquired."""
        try:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock_file = open(self._lock_path, "w")
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError):
            logger.warning("Brain lock is held by another process")
            return False

    def release(self):
        """Release the brain lock."""
        if self._lock_file:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
            except Exception:
                pass
            self._lock_file = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("Cannot acquire brain lock — Ollama in use by another process")
        return self

    def __exit__(self, *args):
        self.release()


class ConversationHistory:
    """SQLite-backed rolling conversation history.

    Stores last N exchanges per session. Auto-expires after idle timeout.
    """

    def __init__(self, db_path: Optional[Path] = None, max_exchanges: int = 3,
                 idle_timeout_min: int = 30):
        self._db_path = str(db_path or Paths.BRAIN_DB)
        self._max = max_exchanges
        self._idle_timeout = idle_timeout_min
        self._local = threading.local()
        self._session_id: Optional[str] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content     TEXT NOT NULL,
                mode        TEXT,
                provider    TEXT,
                created_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversation_history(session_id, created_at)
        """)
        conn.commit()

    def _get_or_create_session(self) -> str:
        """Get current session or create new one if idle too long."""
        conn = self._get_conn()
        if self._session_id:
            row = conn.execute(
                """SELECT created_at FROM conversation_history
                   WHERE session_id = ? ORDER BY created_at DESC LIMIT 1""",
                (self._session_id,),
            ).fetchone()
            if row:
                last = datetime.fromisoformat(row["created_at"])
                elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 60
                if elapsed < self._idle_timeout:
                    return self._session_id

        # New session
        self._session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return self._session_id

    def add_exchange(self, query: str, response: str, mode: str = "", provider: str = ""):
        """Record a query-response exchange."""
        session = self._get_or_create_session()
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO conversation_history (session_id, role, content, mode, provider, created_at)
               VALUES (?, 'user', ?, ?, ?, ?)""",
            (session, query, mode, provider, now),
        )
        conn.execute(
            """INSERT INTO conversation_history (session_id, role, content, mode, provider, created_at)
               VALUES (?, 'assistant', ?, ?, ?, ?)""",
            (session, response[:1000], mode, provider, now),
        )
        conn.commit()

    def get_context(self) -> str:
        """Get recent conversation context as a formatted string."""
        session = self._get_or_create_session()
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT role, content FROM conversation_history
               WHERE session_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (session, self._max * 2),
        ).fetchall()

        if not rows:
            return ""

        exchanges = []
        for row in reversed(rows):
            prefix = "User" if row["role"] == "user" else "Brain"
            exchanges.append(f"{prefix}: {row['content'][:300]}")

        return "Previous conversation:\n" + "\n".join(exchanges)


class Router:
    """AI query routing engine."""

    # Mode detection patterns
    MODE_PATTERNS = {
        "RECALL": re.compile(r"^RECALL[:\s]", re.IGNORECASE),
        "CONNECT": re.compile(r"^CONNECT[:\s]", re.IGNORECASE),
        "DO": re.compile(r"^DO[:\s]", re.IGNORECASE),
        "DECIDE": re.compile(r"^DECIDE[:\s]", re.IGNORECASE),
        "PREDICT": re.compile(r"^PREDICT[:\s]", re.IGNORECASE),
        "CREATE": re.compile(r"^CREATE[:\s]", re.IGNORECASE),
    }

    def __init__(self):
        self._lock = BrainLock()
        self._groq_client = None
        self._gemini_model = None
        self._history = ConversationHistory()

    # ── Mode Detection ───────────────────────────────────

    def detect_mode(self, query: str) -> str:
        """Detect query mode from prefix tags or infer from content."""
        query_stripped = query.strip()

        # Check explicit tags
        for mode, pattern in self.MODE_PATTERNS.items():
            if pattern.match(query_stripped):
                return mode

        # Infer from content keywords
        q_lower = query_stripped.lower()
        if any(w in q_lower for w in ["what do i know", "remember", "recall", "what is"]):
            return "RECALL"
        if any(w in q_lower for w in ["what links", "connected to", "related to"]):
            return "CONNECT"
        if any(w in q_lower for w in ["what should i", "action", "todo", "task"]):
            return "DO"
        if any(w in q_lower for w in ["decide", "should i", "help me choose", "compare"]):
            return "DECIDE"
        if any(w in q_lower for w in ["predict", "pattern", "trend", "forecast", "missing"]):
            return "PREDICT"
        if any(w in q_lower for w in ["brainstorm", "create", "generate", "imagine", "idea"]):
            return "CREATE"

        # Default to RECALL (handled locally)
        return "RECALL"

    def strip_mode_tag(self, query: str) -> str:
        """Remove the mode tag prefix from a query."""
        for pattern in self.MODE_PATTERNS.values():
            query = pattern.sub("", query.strip()).strip()
        return query

    # ── Local LLM (Ollama) ───────────────────────────────

    def ask_local(self, prompt: str, system_prompt: str = "") -> str:
        """Query qwen2.5:3b via Ollama. Unloads model after use.

        Acquires brain lock to prevent concurrent usage.
        """
        with self._lock:
            try:
                payload = {
                    "model": Hardware.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                }
                if system_prompt:
                    payload["system"] = system_prompt

                r = requests.post(
                    f"{API.OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=120,
                )
                r.raise_for_status()
                response = r.json().get("response", "")

            except Exception as e:
                logger.error(f"Ollama query failed: {e}")
                return f"[Local LLM error: {e}]"
            finally:
                self._unload_ollama()

        return response

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector via Ollama. Unloads model after."""
        with self._lock:
            try:
                r = requests.post(
                    f"{API.OLLAMA_HOST}/api/embeddings",
                    json={"model": Hardware.OLLAMA_MODEL, "prompt": text},
                    timeout=60,
                )
                r.raise_for_status()
                return r.json().get("embedding", [])
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return []
            finally:
                self._unload_ollama()

    def _unload_ollama(self):
        """Explicitly unload Ollama model to free RAM."""
        try:
            requests.post(
                f"{API.OLLAMA_HOST}/api/generate",
                json={
                    "model": Hardware.OLLAMA_MODEL,
                    "keep_alive": 0,
                },
                timeout=10,
            )
            logger.debug("Ollama model unloaded")
        except Exception:
            pass  # non-fatal

    # ── Cloud LLMs ───────────────────────────────────────

    def ask_groq(self, prompt: str, system_prompt: str = "") -> str:
        """Query Groq API (Llama 3.1 70B). Free tier."""
        if not API.has_groq():
            raise RuntimeError("Groq API key not configured")

        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=API.GROQ_API_KEY)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = self._groq_client.chat.completions.create(
            model=Hardware.GROQ_MODEL,
            messages=messages,
        )
        return res.choices[0].message.content

    def ask_gemini(self, prompt: str) -> str:
        """Query Gemini 2.0 Flash. Free tier fallback."""
        if not API.has_gemini():
            raise RuntimeError("Gemini API key not configured")

        if self._gemini_model is None:
            import google.generativeai as genai
            genai.configure(api_key=API.GEMINI_API_KEY)
            self._gemini_model = genai.GenerativeModel(Hardware.GEMINI_MODEL)

        return self._gemini_model.generate_content(prompt).text

    # ── Main Router ──────────────────────────────────────

    def route(
        self,
        query: str,
        context: str = "",
        system_prompt: str = "",
    ) -> dict:
        """Route a query to the appropriate AI.

        Injects conversation history and semantic context.
        Returns {mode, response, provider, query}.
        """
        mode = self.detect_mode(query)
        clean_query = self.strip_mode_tag(query)

        # Build full prompt with conversation history + semantic context
        parts = []
        history = self._history.get_context()
        if history:
            parts.append(history)
        if context:
            parts.append(f"Context from memory:\n{context}")
        parts.append(f"Query: {clean_query}")
        full_prompt = "\n\n".join(parts)

        provider = "unknown"
        response = ""

        if mode in Brain.LOCAL_MODES:
            provider = "ollama"
            response = self.ask_local(full_prompt, system_prompt)
        elif mode in Brain.CLOUD_MODES:
            try:
                provider = "groq"
                response = self.ask_groq(full_prompt, system_prompt)
            except Exception as e:
                logger.warning(f"Groq failed ({e}), trying Gemini")
                try:
                    provider = "gemini"
                    response = self.ask_gemini(full_prompt)
                except Exception as e2:
                    logger.warning(f"Gemini failed ({e2}), falling back to local")
                    provider = "ollama-fallback"
                    response = self.ask_local(full_prompt, system_prompt)

        # Record exchange in history
        self._history.add_exchange(clean_query, response, mode=mode, provider=provider)

        return {
            "mode": mode,
            "response": response,
            "provider": provider,
            "query": clean_query,
        }
