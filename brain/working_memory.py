"""
Working memory buffer — prefrontal scratchpad.

Biological basis:
    The prefrontal cortex maintains a small working memory buffer
    (~7±2 items) that persists across related operations.  Active
    neurons stay "primed" so the brain doesn't re-retrieve them every
    few seconds.  Items are ranked by recency × frequency and the
    buffer flushes when the topic shifts.

This gives query-to-query continuity without re-running ChromaDB
on every call.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.config import Paths
from brain.eventbus import EventBus

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────
MAX_BUFFER_SIZE = 7            # Miller's 7±2
FOCUS_DECAY_SECONDS = 600     # topic focus decays after 10 min
MIN_RELEVANCE_SCORE = 0.1


class WorkingMemoryBuffer:
    """Holds the active "working set" of primed neurons.

    Updated on every query.  Persisted to brain.db so it survives
    process restarts.  Implements recency × frequency scoring.
    """

    def __init__(self, db_path: Optional[Path] = None, max_size: int = MAX_BUFFER_SIZE):
        self._db_path = str(db_path or Paths.BRAIN_DB)
        self._max_size = max_size
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=3000")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                neuron_id       TEXT PRIMARY KEY,
                content_preview TEXT NOT NULL DEFAULT '',
                region          TEXT NOT NULL DEFAULT '',
                frequency       INTEGER NOT NULL DEFAULT 1,
                last_accessed   REAL NOT NULL,
                first_accessed  REAL NOT NULL,
                relevance_score REAL NOT NULL DEFAULT 1.0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS working_memory_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()

    # ── Buffer operations ────────────────────────────────

    def prime(self, neuron_id: str, content_preview: str = "", region: str = ""):
        """Add or refresh a neuron in working memory.

        If already present, increments frequency and updates timestamp.
        If buffer is full, evicts the least relevant item.
        """
        now = time.time()
        conn = self._get_conn()

        existing = conn.execute(
            "SELECT * FROM working_memory WHERE neuron_id = ?",
            (neuron_id,),
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE working_memory
                   SET frequency = frequency + 1, last_accessed = ?, relevance_score = ?
                   WHERE neuron_id = ?""",
                (now, self._compute_relevance(existing["frequency"] + 1, now, existing["first_accessed"]), neuron_id),
            )
        else:
            # Evict if full
            self._evict_if_full(conn)
            conn.execute(
                """INSERT INTO working_memory
                   (neuron_id, content_preview, region, frequency, last_accessed, first_accessed, relevance_score)
                   VALUES (?, ?, ?, 1, ?, ?, 1.0)""",
                (neuron_id, content_preview[:500], region, now, now),
            )

        conn.commit()

    def prime_batch(self, items: list[dict]):
        """Prime multiple neurons at once.

        Each item: {neuron_id, content_preview, region}.
        """
        for item in items:
            self.prime(
                item.get("neuron_id", ""),
                item.get("content_preview", ""),
                item.get("region", ""),
            )

    def _evict_if_full(self, conn: sqlite3.Connection):
        """Evict lowest-relevance item if buffer is at capacity."""
        count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        if count >= self._max_size:
            # Evict the item with lowest relevance
            conn.execute("""
                DELETE FROM working_memory
                WHERE neuron_id = (
                    SELECT neuron_id FROM working_memory
                    ORDER BY relevance_score ASC LIMIT 1
                )
            """)

    @staticmethod
    def _compute_relevance(frequency: int, last_accessed: float, first_accessed: float) -> float:
        """Recency × frequency scoring.

        relevance = frequency × recency_factor
        recency_factor decays exponentially from the last access time.
        """
        age_seconds = max(1.0, time.time() - last_accessed)
        recency = 1.0 / (1.0 + age_seconds / FOCUS_DECAY_SECONDS)
        return frequency * recency

    # ── Retrieval ────────────────────────────────────────

    def get_active_set(self) -> list[dict]:
        """Return all items in working memory, sorted by relevance."""
        conn = self._get_conn()
        now = time.time()

        rows = conn.execute(
            "SELECT * FROM working_memory ORDER BY relevance_score DESC",
        ).fetchall()

        result = []
        for r in rows:
            # Recompute relevance on read (it's time-dependent)
            rel = self._compute_relevance(r["frequency"], r["last_accessed"], r["first_accessed"])
            if rel < MIN_RELEVANCE_SCORE:
                continue
            result.append({
                "neuron_id": r["neuron_id"],
                "content_preview": r["content_preview"],
                "region": r["region"],
                "frequency": r["frequency"],
                "relevance": round(rel, 4),
            })

        return result

    def get_primed_ids(self) -> list[str]:
        """Return just the neuron IDs in the active working set."""
        return [item["neuron_id"] for item in self.get_active_set()]

    def get_context_string(self) -> str:
        """Return working memory as a formatted context string for the LLM."""
        items = self.get_active_set()
        if not items:
            return ""

        lines = ["Active working memory (primed concepts):"]
        for item in items:
            lines.append(f"  [{item['neuron_id']}] {item['content_preview'][:200]}")
        return "\n".join(lines)

    # ── Focus tracking ───────────────────────────────────

    def set_focus_topic(self, topic: str):
        """Set the current focus topic."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO working_memory_meta (key, value) VALUES ('focus_topic', ?)",
            (json.dumps({"topic": topic, "set_at": time.time()}),),
        )
        conn.commit()

    def get_focus_topic(self) -> Optional[str]:
        """Get the current focus topic, or None if expired."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM working_memory_meta WHERE key = 'focus_topic'",
        ).fetchone()
        if not row:
            return None
        data = json.loads(row["value"])
        elapsed = time.time() - data.get("set_at", 0)
        if elapsed > FOCUS_DECAY_SECONDS * 3:  # 30 min total focus window
            return None
        return data.get("topic")

    # ── Flush ────────────────────────────────────────────

    def flush(self):
        """Clear the entire working memory buffer."""
        conn = self._get_conn()
        conn.execute("DELETE FROM working_memory")
        conn.execute("DELETE FROM working_memory_meta WHERE key = 'focus_topic'")
        conn.commit()
        logger.info("Working memory flushed")

    def cleanup_stale(self, max_age_seconds: float = 3600):
        """Remove items not accessed in the last hour."""
        cutoff = time.time() - max_age_seconds
        conn = self._get_conn()
        conn.execute("DELETE FROM working_memory WHERE last_accessed < ?", (cutoff,))
        conn.commit()

    # ── EventBus integration ─────────────────────────────

    def on_query_completed(self, event_type: str, payload: dict):
        """EventBus subscriber: update working memory after each query."""
        # Prime neurons that were used as context
        context_neurons = payload.get("context_neuron_ids", [])
        for nid in context_neurons:
            self.prime(nid)

        # Update focus topic
        query = payload.get("query", "")
        if query:
            self.set_focus_topic(query[:200])

    def register(self, bus: Optional[EventBus] = None):
        """Register on the event bus."""
        bus = bus or EventBus.get()
        bus.subscribe("query.completed", self.on_query_completed)
        logger.info("Working memory registered on event bus")

    # ── Stats ────────────────────────────────────────────

    def size(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
