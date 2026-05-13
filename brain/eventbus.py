"""
Neural event bus — the signaling backbone of the digital brain.

Biological analogue: action potentials propagating through the neural
network.  Every subsystem (plasticity, working memory, decay, etc.)
can emit and subscribe to typed events *without* coupling to each other.

Design:
    - In-process pub/sub via dict[event_type → list[callable]].
    - Thread-safe (stdlib threading.Lock).
    - Events persisted to brain.db for replay / audit.
    - Zero external dependencies.

Event types:
    neuron.created      — new neuron written to vault
    neuron.accessed     — neuron read during query / browse
    synapse.reinforced  — synapse strength changed
    query.completed     — full query→response cycle done
    ingestion.completed — file ingestion pipeline finished
    consolidation.done  — nightly consolidation pass finished
    decay.applied       — nightly decay pass finished
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from brain.config import Paths

logger = logging.getLogger(__name__)

# ── Type alias for subscribers ───────────────────────────
Subscriber = Callable[[str, dict[str, Any]], None]


class EventBus:
    """Lightweight in-process event bus with SQLite persistence.

    Usage:
        bus = EventBus.get()                      # singleton
        bus.subscribe("neuron.accessed", my_fn)
        bus.emit("neuron.accessed", {"neuron_id": "NRN-20260513-0001"})
    """

    _instance: Optional[EventBus] = None
    _init_lock = threading.Lock()

    # ── Singleton ────────────────────────────────────────

    @classmethod
    def get(cls) -> EventBus:
        """Return the process-wide singleton event bus."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Init ─────────────────────────────────────────────

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = str(db_path or Paths.BRAIN_DB)
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local connection (matches project convention)."""
        local = threading.local()
        if not hasattr(local, "_eb_conn") or local._eb_conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=3000")
            local._eb_conn = conn
        return local._eb_conn

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT    NOT NULL,
                payload     TEXT    NOT NULL DEFAULT '{}',
                created_at  TEXT    NOT NULL,
                processed   INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type_time
            ON event_log(event_type, created_at)
        """)
        conn.commit()
        conn.close()

    # ── Subscribe / Unsubscribe ──────────────────────────

    def subscribe(self, event_type: str, callback: Subscriber):
        """Register a callback for an event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)
        logger.debug("Subscribed %s to '%s'", callback.__qualname__, event_type)

    def unsubscribe(self, event_type: str, callback: Subscriber):
        """Remove a callback for an event type."""
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            self._subscribers[event_type] = [s for s in subs if s is not callback]

    # ── Emit ─────────────────────────────────────────────

    def emit(self, event_type: str, payload: dict[str, Any] | None = None):
        """Emit an event: persist to DB, then notify all subscribers.

        Subscriber exceptions are logged but never propagate — a misbehaving
        subscriber must not crash the emitter (same as biological robustness:
        one dead synapse doesn't kill the neuron).
        """
        payload = payload or {}
        now = datetime.now(timezone.utc).isoformat()

        # Persist
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "INSERT INTO event_log (event_type, payload, created_at) VALUES (?, ?, ?)",
                (event_type, json.dumps(payload, default=str), now),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Event persistence failed for '%s': %s", event_type, e)

        # Dispatch
        with self._lock:
            subs = list(self._subscribers.get(event_type, []))

        for callback in subs:
            try:
                callback(event_type, payload)
            except Exception as e:
                logger.warning(
                    "Subscriber %s failed on '%s': %s",
                    callback.__qualname__, event_type, e,
                )

    # ── Query log ────────────────────────────────────────

    def recent_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent events from the persistent log."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        if event_type:
            rows = conn.execute(
                "SELECT * FROM event_log WHERE event_type = ? ORDER BY id DESC LIMIT ?",
                (event_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM event_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        conn.close()
        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def event_count(self, event_type: Optional[str] = None) -> int:
        """Total number of events in the log."""
        conn = sqlite3.connect(self._db_path)
        if event_type:
            row = conn.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = ?",
                (event_type,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM event_log").fetchone()
        conn.close()
        return row[0] if row else 0

    def purge_old_events(self, keep_days: int = 30):
        """Remove events older than keep_days."""
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "DELETE FROM event_log WHERE datetime(created_at) < datetime('now', ?)",
            (f"-{keep_days} days",),
        )
        conn.commit()
        conn.close()
