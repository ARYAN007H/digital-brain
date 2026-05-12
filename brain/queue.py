"""
SQLite async write queue.

Every external write (Supabase, PocketBase, ChromaDB) goes through this
queue first. The UI/ingestion pipeline never blocks on network I/O.

A background worker drains the queue periodically.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.config import Paths


class WriteQueue:
    """SQLite-backed async write queue. Thread-safe."""

    # Valid targets for queued writes
    TARGETS = {"supabase", "pocketbase", "chromadb"}

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = str(db_path or Paths.QUEUE_DB)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        """Create the queue table if it doesn't exist."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_writes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                target      TEXT NOT NULL CHECK(target IN ('supabase', 'pocketbase', 'chromadb')),
                operation   TEXT NOT NULL DEFAULT 'upsert',
                payload     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'done', 'failed', 'dead_letter')),
                attempts    INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 5,
                last_error  TEXT,
                last_attempt_at TEXT,
                completed_at TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_target_status
            ON pending_writes(target, status)
        """)
        # Migration: add columns if missing (for existing DBs)
        for col, typedef in [("max_retries", "INTEGER NOT NULL DEFAULT 5"),
                              ("last_attempt_at", "TEXT"),
                              ("status", None)]:
            if typedef:
                try:
                    conn.execute(f"ALTER TABLE pending_writes ADD COLUMN {col} {typedef}")
                except Exception:
                    pass  # column already exists
        # Migrate old 'failed' status check constraint
        try:
            conn.execute("""
                UPDATE pending_writes SET status = 'dead_letter'
                WHERE status = 'failed' AND attempts >= 5
            """)
        except Exception:
            pass
        conn.commit()

    def enqueue(self, target: str, payload: dict, operation: str = "upsert") -> int:
        """Add a write operation to the queue. Returns the queue item ID.

        Args:
            target: One of 'supabase', 'pocketbase', 'chromadb'.
            payload: Dict to be JSON-serialized.
            operation: Type of operation (upsert, delete, etc.).
        """
        if target not in self.TARGETS:
            raise ValueError(f"Invalid target '{target}'. Must be one of {self.TARGETS}")

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO pending_writes (target, operation, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (target, operation, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid

    def dequeue(self, target: str, batch_size: int = 10) -> list[dict]:
        """Fetch a batch of pending items for a target, marking them as 'processing'.

        Respects exponential backoff: skips items where not enough time
        has elapsed since last attempt (attempts * 60 seconds).

        Returns list of dicts with keys: id, target, operation, payload, created_at.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute(
            """
            SELECT id, target, operation, payload, created_at, attempts, last_attempt_at
            FROM pending_writes
            WHERE target = ? AND status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (target, batch_size * 2),  # fetch extra to account for backoff skips
        ).fetchall()

        items = []
        for row in rows:
            if len(items) >= batch_size:
                break

            # Exponential backoff check
            if row["attempts"] > 0 and row["last_attempt_at"]:
                try:
                    last = datetime.fromisoformat(row["last_attempt_at"])
                    backoff_seconds = min(row["attempts"] * 60, 3600)  # cap at 1 hour
                    next_allowed = last.timestamp() + backoff_seconds
                    if datetime.now(timezone.utc).timestamp() < next_allowed:
                        continue  # skip — not ready for retry yet
                except (ValueError, TypeError):
                    pass

            item = {
                "id": row["id"],
                "target": row["target"],
                "operation": row["operation"],
                "payload": json.loads(row["payload"]),
                "created_at": row["created_at"],
            }
            items.append(item)
            # Mark as processing
            conn.execute(
                """UPDATE pending_writes
                   SET status = 'processing', attempts = attempts + 1, last_attempt_at = ?
                   WHERE id = ?""",
                (now, row["id"]),
            )

        conn.commit()
        return items

    def mark_done(self, item_id: int):
        """Mark a queue item as successfully completed."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE pending_writes SET status = 'done', completed_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        conn.commit()

    def mark_failed(self, item_id: int, error: str):
        """Mark a queue item as failed.

        If attempts >= max_retries, moves to dead_letter status.
        Otherwise, sets back to pending for retry with backoff.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT attempts, max_retries FROM pending_writes WHERE id = ?",
            (item_id,),
        ).fetchone()

        if row and row["attempts"] >= (row["max_retries"] or 5):
            # Exceeded max retries → dead letter
            conn.execute(
                """UPDATE pending_writes
                   SET status = 'dead_letter', last_error = ?
                   WHERE id = ?""",
                (error, item_id),
            )
        else:
            # Back to pending for retry with backoff
            conn.execute(
                """UPDATE pending_writes
                   SET status = 'pending', last_error = ?
                   WHERE id = ?""",
                (error, item_id),
            )
        conn.commit()

    def pending_count(self, target: Optional[str] = None) -> int:
        """Count pending items, optionally filtered by target."""
        conn = self._get_conn()
        if target:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM pending_writes WHERE status = 'pending' AND target = ?",
                (target,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM pending_writes WHERE status = 'pending'",
            ).fetchone()
        return row["cnt"]

    def total_count(self) -> dict:
        """Get counts by status. Returns {pending, processing, done, failed, dead_letter}."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM pending_writes GROUP BY status",
        ).fetchall()
        counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0, "dead_letter": 0}
        for row in rows:
            counts[row["status"]] = row["cnt"]
        return counts

    def dead_letter_count(self) -> int:
        """Count items in dead letter queue."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM pending_writes WHERE status = 'dead_letter'",
        ).fetchone()
        return row["cnt"]

    def retry_dead_letters(self, target: Optional[str] = None) -> int:
        """Move dead letter items back to pending for retry."""
        conn = self._get_conn()
        if target:
            conn.execute(
                """UPDATE pending_writes
                   SET status = 'pending', attempts = 0
                   WHERE status = 'dead_letter' AND target = ?""",
                (target,),
            )
        else:
            conn.execute(
                "UPDATE pending_writes SET status = 'pending', attempts = 0 WHERE status = 'dead_letter'",
            )
        count = conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        return count

    def purge_completed(self, older_than_hours: int = 24):
        """Remove completed items older than the specified hours."""
        conn = self._get_conn()
        conn.execute(
            """
            DELETE FROM pending_writes
            WHERE status = 'done'
            AND completed_at < datetime('now', ? || ' hours')
            """,
            (f"-{older_than_hours}",),
        )
        conn.commit()

    def close(self):
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
