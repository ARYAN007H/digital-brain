"""
Smart deduplication engine.

SHA-256 content fingerprinting stored in brain.db SQLite.
Prevents duplicate neurons when the same file is ingested twice.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.config import Paths

logger = logging.getLogger(__name__)


class DedupEngine:
    """Content-hash based deduplication using SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = str(db_path or Paths.BRAIN_DB)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
        return self._local.conn

    def _init_db(self):
        """Create content_hashes table."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                hash        TEXT PRIMARY KEY,
                filepath    TEXT NOT NULL,
                neuron_ids  TEXT NOT NULL DEFAULT '[]',
                first_seen  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                ingest_count INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.commit()

    @staticmethod
    def hash_content(content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_file(filepath: Path) -> str:
        """Generate SHA-256 hash of file content."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def is_duplicate(self, filepath: Path) -> bool:
        """Check if file content has already been ingested."""
        content_hash = self.hash_file(filepath)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT hash FROM content_hashes WHERE hash = ?",
            (content_hash,),
        ).fetchone()
        return row is not None

    def check_and_record(self, filepath: Path, force: bool = False) -> tuple[bool, str]:
        """Check for duplicate and record if new.

        Returns (is_new, content_hash).
        If force=True, allows re-ingestion but updates the record.
        """
        content_hash = self.hash_file(filepath)
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        existing = conn.execute(
            "SELECT * FROM content_hashes WHERE hash = ?",
            (content_hash,),
        ).fetchone()

        if existing:
            # Update last_seen and increment count
            conn.execute(
                """UPDATE content_hashes
                   SET last_seen = ?, ingest_count = ingest_count + 1
                   WHERE hash = ?""",
                (now, content_hash),
            )
            conn.commit()

            if not force:
                logger.info(
                    f"Duplicate detected: {filepath.name} "
                    f"(first seen: {existing['first_seen']}, "
                    f"ingested {existing['ingest_count']} times)"
                )
                return False, content_hash

            logger.info(f"Force re-ingesting duplicate: {filepath.name}")

        else:
            # New content — record it
            conn.execute(
                """INSERT INTO content_hashes (hash, filepath, first_seen, last_seen)
                   VALUES (?, ?, ?, ?)""",
                (content_hash, str(filepath), now, now),
            )
            conn.commit()

        return True, content_hash

    def record_neurons(self, content_hash: str, neuron_ids: list[str]):
        """Associate neuron IDs with a content hash."""
        import json
        conn = self._get_conn()
        conn.execute(
            "UPDATE content_hashes SET neuron_ids = ? WHERE hash = ?",
            (json.dumps(neuron_ids), content_hash),
        )
        conn.commit()

    def get_stats(self) -> dict:
        """Get dedup statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) as cnt FROM content_hashes").fetchone()["cnt"]
        dupes = conn.execute(
            "SELECT COUNT(*) as cnt FROM content_hashes WHERE ingest_count > 1"
        ).fetchone()["cnt"]
        return {"total_unique": total, "duplicates_caught": dupes}

    def close(self):
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
