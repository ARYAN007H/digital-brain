"""
Associative recall engine — content-addressable memory retrieval.

Biological basis: partial cues trigger pattern completion in the
hippocampus.  Three retrieval pathways beyond pure vector search:
    1. Tag co-occurrence index
    2. Temporal-contextual recall
    3. SQLite FTS5 fuzzy search
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from brain.config import Paths
from brain.vault import Vault

logger = logging.getLogger(__name__)


class AssociativeRecallEngine:
    """Multi-modal associative retrieval for the digital brain."""

    def __init__(self, vault: Optional[Vault] = None, db_path: Optional[Path] = None):
        self._vault = vault or Vault()
        self._db_path = str(db_path or Paths.BRAIN_DB)
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
            CREATE TABLE IF NOT EXISTS tag_index (
                tag TEXT NOT NULL, neuron_id TEXT NOT NULL, region TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (tag, neuron_id))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_lookup ON tag_index(tag)")
        try:
            conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS neuron_fts
                USING fts5(neuron_id, title, body, region, tags, tokenize='trigram')""")
        except Exception:
            conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS neuron_fts
                USING fts5(neuron_id, title, body, region, tags, tokenize='unicode61')""")
        conn.commit()

    def rebuild_index(self) -> dict:
        """Rebuild tag + FTS indices from vault."""
        conn = self._get_conn()
        conn.execute("DELETE FROM tag_index")
        conn.execute("DELETE FROM neuron_fts")
        tag_count = fts_count = 0
        for fp in self._vault.list_neurons():
            try:
                n = self._vault.read_neuron(fp)
            except Exception:
                continue
            tags = getattr(n, "tags", []) or []
            for tag in tags:
                t = tag.strip().lower()
                if t:
                    conn.execute("INSERT OR IGNORE INTO tag_index (tag,neuron_id,region) VALUES (?,?,?)", (t, n.id, n.region))
                    tag_count += 1
            conn.execute("INSERT INTO neuron_fts (neuron_id,title,body,region,tags) VALUES (?,?,?,?,?)",
                         (n.id, n.title, (n.body or "")[:2000], n.region, " ".join(tags)))
            fts_count += 1
        conn.commit()
        logger.info("Associative index rebuilt: tags=%d fts=%d", tag_count, fts_count)
        return {"tags_indexed": tag_count, "neurons_indexed": fts_count}

    def index_neuron(self, neuron_id: str, title: str, body: str, region: str, tags: list[str]):
        """Index a single neuron (called during ingestion)."""
        conn = self._get_conn()
        for tag in tags:
            t = tag.strip().lower()
            if t:
                conn.execute("INSERT OR IGNORE INTO tag_index (tag,neuron_id,region) VALUES (?,?,?)", (t, neuron_id, region))
        conn.execute("INSERT OR REPLACE INTO neuron_fts (neuron_id,title,body,region,tags) VALUES (?,?,?,?,?)",
                     (neuron_id, title, body[:2000], region, " ".join(tags)))
        conn.commit()

    def recall_by_tags(self, query_tags: list[str], limit: int = 20) -> list[dict]:
        if not query_tags:
            return []
        conn = self._get_conn()
        clean = [t.strip().lower() for t in query_tags if t.strip()]
        ph = ",".join("?" * len(clean))
        rows = conn.execute(
            f"SELECT neuron_id,region,GROUP_CONCAT(tag) as mt,COUNT(*) as score FROM tag_index WHERE tag IN ({ph}) GROUP BY neuron_id ORDER BY score DESC LIMIT ?",
            (*clean, limit)).fetchall()
        return [{"neuron_id": r["neuron_id"], "region": r["region"], "matched_tags": r["mt"].split(","), "score": r["score"], "source": "tag_recall"} for r in rows]

    def recall_by_time(self, target_date: str, limit: int = 10) -> list[dict]:
        results = []
        for fp in self._vault.list_neurons():
            try:
                n = self._vault.read_neuron(fp)
                if n.created and n.created.startswith(target_date):
                    results.append({"neuron_id": n.id, "title": n.title, "region": n.region, "created": n.created, "source": "temporal_recall"})
            except Exception:
                continue
            if len(results) >= limit:
                break
        return results

    def recall_fuzzy(self, query: str, limit: int = 10) -> list[dict]:
        if not query.strip():
            return []
        conn = self._get_conn()
        safe = query.replace('"', '""')
        try:
            rows = conn.execute(
                'SELECT neuron_id,title,region,snippet(neuron_fts,2,">>>","<<<","...",40) as snip,rank FROM neuron_fts WHERE neuron_fts MATCH ? ORDER BY rank LIMIT ?',
                (f'"{safe}"', limit)).fetchall()
            return [{"neuron_id": r["neuron_id"], "title": r["title"], "region": r["region"], "snippet": r["snip"], "source": "fuzzy_recall"} for r in rows]
        except Exception:
            return []

    def recall(self, query: str, tags: list[str] | None = None, limit: int = 10) -> list[dict]:
        """Multi-modal associative recall: tags + fuzzy + temporal."""
        all_r: dict[str, dict] = {}
        if tags:
            for r in self.recall_by_tags(tags, limit):
                all_r[r["neuron_id"]] = r
                all_r[r["neuron_id"]]["combined_score"] = r.get("score", 1) * 2
        for r in self.recall_fuzzy(query, limit):
            nid = r["neuron_id"]
            if nid in all_r:
                all_r[nid]["combined_score"] = all_r[nid].get("combined_score", 0) + 3
            else:
                all_r[nid] = r
                all_r[nid]["combined_score"] = 3
        dm = re.search(r"(\d{4}-\d{2}-\d{2})", query)
        if dm:
            for r in self.recall_by_time(dm.group(1), 5):
                nid = r["neuron_id"]
                if nid in all_r:
                    all_r[nid]["combined_score"] = all_r[nid].get("combined_score", 0) + 2
                else:
                    all_r[nid] = r
                    all_r[nid]["combined_score"] = 2
        return sorted(all_r.values(), key=lambda x: x.get("combined_score", 0), reverse=True)[:limit]

    def index_stats(self) -> dict:
        conn = self._get_conn()
        tc = conn.execute("SELECT COUNT(DISTINCT tag) FROM tag_index").fetchone()[0]
        nc = conn.execute("SELECT COUNT(DISTINCT neuron_id) FROM tag_index").fetchone()[0]
        try:
            fc = conn.execute("SELECT COUNT(*) FROM neuron_fts").fetchone()[0]
        except Exception:
            fc = 0
        return {"unique_tags": tc, "tagged_neurons": nc, "fts_indexed": fc}
