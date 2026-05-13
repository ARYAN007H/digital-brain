"""
Synaptic decay engine — long-term depression and synaptic pruning.

Biological basis:
    Every synapse has a half-life.  Connections that aren't used undergo
    long-term depression (LTD) and eventually get pruned (synaptic
    elimination).  Without forgetting, the brain drowns in noise.
    Emotional memories (amygdala-tagged) decay slower — a real
    phenomenon called emotional memory persistence.

Runs as part of the nightly consolidation job.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from brain.config import API, Brain, HTTP, Paths
from brain.eventbus import EventBus

logger = logging.getLogger(__name__)

# ── Default decay parameters ─────────────────────────────
# λ = ln(2) / half_life_days → default half-life ~35 days
DEFAULT_LAMBDA = 0.02
AMYGDALA_LAMBDA = 0.01          # emotional memories decay 2x slower
MIN_STRENGTH_THRESHOLD = 1      # prune below this
MAX_BATCH_SIZE = 500            # PocketBase page size


class DecayEngine:
    """Applies exponential synaptic decay and prunes dead connections.

    new_strength = strength × exp(-λ × days_since_last_reinforced)
    """

    def __init__(
        self,
        pocketbase_url: Optional[str] = None,
        db_path: Optional[Path] = None,
        default_lambda: float = DEFAULT_LAMBDA,
        amygdala_lambda: float = AMYGDALA_LAMBDA,
    ):
        self._base_url = (pocketbase_url or API.POCKETBASE_URL).rstrip("/")
        self._api = f"{self._base_url}/api/collections"
        self._headers = HTTP.build_headers(API.POCKETBASE_AUTH_TOKEN)
        self._db_path = str(db_path or Paths.BRAIN_DB)
        self._default_lambda = default_lambda
        self._amygdala_lambda = amygdala_lambda
        self._local = threading.local()
        self._init_decay_log()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_decay_log(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decay_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at          TEXT    NOT NULL,
                synapses_decayed INTEGER NOT NULL DEFAULT 0,
                synapses_pruned  INTEGER NOT NULL DEFAULT 0,
                details         TEXT    NOT NULL DEFAULT '{}'
            )
        """)
        conn.commit()

    # ── Fetch all synapses from PocketBase ───────────────

    def _fetch_all_synapses(self) -> list[dict]:
        """Paginate through all synapse_scores records."""
        all_records = []
        page = 1
        while True:
            try:
                r = HTTP.request(
                    "GET",
                    f"{self._api}/synapse_scores/records",
                    service_name="PocketBase",
                    params={"page": str(page), "perPage": str(MAX_BATCH_SIZE)},
                    timeout=10,
                    headers=self._headers,
                )
                if r.status_code != 200:
                    break
                data = r.json()
                items = data.get("items", [])
                all_records.extend(items)
                if page >= data.get("totalPages", 1):
                    break
                page += 1
            except Exception as e:
                logger.warning("Failed to fetch synapses page %d: %s", page, e)
                break
        return all_records

    # ── Emotional tag lookup ─────────────────────────────

    def _is_amygdala_tagged(self, neuron_id: str) -> bool:
        """Check if a neuron has an amygdala emotional tag."""
        try:
            r = HTTP.request(
                "GET",
                f"{self._api}/emotional_tags/records",
                service_name="PocketBase",
                params={"filter": f'neuron_id="{neuron_id}"', "perPage": "1"},
                timeout=5,
                headers=self._headers,
            )
            if r.status_code == 200:
                items = r.json().get("items", [])
                if items:
                    return items[0].get("flagged", False)
        except Exception:
            pass
        return False

    # ── Core decay logic ─────────────────────────────────

    def apply_decay(self, dry_run: bool = False) -> dict:
        """Apply exponential decay to all synapses.

        Returns summary dict with counts.
        """
        records = self._fetch_all_synapses()
        now = datetime.now(timezone.utc)
        decayed = 0
        pruned = 0
        details: list[dict] = []

        # Cache amygdala lookups to avoid repeated API calls
        amygdala_cache: dict[str, bool] = {}

        for rec in records:
            strength = rec.get("strength", 0)
            last_reinforced = rec.get("last_reinforced", "")
            record_id = rec.get("id", "")

            if not last_reinforced or strength <= 0:
                continue

            # Parse last_reinforced date
            try:
                lr_date = datetime.fromisoformat(
                    last_reinforced.replace("Z", "+00:00")
                )
                if lr_date.tzinfo is None:
                    lr_date = lr_date.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            days_elapsed = (now - lr_date).total_seconds() / 86400.0
            if days_elapsed < 1.0:
                continue  # don't decay things reinforced today

            # Determine decay rate — slower for emotional memories
            source_id = rec.get("source_id", "")
            target_id = rec.get("target_id", "")

            for nid in (source_id, target_id):
                if nid not in amygdala_cache:
                    amygdala_cache[nid] = self._is_amygdala_tagged(nid)

            is_emotional = amygdala_cache.get(source_id, False) or amygdala_cache.get(target_id, False)
            decay_lambda = self._amygdala_lambda if is_emotional else self._default_lambda

            new_strength = strength * math.exp(-decay_lambda * days_elapsed)
            new_strength_int = max(0, int(round(new_strength)))

            if new_strength_int == strength:
                continue  # no change

            if new_strength_int < MIN_STRENGTH_THRESHOLD:
                # Prune
                if not dry_run:
                    self._delete_synapse(record_id)
                pruned += 1
                details.append({
                    "action": "pruned",
                    "source": source_id,
                    "target": target_id,
                    "old_strength": strength,
                    "days_idle": round(days_elapsed, 1),
                })
            else:
                # Decay
                if not dry_run:
                    self._update_strength(record_id, new_strength_int)
                decayed += 1
                details.append({
                    "action": "decayed",
                    "source": source_id,
                    "target": target_id,
                    "old_strength": strength,
                    "new_strength": new_strength_int,
                    "days_idle": round(days_elapsed, 1),
                    "emotional": is_emotional,
                })

        # Log results
        result = {
            "total_synapses": len(records),
            "decayed": decayed,
            "pruned": pruned,
            "dry_run": dry_run,
        }

        if not dry_run:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO decay_log (run_at, synapses_decayed, synapses_pruned, details) VALUES (?, ?, ?, ?)",
                (now.isoformat(), decayed, pruned, json.dumps(details[:50])),
            )
            conn.commit()
            EventBus.get().emit("decay.applied", result)

        logger.info("Decay pass: %s", result)
        return result

    def preview_decay(self) -> dict:
        """Dry-run decay to show what would happen."""
        return self.apply_decay(dry_run=True)

    # ── PocketBase mutations ─────────────────────────────

    def _update_strength(self, record_id: str, new_strength: int):
        try:
            HTTP.request(
                "PATCH",
                f"{self._api}/synapse_scores/records/{record_id}",
                service_name="PocketBase",
                json={"strength": new_strength, "supabase_synced": False},
                timeout=5,
                headers=self._headers,
            )
        except Exception as e:
            logger.warning("Failed to update synapse %s: %s", record_id, e)

    def _delete_synapse(self, record_id: str):
        try:
            HTTP.request(
                "DELETE",
                f"{self._api}/synapse_scores/records/{record_id}",
                service_name="PocketBase",
                timeout=5,
                headers=self._headers,
            )
        except Exception as e:
            logger.warning("Failed to prune synapse %s: %s", record_id, e)

    # ── Stats ────────────────────────────────────────────

    def recent_decay_stats(self, limit: int = 5) -> list[dict]:
        """Return recent decay log entries."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM decay_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "run_at": r["run_at"],
                "decayed": r["synapses_decayed"],
                "pruned": r["synapses_pruned"],
            }
            for r in rows
        ]
