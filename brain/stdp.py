"""
Spike-Timing Dependent Plasticity (STDP) engine.

Biological basis:
    In real synapses, the *order* and *timing* of firing determines whether
    a connection strengthens (LTP) or weakens (LTD).  If neuron A fires
    just before neuron B, the A→B synapse strengthens.  If B fires before
    A, the synapse weakens.  The magnitude follows an exponential decay
    with a time constant τ.

    Our "firing" = a neuron being accessed (read, queried, referenced).
    Our τ = 300 seconds (5 minutes) by default.

This replaces the blunt co-mention scan in the original PlasticityEngine
with temporally precise, directional reinforcement.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

from brain.config import Brain, Paths
from brain.eventbus import EventBus
from brain.synapses import SynapseManager

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────
TAU_SECONDS = 300.0          # time constant for exponential window
LTP_MAX_SCORE = 5            # max reinforcement per pair per pass
LTD_MAX_SCORE = -2           # max weakening (anti-Hebbian)
STDP_WINDOW_SECONDS = 600    # 10 min — events further apart are ignored


class STDPEngine:
    """Spike-timing dependent plasticity for the digital brain.

    Listens to neuron.accessed events (via EventBus) and periodically
    computes directional reinforcement between temporally proximate
    neuron access pairs.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        synapses: Optional[SynapseManager] = None,
    ):
        self._db_path = str(db_path or Paths.BRAIN_DB)
        self._syn = synapses or SynapseManager()
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
        """Create the neuron_access_events table for sub-second timing."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS neuron_access_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                neuron_id   TEXT    NOT NULL,
                context     TEXT    NOT NULL DEFAULT 'query',
                timestamp   REAL   NOT NULL,
                session_id  TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_time
            ON neuron_access_events(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_neuron
            ON neuron_access_events(neuron_id, timestamp)
        """)
        conn.commit()

    # ── Event recording ──────────────────────────────────

    def record_access(self, neuron_id: str, context: str = "query", session_id: str = ""):
        """Record a neuron access event with high-resolution timestamp."""
        now = datetime.now(timezone.utc).timestamp()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO neuron_access_events (neuron_id, context, timestamp, session_id) VALUES (?, ?, ?, ?)",
            (neuron_id, context, now, session_id),
        )
        conn.commit()

    def record_access_batch(self, neuron_ids: list[str], context: str = "query", session_id: str = ""):
        """Record multiple access events at the same timestamp."""
        now = datetime.now(timezone.utc).timestamp()
        conn = self._get_conn()
        conn.executemany(
            "INSERT INTO neuron_access_events (neuron_id, context, timestamp, session_id) VALUES (?, ?, ?, ?)",
            [(nid, context, now, session_id) for nid in neuron_ids],
        )
        conn.commit()

    # ── STDP scoring ─────────────────────────────────────

    @staticmethod
    def compute_stdp_score(delta_t_seconds: float, tau: float = TAU_SECONDS) -> float:
        """Compute STDP score from time difference.

        Args:
            delta_t_seconds: t_post - t_pre.
                Positive = pre fired before post → LTP (strengthen).
                Negative = post fired before pre → LTD (weaken).
            tau: time constant in seconds.

        Returns:
            Score in range [LTD_MAX_SCORE, LTP_MAX_SCORE].
        """
        if abs(delta_t_seconds) > STDP_WINDOW_SECONDS:
            return 0.0

        magnitude = math.exp(-abs(delta_t_seconds) / tau)

        if delta_t_seconds >= 0:
            # Pre before post → potentiation
            return magnitude * LTP_MAX_SCORE
        else:
            # Post before pre → depression
            return magnitude * LTD_MAX_SCORE

    # ── Main plasticity pass ─────────────────────────────

    def run_stdp_pass(self, hours: int = 24, dry_run: bool = False) -> dict:
        """Compute and apply STDP updates from recent access events.

        For every pair of neurons accessed within the STDP window,
        compute a directional score based on firing order and timing.

        Returns summary dict.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        conn = self._get_conn()

        rows = conn.execute(
            """SELECT neuron_id, timestamp FROM neuron_access_events
               WHERE timestamp >= ?
               ORDER BY timestamp ASC""",
            (cutoff,),
        ).fetchall()

        if len(rows) < 2:
            return {"pairs_evaluated": 0, "potentiated": 0, "depressed": 0, "dry_run": dry_run}

        # Build per-neuron access timeline
        timelines: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            timelines[row["neuron_id"]].append(row["timestamp"])

        neuron_ids = list(timelines.keys())
        potentiated = 0
        depressed = 0
        pairs_evaluated = 0

        for i, id_a in enumerate(neuron_ids):
            for id_b in neuron_ids[i + 1:]:
                # Find the most recent access pair within the window
                times_a = timelines[id_a]
                times_b = timelines[id_b]

                # Use the latest access of each to compute delta
                latest_a = times_a[-1]
                latest_b = times_b[-1]
                delta_t = latest_b - latest_a  # positive if A fired first

                score = self.compute_stdp_score(delta_t)
                if score == 0.0:
                    continue

                pairs_evaluated += 1
                int_score = max(-2, min(5, int(round(score))))

                if int_score == 0:
                    continue

                if not dry_run:
                    if int_score > 0:
                        self._syn.reinforce(id_a, id_b, event="stdp-ltp", score=int_score)
                        potentiated += 1
                    else:
                        # For depression: reduce strength (reinforce with negative)
                        self._syn.reinforce(id_a, id_b, event="stdp-ltd", score=int_score)
                        depressed += 1
                else:
                    if int_score > 0:
                        potentiated += 1
                    else:
                        depressed += 1

        result = {
            "pairs_evaluated": pairs_evaluated,
            "potentiated": potentiated,
            "depressed": depressed,
            "total_events": len(rows),
            "unique_neurons": len(neuron_ids),
            "dry_run": dry_run,
        }
        logger.info("STDP pass: %s", result)

        # Emit event
        if not dry_run:
            EventBus.get().emit("stdp.completed", result)

        return result

    # ── EventBus integration ─────────────────────────────

    def on_neuron_accessed(self, event_type: str, payload: dict):
        """EventBus subscriber: record access events in real time."""
        neuron_id = payload.get("neuron_id", "")
        if neuron_id:
            self.record_access(
                neuron_id,
                context=payload.get("context", "query"),
                session_id=payload.get("session_id", ""),
            )

        # Also handle batch access
        neuron_ids = payload.get("neuron_ids", [])
        if neuron_ids:
            self.record_access_batch(
                neuron_ids,
                context=payload.get("context", "query"),
                session_id=payload.get("session_id", ""),
            )

    def register(self, bus: Optional[EventBus] = None):
        """Register this engine's subscribers on the event bus."""
        bus = bus or EventBus.get()
        bus.subscribe("neuron.accessed", self.on_neuron_accessed)
        logger.info("STDP engine registered on event bus")

    # ── Cleanup ──────────────────────────────────────────

    def purge_old_events(self, keep_days: int = 7):
        """Remove access events older than keep_days."""
        cutoff = datetime.now(timezone.utc).timestamp() - (keep_days * 86400)
        conn = self._get_conn()
        conn.execute("DELETE FROM neuron_access_events WHERE timestamp < ?", (cutoff,))
        conn.commit()

    def access_count(self) -> int:
        """Total recorded access events."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM neuron_access_events").fetchone()
        return row[0] if row else 0
