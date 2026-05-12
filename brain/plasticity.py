"""Online plasticity engine: reinforce connections from recent usage signals."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from itertools import combinations
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from brain.config import Brain, Paths
from brain.synapses import SynapseManager

logger = logging.getLogger(__name__)


ID_PATTERN = re.compile(r"\b(?:NRN|MEM)-\d{8}-\d{3,4}\b")


class PlasticityEngine:
    """Lightweight brain-like plasticity pass for low-resource hardware."""

    def __init__(self):
        self._db = Paths.BRAIN_DB
        self._syn = SynapseManager()

    def _recent_text(self, hours: int = 24) -> list[str]:
        if not self._db.exists():
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        conn = sqlite3.connect(str(self._db))
        try:
            rows = conn.execute(
                """SELECT content FROM conversation_history
                   WHERE created_at >= ?
                   ORDER BY created_at DESC
                   LIMIT 800""",
                (cutoff,),
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def reinforce_from_recent_activity(self, hours: int = 24, dry_run: bool = False) -> dict:
        """Reinforce synapses from co-mentioned neuron ids in recent conversations.

        Uses time-aware weighting: recent traces contribute larger reinforcement.
        """
        texts = self._recent_text(hours=hours)
        pair_scores: dict[tuple[str, str], float] = defaultdict(float)

        for idx, t in enumerate(texts):
            ids = sorted(set(ID_PATTERN.findall(t or "")))
            if len(ids) < 2:
                continue
            # recent items in the list get slightly larger weight
            recency_weight = 1.0 + max(0.0, (len(texts) - idx) / max(1, len(texts)))
            for a, b in combinations(ids, 2):
                pair_scores[(a, b)] += recency_weight

        reinforced = 0
        total_score = 0
        for (a, b), raw_score in pair_scores.items():
            score = min(max(1, int(round(raw_score))), 6)
            total_score += score
            if not dry_run:
                self._syn.reinforce(a, b, event="co-access", score=score)
            reinforced += 1

        logger.info("Plasticity pass complete: %s pairs reinforced", reinforced)
        return {
            "pairs_reinforced": reinforced,
            "source_texts": len(texts),
            "total_reinforcement": total_score,
            "dry_run": dry_run,
        }

    def run_forever(self, interval_sec: int = 300):
        """Continuous low-cost plasticity updates."""
        logger.info("Starting plasticity engine loop (%ss)", interval_sec)
        while True:
            try:
                self.reinforce_from_recent_activity(hours=24, dry_run=False)
            except Exception as e:
                logger.warning("Plasticity pass failed: %s", e)
            time.sleep(max(30, interval_sec))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    PlasticityEngine().run_forever()


if __name__ == "__main__":
    main()
