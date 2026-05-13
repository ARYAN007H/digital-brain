"""
Thalamus relay — cross-region integration and binding.

Biological basis: the thalamus relays information across cortical
regions.  When multiple regions co-activate, the thalamus binds them
into a unified percept and creates cross-region connections.
"""

from __future__ import annotations

import logging
from typing import Optional

from brain.config import Brain
from brain.synapses import SynapseManager
from brain.vectors import VectorStore

logger = logging.getLogger(__name__)

# Region affinity matrix — how much each region "talks to" each other
# Values 0-1.  Higher = more likely to form cross-region synapses.
REGION_AFFINITY = {
    ("prefrontal", "executive"):  0.9,
    ("prefrontal", "predictive"): 0.7,
    ("hippocampus", "creative"):  0.6,
    ("hippocampus", "prefrontal"):0.7,
    ("creative", "predictive"):   0.5,
    ("executive", "predictive"):  0.6,
    ("hippocampus", "executive"): 0.5,
    ("creative", "executive"):    0.4,
    ("prefrontal", "creative"):   0.6,
    ("hippocampus", "predictive"):0.5,
}


def _affinity(r1: str, r2: str) -> float:
    """Get affinity between two regions (symmetric)."""
    if r1 == r2:
        return 1.0
    return max(
        REGION_AFFINITY.get((r1, r2), 0.3),
        REGION_AFFINITY.get((r2, r1), 0.3),
    )


class ThalamusRelay:
    """Cross-region search, binding, and integration."""

    def __init__(
        self,
        vectors: Optional[VectorStore] = None,
        synapses: Optional[SynapseManager] = None,
    ):
        self._vectors = vectors or VectorStore()
        self._syn = synapses or SynapseManager()

    def cross_region_search(
        self,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = 8,
    ) -> list[dict]:
        """Search ALL ChromaDB collections and merge with affinity weighting.

        Unlike single-region search, this finds cross-cutting connections.
        """
        all_results = []

        for region in Brain.CHROMA_COLLECTIONS:
            try:
                if query_text:
                    hits = self._vectors.hybrid_search(
                        query_text, query_embedding, region=region,
                        top_k=max(3, top_k // 2),
                    )
                else:
                    hits = self._vectors.search(
                        query_embedding, region=region,
                        top_k=max(3, top_k // 2),
                    )
                for h in hits:
                    h["source_region"] = region
                all_results.extend(hits)
            except Exception as e:
                logger.debug("Thalamus search in %s failed: %s", region, e)

        # Score with affinity — regions that appear together get boosted
        active_regions = set(r.get("source_region", "") for r in all_results)

        for r in all_results:
            base_score = r.get("hybrid_score") or max(0, 1.0 - r.get("distance", 1.0))
            # Cross-region bonus: if 3+ regions active, boost everything
            cross_bonus = 1.0 + 0.1 * max(0, len(active_regions) - 1)
            r["thalamic_score"] = round(base_score * cross_bonus, 4)

        all_results.sort(key=lambda x: x.get("thalamic_score", 0), reverse=True)
        return all_results[:top_k]

    def detect_and_bind(self, activated_neurons: list[dict]) -> int:
        """If neurons from 3+ regions co-activate, create cross-region synapses.

        Returns count of new synapses created.
        """
        by_region: dict[str, list[str]] = {}
        for n in activated_neurons:
            region = n.get("region") or n.get("source_region", "")
            nid = n.get("id") or n.get("neuron_id", "")
            if region and nid:
                by_region.setdefault(region, []).append(nid)

        if len(by_region) < 3:
            return 0  # no binding needed

        # Create cross-region synapses between top neuron per region
        created = 0
        region_reps = [(reg, ids[0]) for reg, ids in by_region.items() if ids]

        for i, (r1, id1) in enumerate(region_reps):
            for r2, id2 in region_reps[i + 1:]:
                affinity = _affinity(r1, r2)
                score = max(1, int(round(affinity * Brain.SYNAPSE_AI_CONFIRMED)))
                self._syn.reinforce(id1, id2, event=f"thalamic-bind-{r1}-{r2}", score=score)
                created += 1

        if created:
            logger.info("Thalamic binding: created %d cross-region synapses across %d regions",
                        created, len(by_region))
        return created
