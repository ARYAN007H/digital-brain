"""
Spreading activation engine.

Biological basis:
    When you think of "dog", neurons for "cat", "pet", "bark", "leash"
    all partially activate.  Energy propagates through the synaptic
    graph, attenuating with distance.  This is how the brain does
    associative retrieval — related concepts "light up" even without
    direct semantic match.

The current system only retrieves direct ChromaDB hits.  Spreading
activation adds *graph-augmented retrieval*: neurons connected through
strong synapses are surfaced even if their embeddings aren't close.
"""

from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import Optional

from brain.config import API, Brain, HTTP, Paths
from brain.synapses import SynapseManager

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────
DEFAULT_MAX_HOPS = 3
DEFAULT_DECAY_PER_HOP = 0.5
DEFAULT_MIN_ACTIVATION = 0.05
DEFAULT_TOP_K = 10
MAX_SYNAPSE_FETCH = 500


class SpreadingActivationEngine:
    """Propagate activation through the synapse graph from seed neurons.

    Algorithm:
        1. Seed neurons get activation = 1.0.
        2. BFS outward through synapse edges.
        3. At each hop, activation = parent_activation × (edge_strength / max_strength) × decay.
        4. Accumulate activation per neuron across all paths.
        5. Return top-K by total activation.

    This is pure Python, no heavy dependencies, ~O(V + E) per query.
    """

    def __init__(self, synapses: Optional[SynapseManager] = None):
        self._syn = synapses or SynapseManager()

    def _fetch_adjacency(self) -> tuple[dict[str, list[tuple[str, int]]], int]:
        """Build adjacency list from PocketBase synapse_scores.

        Returns (adjacency_dict, max_strength).
        """
        adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
        max_strength = 1

        try:
            base_url = API.POCKETBASE_URL.rstrip("/")
            headers = HTTP.build_headers(API.POCKETBASE_AUTH_TOKEN)
            page = 1
            while True:
                r = HTTP.request(
                    "GET",
                    f"{base_url}/api/collections/synapse_scores/records",
                    service_name="PocketBase",
                    params={"page": str(page), "perPage": str(MAX_SYNAPSE_FETCH), "sort": "-strength"},
                    timeout=10,
                    headers=headers,
                )
                if r.status_code != 200:
                    break
                data = r.json()
                for item in data.get("items", []):
                    s = item.get("source_id", "")
                    t = item.get("target_id", "")
                    w = item.get("strength", 0)
                    if s and t and w > 0:
                        adj[s].append((t, w))
                        adj[t].append((s, w))
                        max_strength = max(max_strength, w)
                if page >= data.get("totalPages", 1):
                    break
                page += 1
        except Exception as e:
            logger.warning("Failed to build adjacency: %s", e)

        return dict(adj), max_strength

    def activate(
        self,
        seed_ids: list[str],
        max_hops: int = DEFAULT_MAX_HOPS,
        decay: float = DEFAULT_DECAY_PER_HOP,
        min_activation: float = DEFAULT_MIN_ACTIVATION,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Run spreading activation from seed neurons.

        Args:
            seed_ids: Starting neuron IDs (e.g. from semantic search).
            max_hops: Maximum graph hops to propagate.
            decay: Activation decay factor per hop (0-1).
            min_activation: Stop propagating below this threshold.
            top_k: Return this many activated neurons.

        Returns:
            List of {neuron_id, activation, hops, path} sorted by
            activation descending.  Seed neurons are excluded.
        """
        if not seed_ids:
            return []

        adj, max_strength = self._fetch_adjacency()
        if not adj:
            return []

        # Activation accumulator
        activation: dict[str, float] = defaultdict(float)
        visited_hops: dict[str, int] = {}

        # BFS with priority queue (max-heap via negative activation)
        # Queue entries: (-activation, hop_count, neuron_id)
        queue: list[tuple[float, int, str]] = []
        seed_set = set(seed_ids)

        for sid in seed_ids:
            activation[sid] = 1.0
            visited_hops[sid] = 0
            heapq.heappush(queue, (-1.0, 0, sid))

        while queue:
            neg_act, hop, current = heapq.heappop(queue)
            current_act = -neg_act

            if hop >= max_hops:
                continue

            neighbors = adj.get(current, [])
            for neighbor_id, edge_strength in neighbors:
                # Compute propagated activation
                prop = current_act * (edge_strength / max_strength) * decay

                if prop < min_activation:
                    continue

                new_hop = hop + 1

                # Accumulate (a neuron can be reached via multiple paths)
                if prop > activation.get(neighbor_id, 0):
                    activation[neighbor_id] = max(activation[neighbor_id], prop)
                    visited_hops[neighbor_id] = min(
                        visited_hops.get(neighbor_id, new_hop), new_hop
                    )
                    heapq.heappush(queue, (-prop, new_hop, neighbor_id))

        # Build results — exclude seeds
        results = []
        for nid, act in activation.items():
            if nid in seed_set:
                continue
            results.append({
                "neuron_id": nid,
                "activation": round(act, 4),
                "hops": visited_hops.get(nid, 0),
            })

        results.sort(key=lambda x: x["activation"], reverse=True)
        return results[:top_k]

    def activate_from_query_results(
        self,
        semantic_results: list[dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """Convenience: run activation from semantic search result dicts.

        Expects each dict to have an "id" key.
        """
        seed_ids = [r["id"] for r in semantic_results if r.get("id")]
        return self.activate(seed_ids, top_k=top_k)
