"""
Sleep-phase memory consolidation engine.

Biological basis: during slow-wave sleep the hippocampus replays recent
episodes to the neocortex, merging near-duplicates, bridging disconnected
clusters, and promoting important memories to long-term storage.

Runs as part of the nightly job.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from brain.config import Brain, Paths
from brain.eventbus import EventBus
from brain.models import AtomicNeuron, MemoryNeuron
from brain.router import Router
from brain.synapses import SynapseManager
from brain.vault import Vault
from brain.vectors import VectorStore

logger = logging.getLogger(__name__)

MERGE_SIMILARITY_THRESHOLD = 0.92
PROMOTION_ACCESS_THRESHOLD = 5
MAX_BRIDGES_PER_RUN = 5


class ConsolidationEngine:
    """Nightly sleep-phase consolidation: replay, merge, promote, bridge."""

    def __init__(
        self,
        vault: Optional[Vault] = None,
        vectors: Optional[VectorStore] = None,
        synapses: Optional[SynapseManager] = None,
        router: Optional[Router] = None,
    ):
        self.vault = vault or Vault()
        self.vectors = vectors or VectorStore()
        self.synapses = synapses or SynapseManager()
        self.router = router or Router()

    def run_consolidation(self) -> dict:
        """Full consolidation pass. Returns summary."""
        results = {"replay": 0, "bridges": 0, "promotions": []}

        # Phase 1: Replay — re-check recent neurons for missed connections
        replay = self._replay_recent()
        results["replay"] = replay

        # Phase 2: Bridge disconnected clusters
        bridges = self._bridge_clusters()
        results["bridges"] = bridges

        # Phase 3: Promotion suggestions
        promotions = self._suggest_promotions()
        results["promotions"] = promotions

        EventBus.get().emit("consolidation.done", results)
        logger.info("Consolidation complete: %s", results)
        return results

    def _replay_recent(self, days: int = 1) -> int:
        """Re-check neurons from the last N days for missed semantic links."""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        reinforced = 0

        recent = []
        for fp in self.vault.list_neurons():
            try:
                n = self.vault.read_neuron(fp)
                if n.created and n.created >= cutoff and n.region != "amygdala":
                    recent.append(n)
            except Exception:
                continue

        # For each recent neuron, find similar neurons it may not be connected to
        for neuron in recent[:50]:  # cap to avoid overloading
            similar = self.vectors.get_similar(neuron.id, neuron.region, threshold=0.80, top_k=5)
            for match in similar:
                existing = self.synapses.get_connections(neuron.id)
                connected_ids = {s.source_id for s in existing} | {s.target_id for s in existing}
                if match["id"] not in connected_ids:
                    self.synapses.reinforce(neuron.id, match["id"], "consolidation-replay",
                                            Brain.SYNAPSE_SEMANTIC_SIM)
                    reinforced += 1

        logger.info("Replay: %d new connections discovered", reinforced)
        return reinforced

    def _bridge_clusters(self) -> int:
        """Find neurons that could bridge disconnected clusters."""
        bridges = 0
        all_neurons = self.vault.list_neurons()
        if len(all_neurons) < 10:
            return 0

        # Sample neurons with zero or few connections
        isolated = []
        for fp in all_neurons[-100:]:  # check recent
            try:
                n = self.vault.read_neuron(fp)
                conns = self.synapses.get_connections(n.id)
                if len(conns) <= 1 and n.region != "amygdala":
                    isolated.append(n)
            except Exception:
                continue

        # For each isolated neuron, search ALL regions for matches
        for neuron in isolated[:20]:
            for region in Brain.CHROMA_COLLECTIONS:
                if region == neuron.region:
                    continue
                matches = self.vectors.get_similar(neuron.id, region, threshold=0.78, top_k=2)
                for m in matches:
                    self.synapses.reinforce(neuron.id, m["id"], "consolidation-bridge",
                                            Brain.SYNAPSE_SEMANTIC_SIM)
                    bridges += 1
                    if bridges >= MAX_BRIDGES_PER_RUN:
                        return bridges

        logger.info("Bridge: %d cross-cluster connections created", bridges)
        return bridges

    def _suggest_promotions(self) -> list[dict]:
        """Suggest region reclassification for frequently-accessed hippocampus neurons."""
        promotions = []
        for fp in self.vault.list_neurons(region="hippocampus"):
            try:
                n = self.vault.read_neuron(fp)
                conns = self.synapses.get_connections(n.id)
                total_strength = sum(c.strength for c in conns)
                if total_strength >= PROMOTION_ACCESS_THRESHOLD * Brain.SYNAPSE_CO_ACCESS:
                    # This neuron is heavily connected — might belong in prefrontal or predictive
                    promotions.append({
                        "neuron_id": n.id,
                        "title": n.title,
                        "current_region": "hippocampus",
                        "total_connection_strength": total_strength,
                        "suggestion": "Consider promoting to prefrontal (planning) or predictive (patterns)",
                    })
            except Exception:
                continue
            if len(promotions) >= 10:
                break

        if promotions:
            # Write promotion suggestions to meta
            content = "# 🔄 Promotion Suggestions\n\n"
            for p in promotions:
                content += f"- **[[{p['neuron_id']}]]** {p['title']} (strength: {p['total_connection_strength']})\n"
                content += f"  {p['suggestion']}\n\n"
            self.vault.write_meta("promotion-suggestions.md", content)

        return promotions
