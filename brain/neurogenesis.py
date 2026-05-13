"""
Neurogenesis engine — autonomous knowledge generation.

Biological basis: the hippocampus generates new neurons throughout life.
These new neurons help form novel representations by synthesizing
patterns across existing knowledge that aren't explicitly captured.

Rate-limited to 3 new neurons per nightly run.  All auto-generated
neurons are flagged so the user can review/delete.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Optional

from brain.config import Brain, Paths
from brain.eventbus import EventBus
from brain.models import AtomicNeuron, NeuronSource
from brain.router import Router
from brain.synapses import SynapseManager
from brain.vault import Vault
from brain.vectors import VectorStore

logger = logging.getLogger(__name__)

MAX_NEW_NEURONS_PER_RUN = 3

INSIGHT_PROMPT = """You are an expert knowledge synthesizer. Given these related knowledge fragments,
generate ONE novel insight that isn't explicitly stated but emerges from their combination.

Knowledge fragments:
{fragments}

Requirements:
- The insight must be non-obvious and genuinely useful
- 2-4 sentences maximum
- Must be a new connection or pattern, not a summary

Reply as JSON: {{"title": "...", "body": "...", "tags": ["..."]}}
Return ONLY valid JSON."""


class NeurogenesisEngine:
    """Generates new knowledge neurons from existing knowledge clusters."""

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

    def run_neurogenesis(self) -> dict:
        """Generate new insight neurons from strongest clusters.

        Returns summary dict.
        """
        clusters = self._find_strong_clusters()
        generated = []

        for cluster in clusters[:MAX_NEW_NEURONS_PER_RUN]:
            neuron = self._generate_insight(cluster)
            if neuron:
                generated.append({
                    "id": neuron.id,
                    "title": neuron.title,
                    "parent_ids": cluster["neuron_ids"],
                })

        result = {
            "clusters_found": len(clusters),
            "neurons_generated": len(generated),
            "neurons": generated,
        }

        if generated:
            self._write_report(generated)
            EventBus.get().emit("neurogenesis.completed", result)

        logger.info("Neurogenesis: %s", result)
        return result

    def _find_strong_clusters(self, min_cluster_size: int = 3) -> list[dict]:
        """Find clusters of strongly-connected neurons.

        Uses top synapses to identify tightly-knit groups.
        """
        top = self.synapses.top_synapses(limit=50)
        if not top:
            return []

        # Build adjacency from top synapses
        adj: dict[str, set[str]] = {}
        for s in top:
            adj.setdefault(s.source_id, set()).add(s.target_id)
            adj.setdefault(s.target_id, set()).add(s.source_id)

        # Simple greedy clustering: start from highest-degree nodes
        visited: set[str] = set()
        clusters = []

        sorted_nodes = sorted(adj.keys(), key=lambda n: len(adj[n]), reverse=True)

        for seed in sorted_nodes:
            if seed in visited:
                continue
            cluster = {seed}
            visited.add(seed)
            for neighbor in adj.get(seed, set()):
                if neighbor not in visited:
                    cluster.add(neighbor)
                    visited.add(neighbor)

            if len(cluster) >= min_cluster_size:
                # Load neuron content for this cluster
                fragments = []
                for nid in list(cluster)[:6]:  # cap for prompt size
                    neuron = self.vault.read_neuron_by_id(nid)
                    if neuron:
                        fragments.append(f"[{nid}] {neuron.title}: {neuron.body[:200]}")

                if len(fragments) >= min_cluster_size:
                    clusters.append({
                        "neuron_ids": list(cluster)[:6],
                        "fragments": fragments,
                        "size": len(cluster),
                    })

        return clusters

    def _generate_insight(self, cluster: dict) -> Optional[AtomicNeuron]:
        """Use LLM to synthesize an insight from a cluster."""
        fragments_text = "\n".join(cluster["fragments"])
        prompt = INSIGHT_PROMPT.format(fragments=fragments_text)

        try:
            response = self.router.ask_local(prompt)
            # Parse JSON
            import re
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())

            neuron = AtomicNeuron(
                region="creative",
                title=data.get("title", "Auto-generated insight"),
                source="neurogenesis",
                tags=data.get("tags", []) + ["auto-generated", "neurogenesis"],
                body=data.get("body", ""),
            )
            self.vault.write_neuron(neuron)

            # Create synapses to parent cluster
            for parent_id in cluster["neuron_ids"]:
                self.synapses.reinforce(
                    neuron.id, parent_id,
                    event="neurogenesis-parent",
                    score=Brain.SYNAPSE_AI_CONFIRMED,
                )

            # Generate and store embedding
            text = f"{neuron.title}\n{neuron.body}"
            embedding = self.router.get_embedding(text)
            if embedding:
                self.vectors.store_neuron(
                    neuron_id=neuron.id,
                    region=neuron.region,
                    embedding=embedding,
                    document=text,
                    metadata={"title": neuron.title, "source": "neurogenesis",
                              "created": neuron.created},
                )

            logger.info("Neurogenesis: created %s — %s", neuron.id, neuron.title)
            return neuron

        except Exception as e:
            logger.warning("Insight generation failed: %s", e)
            return None

    def _write_report(self, generated: list[dict]):
        """Write neurogenesis report to _meta/."""
        content = f"# 🧬 Neurogenesis Report — {date.today().isoformat()}\n\n"
        content += f"**{len(generated)} new neurons generated**\n\n"
        for g in generated:
            content += f"## [[{g['id']}]] {g['title']}\n"
            content += f"Parent cluster: {', '.join(f'[[{p}]]' for p in g['parent_ids'])}\n\n"
        self.vault.write_meta("neurogenesis-report.md", content)
