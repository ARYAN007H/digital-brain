"""
Proactive surfacing and brain health stats.

Nightly cron (2am) generates:
- _meta/daily-surface.md  — 3 most relevant notes today
- _meta/weekly-patterns.md — emerging patterns last 7 days
- _meta/strong-synapses.md — top 10 strongest connections
- _meta/brain-stats.md     — full brain health report
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

from brain.queue import WriteQueue
from brain.router import Router
from brain.synapses import SynapseManager
from brain.vault import Vault
from brain.vectors import VectorStore
from brain.wikilink_scanner import WikilinkScanner

logger = logging.getLogger(__name__)


class Surfacer:
    """Generates proactive insights and brain health reports."""

    def __init__(
        self,
        vault: Vault | None = None,
        vectors: VectorStore | None = None,
        synapses: SynapseManager | None = None,
        router: Router | None = None,
        queue: WriteQueue | None = None,
    ):
        self.vault = vault or Vault()
        self.vectors = vectors or VectorStore()
        self.synapses = synapses or SynapseManager()
        self.router = router or Router()
        self.queue = queue or WriteQueue()

    def generate_all(self):
        """Run all surfacing tasks. Called by nightly cron."""
        logger.info("Starting nightly surfacing...")
        self.generate_brain_stats()
        self.generate_daily_surface()
        self.generate_strong_synapses()
        self.generate_weekly_patterns()
        self._scan_wikilinks()
        logger.info("Nightly surfacing complete")

    def _scan_wikilinks(self):
        """Scan entire vault for wikilinks and reinforce synapses."""
        scanner = WikilinkScanner(vault=self.vault, synapses=self.synapses)
        stats = scanner.scan_vault()
        logger.info(f"Wikilink scan: {stats}")

    def generate_brain_stats(self):
        """Generate _meta/brain-stats.md with full brain health report."""
        counts = self.vault.count_neurons()
        synapse_count = self.synapses.total_count()
        top = self.synapses.top_synapses(1)
        queue_counts = self.queue.total_count()
        vector_counts = self.vectors.count()
        inbox_count = self.vault.inbox_count()
        ingestion_metrics = {"rejected": 0, "quarantined": 0, "reasons": {}}
        metrics_path = self.vault.root / "_meta" / "ingestion-health.json"
        if metrics_path.exists():
            try:
                ingestion_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Invalid ingestion-health.json format")

        strongest = ""
        avg_strength = 0
        if top:
            s = top[0]
            strongest = f"{s.source_id} ↔ {s.target_id} (score: {s.strength})"
            avg_strength = s.strength  # simplified; real avg needs all synapses

        # Count atomic vs memory
        atomic_count = 0
        memory_count = 0
        for fp in self.vault.list_neurons():
            n = self.vault.read_neuron(fp)
            if n.type == "atomic-note":
                atomic_count += 1
            else:
                memory_count += 1

        stats = f"""# 🧠 Brain Health

```
Total Neurons:          {counts['total']}
├── Atomic:             {atomic_count}
└── Memory:             {memory_count}

By Region:
├── Prefrontal:         {counts.get('prefrontal', 0)}
├── Hippocampus:        {counts.get('hippocampus', 0)}
├── Creative:           {counts.get('creative', 0)}
├── Predictive:         {counts.get('predictive', 0)}
├── Amygdala:           {counts.get('amygdala', 0)}
└── Executive:          {counts.get('executive', 0)}

Vectors:
├── Prefrontal:         {vector_counts.get('prefrontal', 0)}
├── Hippocampus:        {vector_counts.get('hippocampus', 0)}
├── Creative:           {vector_counts.get('creative', 0)}
├── Predictive:         {vector_counts.get('predictive', 0)}
└── Executive:          {vector_counts.get('executive', 0)}

Synapses:
├── Total:              {synapse_count}
├── Strongest pair:     {strongest or 'none yet'}
└── Avg strength:       {avg_strength}

Sync status:
├── Supabase pending:   {queue_counts.get('pending', 0)}
├── Processing:         {queue_counts.get('processing', 0)}
└── Completed:          {queue_counts.get('done', 0)}

Inbox pending:          {inbox_count}

Ingestion safety:
├── Rejected files:     {ingestion_metrics.get('rejected', 0)}
└── Quarantined files:  {ingestion_metrics.get('quarantined', 0)}

Last updated:           {datetime.now().isoformat()}
```
"""
        self.vault.write_meta("brain-stats.md", stats)
        logger.info("Brain stats updated")

    def generate_daily_surface(self):
        """Generate _meta/daily-surface.md — 3 most relevant notes."""
        all_neurons = self.vault.list_neurons()
        if not all_neurons:
            return

        # Pick recent neurons with strongest connections
        recent = []
        today = date.today()
        for fp in all_neurons[-20:]:  # last 20 created
            n = self.vault.read_neuron(fp)
            conns = self.synapses.get_connections(n.id)
            total_strength = sum(c.strength for c in conns)
            recent.append((n, total_strength))

        # Sort by connection strength
        recent.sort(key=lambda x: x[1], reverse=True)
        top_3 = recent[:3]

        content = f"# 🌅 Daily Surface — {today.isoformat()}\n\n"
        for neuron, strength in top_3:
            content += f"## [[{neuron.id}]] {neuron.title}\n"
            content += f"Region: {neuron.region} · Connections: {strength}\n\n"
            content += f"{neuron.body[:200]}...\n\n---\n\n"

        self.vault.write_meta("daily-surface.md", content)

    def generate_strong_synapses(self):
        """Generate _meta/strong-synapses.md — top 10 connections."""
        top = self.synapses.top_synapses(10)

        content = "# 🔗 Strongest Synapses\n\n"
        content += "| Source | Target | Strength | Events |\n"
        content += "|--------|--------|----------|--------|\n"

        for s in top:
            events = ", ".join(s.reinforcement_log[-3:])  # last 3 events
            content += f"| [[{s.source_id}]] | [[{s.target_id}]] | {s.strength} | {events} |\n"

        content += f"\n*Updated: {datetime.now().isoformat()}*\n"
        self.vault.write_meta("strong-synapses.md", content)

    def generate_weekly_patterns(self):
        """Generate _meta/weekly-patterns.md using LLM analysis."""
        # Gather recent neurons
        all_neurons = self.vault.list_neurons()
        week_ago = (date.today() - timedelta(days=7)).isoformat()

        recent_titles = []
        for fp in all_neurons:
            n = self.vault.read_neuron(fp)
            if n.created >= week_ago:
                recent_titles.append(f"- [{n.region}] {n.title}")

        if not recent_titles:
            return

        prompt = (
            "Analyze these note titles from the past week and identify "
            "2-3 emerging patterns or themes:\n\n"
            + "\n".join(recent_titles[:30])
            + "\n\nPatterns:"
        )

        response = self.router.ask_local(prompt)

        content = f"# 📊 Weekly Patterns — {date.today().isoformat()}\n\n"
        content += f"**Notes analyzed:** {len(recent_titles)}\n\n"
        content += response + "\n"
        content += f"\n*Generated: {datetime.now().isoformat()}*\n"

        self.vault.write_meta("weekly-patterns.md", content)


# ── CLI Entry Point ──────────────────────────────────────

def main():
    """Run surfacing manually."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    surfacer = Surfacer()
    surfacer.generate_all()


if __name__ == "__main__":
    main()
