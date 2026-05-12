"""
Obsidian vault reader/writer.

Handles reading and writing neuron .md files to the correct
cortex region folders in the Obsidian vault.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from brain.config import Paths
from brain.models import (
    AtomicNeuron,
    IDGenerator,
    MemoryNeuron,
    Neuron,
    _parse_frontmatter,
)

logger = logging.getLogger(__name__)


class Vault:
    """Obsidian vault interface — read/write neurons as .md files."""

    def __init__(self, vault_root: Optional[Path] = None):
        self.root = vault_root or Paths.VAULT

    def _region_path(self, region: str) -> Path:
        """Get the folder path for a cortex region."""
        path = Paths.REGIONS.get(region)
        if path is None:
            raise ValueError(f"Unknown region: {region}. Valid: {list(Paths.REGIONS.keys())}")
        return path

    # ── Write ────────────────────────────────────────────

    def write_neuron(self, neuron: Neuron) -> Path:
        """Write a neuron to the vault as a .md file.

        Returns the path to the written file.
        """
        region_dir = self._region_path(neuron.region)
        region_dir.mkdir(parents=True, exist_ok=True)

        filepath = region_dir / neuron.filename
        content = neuron.to_markdown()

        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Wrote neuron {neuron.id} → {filepath}")
        return filepath

    def write_meta(self, filename: str, content: str) -> Path:
        """Write a metadata file to _meta/ folder.

        Used for brain-stats.md, daily-surface.md, etc.
        """
        filepath = Paths.META / filename
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"Wrote meta file → {filepath}")
        return filepath

    def write_voice_log(self, entry: str) -> Path:
        """Append an entry to _meta/voice-log.md."""
        filepath = Paths.META / "voice-log.md"

        if filepath.exists():
            existing = filepath.read_text(encoding="utf-8")
            content = existing + "\n\n---\n\n" + entry
        else:
            content = "# Voice Log\n\n" + entry

        filepath.write_text(content, encoding="utf-8")
        return filepath

    # ── Read ─────────────────────────────────────────────

    def read_neuron(self, filepath: Path) -> Neuron:
        """Read a neuron from a .md file.

        Automatically detects whether it's an AtomicNeuron or MemoryNeuron
        based on the 'type' field in frontmatter.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Neuron file not found: {filepath}")

        content = filepath.read_text(encoding="utf-8")
        fm, _ = _parse_frontmatter(content)
        neuron_type = fm.get("type", "atomic-note")

        if neuron_type == "conversation-summary":
            return MemoryNeuron.from_markdown(content)
        else:
            return AtomicNeuron.from_markdown(content)

    def read_neuron_by_id(self, neuron_id: str) -> Optional[Neuron]:
        """Find and read a neuron by its ID. Searches all regions."""
        filename = f"{neuron_id}.md"
        for region, region_path in Paths.REGIONS.items():
            filepath = region_path / filename
            if filepath.exists():
                return self.read_neuron(filepath)
        return None

    # ── List / Search ────────────────────────────────────

    def list_neurons(self, region: Optional[str] = None) -> list[Path]:
        """List all neuron .md files, optionally filtered by region.

        Returns list of file paths.
        """
        paths = []

        if region:
            region_dir = self._region_path(region)
            paths = sorted(region_dir.glob("*.md"))
        else:
            for r, region_dir in Paths.REGIONS.items():
                paths.extend(sorted(region_dir.glob("*.md")))

        # Exclude .gitkeep and non-neuron files
        return [p for p in paths if p.name != ".gitkeep"]

    def count_neurons(self) -> dict[str, int]:
        """Count neurons per region. Returns {region: count}."""
        counts = {}
        for region in Paths.REGIONS:
            counts[region] = len(self.list_neurons(region))
        counts["total"] = sum(counts.values())
        return counts

    def search_by_tag(self, tag: str) -> list[Neuron]:
        """Find all neurons with a specific tag."""
        results = []
        for filepath in self.list_neurons():
            neuron = self.read_neuron(filepath)
            if hasattr(neuron, "tags") and tag in neuron.tags:
                results.append(neuron)
        return results

    def search_by_title(self, query: str) -> list[Neuron]:
        """Find neurons whose title contains the query string (case-insensitive)."""
        query_lower = query.lower()
        results = []
        for filepath in self.list_neurons():
            neuron = self.read_neuron(filepath)
            if query_lower in neuron.title.lower():
                results.append(neuron)
        return results

    # ── Inbox ────────────────────────────────────────────

    def list_inbox(self) -> list[Path]:
        """List all files waiting in _raw-logs/inbox/ for ingestion."""
        inbox = Paths.INBOX
        return sorted(
            p for p in inbox.iterdir()
            if p.is_file() and p.name != ".gitkeep"
        )

    def inbox_count(self) -> int:
        """Count pending files in inbox."""
        return len(self.list_inbox())

    # ── ID Counter Sync ──────────────────────────────────

    def sync_id_counters(self):
        """Scan vault to find the highest existing IDs and set counters.

        Call this on startup to avoid ID collisions.
        """
        max_atomic = 0
        max_memory = 0

        for filepath in self.list_neurons():
            name = filepath.stem  # e.g. NRN-20260512-0042
            if name.startswith("NRN-"):
                try:
                    counter = int(name.split("-")[-1])
                    max_atomic = max(max_atomic, counter)
                except (ValueError, IndexError):
                    pass
            elif name.startswith("MEM-"):
                try:
                    counter = int(name.split("-")[-1])
                    max_memory = max(max_memory, counter)
                except (ValueError, IndexError):
                    pass

        IDGenerator.set_counters(atomic=max_atomic, memory=max_memory)
        logger.info(f"ID counters synced: atomic={max_atomic}, memory={max_memory}")
