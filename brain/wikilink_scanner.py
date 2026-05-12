"""
Wikilink synapse detection.

Scans all neuron .md files for [[NRN-*]] and [[MEM-*]] wikilinks
and reinforces synapses with +2 score per link found.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from brain.config import Brain, Paths
from brain.synapses import SynapseManager
from brain.vault import Vault

logger = logging.getLogger(__name__)

# Match [[NRN-20260512-0001]] or [[MEM-20260512-001]] patterns
WIKILINK_PATTERN = re.compile(r"\[\[(NRN-\d{8}-\d{3,4}|MEM-\d{8}-\d{3})\]\]")


class WikilinkScanner:
    """Scans vault for wikilinks and creates/reinforces synapses."""

    def __init__(
        self,
        vault: Optional[Vault] = None,
        synapses: Optional[SynapseManager] = None,
    ):
        self.vault = vault or Vault()
        self.synapses = synapses or SynapseManager()

    def scan_file(self, filepath: Path) -> list[tuple[str, str]]:
        """Scan a single neuron file for wikilinks.

        Returns list of (source_id, target_id) pairs found.
        """
        if not filepath.exists():
            return []

        content = filepath.read_text(encoding="utf-8", errors="ignore")
        source_id = filepath.stem  # e.g. NRN-20260512-0001

        # Don't process files that aren't neurons
        if not (source_id.startswith("NRN-") or source_id.startswith("MEM-")):
            return []

        links = WIKILINK_PATTERN.findall(content)
        pairs = []

        for target_id in links:
            if target_id != source_id:  # no self-links
                pairs.append((source_id, target_id))

        return pairs

    def scan_and_reinforce(self, filepath: Path) -> int:
        """Scan a file and reinforce synapses for found wikilinks.

        Returns count of synapses reinforced.
        """
        pairs = self.scan_file(filepath)
        count = 0

        for source_id, target_id in pairs:
            self.synapses.reinforce(
                source_id,
                target_id,
                "wikilink",
                Brain.SYNAPSE_WIKILINK,
            )
            count += 1
            logger.debug(f"Wikilink synapse: {source_id} → {target_id}")

        if count > 0:
            logger.info(f"Reinforced {count} wikilink synapses from {filepath.name}")

        return count

    def scan_vault(self) -> dict:
        """Scan entire vault for wikilinks.

        Returns stats dict.
        """
        total_files = 0
        total_links = 0
        files_with_links = 0

        for filepath in self.vault.list_neurons():
            total_files += 1
            count = self.scan_and_reinforce(filepath)
            if count > 0:
                files_with_links += 1
                total_links += count

        stats = {
            "files_scanned": total_files,
            "files_with_links": files_with_links,
            "synapses_reinforced": total_links,
        }
        logger.info(f"Vault wikilink scan: {stats}")
        return stats

    def scan_neurons(self, neuron_ids: list[str]) -> int:
        """Scan specific neurons by ID for wikilinks.

        Used after ingestion to check newly created neurons.
        """
        count = 0
        for nid in neuron_ids:
            # Search all regions for the neuron file
            for region_path in Paths.REGIONS.values():
                filepath = region_path / f"{nid}.md"
                if filepath.exists():
                    count += self.scan_and_reinforce(filepath)
                    break
        return count


# ── CLI Entry Point ──────────────────────────────────────

def main():
    """Run wikilink scan manually."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    scanner = WikilinkScanner()
    stats = scanner.scan_vault()
    print(f"Scanned {stats['files_scanned']} files")
    print(f"Found links in {stats['files_with_links']} files")
    print(f"Reinforced {stats['synapses_reinforced']} synapses")


if __name__ == "__main__":
    main()
