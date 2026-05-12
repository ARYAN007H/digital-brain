"""
PocketBase auto-schema setup.

Creates all required collections via REST API on first run.
Idempotent — safe to re-run without destroying data.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import requests

from brain.config import API

logger = logging.getLogger(__name__)

# ── Collection Schemas ───────────────────────────────────

COLLECTIONS = [
    {
        "name": "synapse_scores",
        "type": "base",
        "schema": [
            {"name": "source_id", "type": "text", "required": True,
             "options": {"min": 1, "max": 30}},
            {"name": "target_id", "type": "text", "required": True,
             "options": {"min": 1, "max": 30}},
            {"name": "strength", "type": "number", "required": True,
             "options": {"min": 0}},
            {"name": "last_reinforced", "type": "text", "required": False,
             "options": {"max": 30}},
            {"name": "reinforcement_log", "type": "json", "required": False},
            {"name": "supabase_synced", "type": "bool", "required": False},
        ],
    },
    {
        "name": "emotional_tags",
        "type": "base",
        "schema": [
            {"name": "neuron_id", "type": "text", "required": True,
             "options": {"min": 1, "max": 30}},
            {"name": "tone", "type": "select", "required": True,
             "options": {"values": ["neutral", "positive", "negative", "charged"]}},
            {"name": "urgency", "type": "select", "required": True,
             "options": {"values": ["low", "medium", "high", "critical"]}},
            {"name": "flagged", "type": "bool", "required": False},
        ],
    },
    {
        "name": "pattern_signals",
        "type": "base",
        "schema": [
            {"name": "pattern_type", "type": "text", "required": True,
             "options": {"max": 100}},
            {"name": "description", "type": "text", "required": True,
             "options": {"max": 1000}},
            {"name": "neuron_ids", "type": "json", "required": False},
            {"name": "confidence", "type": "number", "required": False,
             "options": {"min": 0, "max": 1}},
            {"name": "detected_at", "type": "text", "required": False},
        ],
    },
    {
        "name": "task_events",
        "type": "base",
        "schema": [
            {"name": "neuron_id", "type": "text", "required": True,
             "options": {"min": 1, "max": 30}},
            {"name": "action", "type": "text", "required": True,
             "options": {"max": 500}},
            {"name": "status", "type": "select", "required": True,
             "options": {"values": ["pending", "in_progress", "done", "cancelled"]}},
            {"name": "due_date", "type": "text", "required": False,
             "options": {"max": 20}},
        ],
    },
]


class PocketBaseSetup:
    """Auto-create PocketBase collections."""

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = (base_url or API.POCKETBASE_URL).rstrip("/")
        self._admin_token: Optional[str] = None

    def _is_available(self) -> bool:
        """Check if PocketBase is running."""
        try:
            r = requests.get(f"{self._base_url}/api/health", timeout=3)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def _get_existing_collections(self) -> set[str]:
        """Get names of existing collections."""
        try:
            r = requests.get(
                f"{self._base_url}/api/collections",
                params={"perPage": 100},
                timeout=5,
            )
            if r.status_code == 200:
                return {c["name"] for c in r.json().get("items", [])}
            # Try without auth (some versions allow it)
            if r.status_code == 403:
                logger.warning(
                    "PocketBase requires admin auth. "
                    "Create collections manually or set up admin auth first."
                )
        except Exception as e:
            logger.warning(f"Failed to list collections: {e}")
        return set()

    def _create_collection(self, spec: dict) -> bool:
        """Create a single collection."""
        try:
            r = requests.post(
                f"{self._base_url}/api/collections",
                json={
                    "name": spec["name"],
                    "type": spec.get("type", "base"),
                    "schema": spec["schema"],
                },
                timeout=10,
            )
            if r.status_code in (200, 201):
                logger.info(f"  ✅ Created collection: {spec['name']}")
                return True
            elif r.status_code == 400:
                data = r.json()
                if "already exists" in str(data).lower():
                    logger.info(f"  ⏭ Collection already exists: {spec['name']}")
                    return True
                logger.error(f"  ❌ Failed to create {spec['name']}: {data}")
            elif r.status_code == 403:
                logger.warning(
                    f"  🔒 Auth required for {spec['name']}. "
                    "Please create via PocketBase admin UI at "
                    f"{self._base_url}/_/"
                )
            else:
                logger.error(f"  ❌ HTTP {r.status_code} creating {spec['name']}")
            return False
        except Exception as e:
            logger.error(f"  ❌ Error creating {spec['name']}: {e}")
            return False

    def setup(self) -> dict:
        """Create all required collections. Returns stats."""
        if not self._is_available():
            logger.error(
                f"PocketBase not running at {self._base_url}. "
                "Start it first: ./scripts/start.sh"
            )
            return {"status": "error", "reason": "pocketbase not running"}

        existing = self._get_existing_collections()
        stats = {"created": 0, "skipped": 0, "failed": 0}

        logger.info("Setting up PocketBase collections...")

        for spec in COLLECTIONS:
            if spec["name"] in existing:
                logger.info(f"  ⏭ Already exists: {spec['name']}")
                stats["skipped"] += 1
            else:
                if self._create_collection(spec):
                    stats["created"] += 1
                else:
                    stats["failed"] += 1

        return stats

    def print_manual_instructions(self):
        """Print manual setup instructions if auto-setup fails."""
        print(f"\n🔧 Manual PocketBase Setup")
        print(f"   Open: {self._base_url}/_/")
        print(f"\n   Create these collections:\n")
        for spec in COLLECTIONS:
            print(f"   📋 {spec['name']}")
            for field in spec["schema"]:
                ftype = field["type"]
                req = " (required)" if field.get("required") else ""
                print(f"      • {field['name']}: {ftype}{req}")
            print()


# ── CLI Entry Point ──────────────────────────────────────

def main():
    """Run PocketBase setup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    setup = PocketBaseSetup()
    stats = setup.setup()

    if stats.get("status") == "error":
        setup.print_manual_instructions()
        sys.exit(1)

    print(f"\n✅ PocketBase setup: {stats}")
    if stats["failed"] > 0:
        setup.print_manual_instructions()


if __name__ == "__main__":
    main()
