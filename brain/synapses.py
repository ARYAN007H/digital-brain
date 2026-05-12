"""
PocketBase synapse manager.

Tracks connection strength between neurons via PocketBase REST API.
All writes queue to SQLite first (PocketBase is local, so fast anyway).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests

from brain.config import API, Brain
from brain.models import Synapse

logger = logging.getLogger(__name__)


class SynapseManager:
    """PocketBase REST client for synapse scores and metadata."""

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = (base_url or API.POCKETBASE_URL).rstrip("/")
        self._api = f"{self._base_url}/api/collections"

    def _url(self, collection: str, record_id: str = "") -> str:
        """Build PocketBase API URL."""
        url = f"{self._api}/{collection}/records"
        if record_id:
            url += f"/{record_id}"
        return url

    def _is_available(self) -> bool:
        """Check if PocketBase is running."""
        try:
            r = requests.get(f"{self._base_url}/api/health", timeout=2)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    # ── Synapse Operations ───────────────────────────────

    def _find_synapse(self, source_id: str, target_id: str) -> Optional[dict]:
        """Find existing synapse between two neurons (bidirectional)."""
        try:
            # Check both directions
            for s, t in [(source_id, target_id), (target_id, source_id)]:
                r = requests.get(
                    self._url("synapse_scores"),
                    params={"filter": f'source_id="{s}" && target_id="{t}"'},
                    timeout=5,
                )
                if r.status_code == 200:
                    items = r.json().get("items", [])
                    if items:
                        return items[0]
        except Exception as e:
            logger.warning(f"PocketBase lookup failed: {e}")
        return None

    def reinforce(self, source_id: str, target_id: str, event: str, score: int) -> Synapse:
        """Create or strengthen a synapse between two neurons.

        Events: co-access, wikilink, semantic-sim, ai-confirmed, user-confirmed
        """
        existing = self._find_synapse(source_id, target_id)

        if existing:
            # Update existing synapse
            log = existing.get("reinforcement_log", [])
            if isinstance(log, str):
                log = json.loads(log) if log else []
            log.append(event)

            synapse = Synapse(
                source_id=existing["source_id"],
                target_id=existing["target_id"],
                strength=existing["strength"] + score,
                reinforcement_log=log,
            )
            synapse.reinforce(event, 0)  # sets last_reinforced + supabase_synced

            try:
                requests.patch(
                    self._url("synapse_scores", existing["id"]),
                    json={
                        "strength": synapse.strength,
                        "last_reinforced": synapse.last_reinforced,
                        "reinforcement_log": json.dumps(synapse.reinforcement_log),
                        "supabase_synced": False,
                    },
                    timeout=5,
                )
            except Exception as e:
                logger.warning(f"PocketBase update failed: {e}")

            return synapse
        else:
            # Create new synapse
            synapse = Synapse(source_id=source_id, target_id=target_id)
            synapse.reinforce(event, score)

            try:
                requests.post(
                    self._url("synapse_scores"),
                    json={
                        "source_id": synapse.source_id,
                        "target_id": synapse.target_id,
                        "strength": synapse.strength,
                        "last_reinforced": synapse.last_reinforced,
                        "reinforcement_log": json.dumps(synapse.reinforcement_log),
                        "supabase_synced": False,
                    },
                    timeout=5,
                )
            except Exception as e:
                logger.warning(f"PocketBase create failed: {e}")

            return synapse

    def get_connections(self, neuron_id: str) -> list[Synapse]:
        """Get all synapses involving a neuron."""
        synapses = []
        try:
            r = requests.get(
                self._url("synapse_scores"),
                params={
                    "filter": f'source_id="{neuron_id}" || target_id="{neuron_id}"',
                    "perPage": 100,
                },
                timeout=5,
            )
            if r.status_code == 200:
                for item in r.json().get("items", []):
                    log = item.get("reinforcement_log", "[]")
                    if isinstance(log, str):
                        log = json.loads(log)
                    synapses.append(Synapse(
                        source_id=item["source_id"],
                        target_id=item["target_id"],
                        strength=item["strength"],
                        last_reinforced=item.get("last_reinforced", ""),
                        reinforcement_log=log,
                        supabase_synced=item.get("supabase_synced", False),
                    ))
        except Exception as e:
            logger.warning(f"PocketBase query failed: {e}")
        return synapses

    def top_synapses(self, limit: int = 10) -> list[Synapse]:
        """Get the strongest synapses across the brain."""
        synapses = []
        try:
            r = requests.get(
                self._url("synapse_scores"),
                params={"sort": "-strength", "perPage": limit},
                timeout=5,
            )
            if r.status_code == 200:
                for item in r.json().get("items", []):
                    log = item.get("reinforcement_log", "[]")
                    if isinstance(log, str):
                        log = json.loads(log)
                    synapses.append(Synapse(
                        source_id=item["source_id"],
                        target_id=item["target_id"],
                        strength=item["strength"],
                        last_reinforced=item.get("last_reinforced", ""),
                        reinforcement_log=log,
                    ))
        except Exception as e:
            logger.warning(f"PocketBase query failed: {e}")
        return synapses

    def total_count(self) -> int:
        """Total number of synapses."""
        try:
            r = requests.get(
                self._url("synapse_scores"),
                params={"perPage": 1},
                timeout=5,
            )
            if r.status_code == 200:
                return r.json().get("totalItems", 0)
        except Exception:
            pass
        return 0

    # ── Emotional Tags ───────────────────────────────────

    def set_emotional_tag(self, neuron_id: str, tone: str, urgency: str):
        """Store emotional metadata for a neuron (amygdala region)."""
        try:
            requests.post(
                self._url("emotional_tags"),
                json={
                    "neuron_id": neuron_id,
                    "tone": tone,
                    "urgency": urgency,
                    "flagged": urgency in ("high", "critical"),
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Failed to set emotional tag: {e}")

    # ── Task Events ──────────────────────────────────────

    def create_task_event(self, neuron_id: str, action: str, due_date: str = ""):
        """Create a task event in executive tracking."""
        try:
            requests.post(
                self._url("task_events"),
                json={
                    "neuron_id": neuron_id,
                    "action": action,
                    "status": "pending",
                    "due_date": due_date,
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Failed to create task event: {e}")
