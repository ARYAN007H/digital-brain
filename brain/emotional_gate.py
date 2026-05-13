"""
Emotional gate — amygdala modulation of encoding and retrieval.

Biological basis: the amygdala modulates memory formation and recall.
Emotionally charged events form stronger memories (flashbulb memory),
and emotional state at retrieval biases which memories surface
(mood-congruent recall).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from brain.config import API, Brain, HTTP

logger = logging.getLogger(__name__)

# Encoding multipliers for emotional content
ENCODING_BOOST = {
    "critical": 3.0,
    "high": 2.0,
    "charged": 2.0,
    "negative": 1.5,
    "medium": 1.2,
}


class EmotionalGate:
    """Modulates synapse scoring and retrieval based on emotional metadata."""

    def __init__(self, pocketbase_url: Optional[str] = None):
        self._base_url = (pocketbase_url or API.POCKETBASE_URL).rstrip("/")
        self._api = f"{self._base_url}/api/collections"
        self._headers = HTTP.build_headers(API.POCKETBASE_AUTH_TOKEN)

    def encoding_multiplier(self, tone: str, urgency: str) -> float:
        """Get synapse score multiplier for encoding phase.

        High urgency / charged tone → stronger initial synapses.
        """
        mult = 1.0
        mult = max(mult, ENCODING_BOOST.get(urgency, 1.0))
        mult = max(mult, ENCODING_BOOST.get(tone, 1.0))
        return mult

    def get_neuron_emotion(self, neuron_id: str) -> dict:
        """Fetch emotional metadata for a neuron from PocketBase."""
        try:
            r = HTTP.request("GET", f"{self._api}/emotional_tags/records",
                             service_name="PocketBase",
                             params={"filter": f'neuron_id="{neuron_id}"', "perPage": "1"},
                             timeout=5, headers=self._headers)
            if r.status_code == 200:
                items = r.json().get("items", [])
                if items:
                    return {"tone": items[0].get("tone", "neutral"),
                            "urgency": items[0].get("urgency", "low"),
                            "flagged": items[0].get("flagged", False)}
        except Exception:
            pass
        return {"tone": "neutral", "urgency": "low", "flagged": False}

    def rerank_by_emotion(self, results: list[dict], query_emotion: dict | None = None) -> list[dict]:
        """Re-rank retrieval results using emotional congruence.

        If the query has emotional signal, boost emotionally-matching neurons.
        Always boost flagged (urgent) neurons.
        """
        if not results:
            return results

        reranked = []
        for r in results:
            nid = r.get("id") or r.get("neuron_id", "")
            item = dict(r)
            boost = 1.0

            emotion = self.get_neuron_emotion(nid)
            if emotion.get("flagged"):
                boost *= 1.5  # urgent items always get boosted

            # Mood-congruent recall
            if query_emotion:
                q_tone = query_emotion.get("tone", "neutral")
                if q_tone != "neutral" and emotion.get("tone") == q_tone:
                    boost *= 1.3

            item["emotional_boost"] = round(boost, 2)
            existing_score = item.get("hybrid_score") or item.get("activation") or (1.0 - item.get("distance", 0.5))
            item["emotionally_modulated_score"] = round(existing_score * boost, 4)
            reranked.append(item)

        reranked.sort(key=lambda x: x.get("emotionally_modulated_score", 0), reverse=True)
        return reranked

    def get_unresolved_critical(self, max_age_hours: int = 48) -> list[dict]:
        """Find critical-urgency neurons not yet addressed.

        These get force-injected into working memory.
        """
        try:
            r = HTTP.request("GET", f"{self._api}/emotional_tags/records",
                             service_name="PocketBase",
                             params={"filter": 'flagged=true', "perPage": "50", "sort": "-created"},
                             timeout=5, headers=self._headers)
            if r.status_code == 200:
                return [{"neuron_id": i["neuron_id"], "tone": i.get("tone", ""),
                         "urgency": i.get("urgency", "")}
                        for i in r.json().get("items", [])]
        except Exception:
            pass
        return []
