"""
Supabase cloud mirror — background sync worker.

Pushes local data to Supabase free tier every 5 minutes.
Optional: brain works 100% offline without Supabase.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

from brain.config import API, Brain
from brain.queue import WriteQueue

logger = logging.getLogger(__name__)


class SupabaseMirror:
    """Background worker that syncs local data to Supabase."""

    def __init__(self, queue: Optional[WriteQueue] = None):
        self.queue = queue or WriteQueue()
        self._client = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _get_client(self):
        """Lazy-init Supabase client."""
        if self._client is None:
            if not API.has_supabase():
                return None
            from supabase import create_client
            self._client = create_client(API.SUPABASE_URL, API.SUPABASE_KEY)
        return self._client

    def sync_once(self) -> dict:
        """Run one sync cycle. Returns stats."""
        client = self._get_client()
        if client is None:
            return {"status": "skipped", "reason": "supabase not configured"}

        stats = {"synced": 0, "failed": 0, "skipped": 0}
        items = self.queue.dequeue("supabase", batch_size=50)

        for item in items:
            try:
                self._push_item(client, item)
                self.queue.mark_done(item["id"])
                stats["synced"] += 1
            except Exception as e:
                self.queue.mark_failed(item["id"], str(e))
                stats["failed"] += 1
                logger.warning(f"Sync failed for item {item['id']}: {e}")

        if stats["synced"] > 0:
            logger.info(f"Supabase sync: {stats}")
        return stats

    def _push_item(self, client, item: dict):
        """Push a single queue item to Supabase."""
        payload = item["payload"]
        item_type = payload.get("type", "unknown")

        if item_type == "neuron":
            region = payload.get("region", "hippocampus")
            table = f"neurons_{region}"
            client.table(table).upsert({
                "id": payload["id"],
                "region": region,
                "content": payload.get("data", ""),
            }).execute()

        elif item_type == "synapse":
            client.table("synapses").upsert({
                "source_id": payload["source_id"],
                "target_id": payload["target_id"],
                "strength": payload["strength"],
                "reinforcement_log": json.dumps(
                    payload.get("reinforcement_log", [])
                ),
            }).execute()

    def start(self, interval: int = None):
        """Start background sync thread."""
        if self._running:
            return

        interval = interval or Brain.SYNC_INTERVAL
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(interval,),
            daemon=True,
            name="supabase-sync",
        )
        self._thread.start()
        logger.info(f"Supabase sync started (every {interval}s)")

    def _run_loop(self, interval: int):
        """Background sync loop."""
        while self._running:
            try:
                self.sync_once()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
            time.sleep(interval)

    def stop(self):
        """Stop the background sync."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Supabase sync stopped")
