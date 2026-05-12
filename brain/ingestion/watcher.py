"""
Watchdog file watcher daemon.

Monitors vault/_raw-logs/inbox/ for new files and triggers
the ingestion pipeline. Runs as an always-on daemon (~30MB RAM).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from brain.config import Paths

logger = logging.getLogger(__name__)


class InboxHandler(FileSystemEventHandler):
    """Handles new files dropped into the inbox."""

    # Wait for file to stabilize before processing (avoid partial writes)
    STABILITY_SECONDS = 2

    def __init__(self, process_callback):
        super().__init__()
        self._process = process_callback
        self._pending: dict[str, float] = {}

    def on_created(self, event):
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if filepath.name.startswith("."):
            return

        logger.info(f"New file detected: {filepath.name}")
        self._pending[str(filepath)] = time.time()

    def on_modified(self, event):
        if event.is_directory:
            return
        # Update timestamp for stability check
        filepath = str(event.src_path)
        if filepath in self._pending:
            self._pending[filepath] = time.time()

    def check_pending(self):
        """Process files that have been stable for STABILITY_SECONDS."""
        now = time.time()
        ready = [
            fp for fp, ts in self._pending.items()
            if now - ts >= self.STABILITY_SECONDS
        ]

        for fp in ready:
            filepath = Path(fp)
            del self._pending[fp]

            if not filepath.exists():
                continue

            try:
                self._process(filepath)
            except Exception as e:
                logger.error(f"Failed to process {filepath.name}: {e}")


class InboxWatcher:
    """Watches the inbox directory for new files."""

    def __init__(self, inbox_path: Optional[Path] = None, process_callback=None):
        self._inbox = inbox_path or Paths.INBOX
        self._inbox.mkdir(parents=True, exist_ok=True)

        if process_callback is None:
            process_callback = self._default_process

        self._handler = InboxHandler(process_callback)
        self._observer = Observer()

    @staticmethod
    def _default_process(filepath: Path):
        """Default processing: run the full ingestion pipeline."""
        from brain.ingestion.processor import Processor
        processor = Processor()
        result = processor.process_file(filepath)
        logger.info(f"Processed {filepath.name}: {result}")

    def start(self):
        """Start watching the inbox (non-blocking)."""
        self._observer.schedule(self._handler, str(self._inbox), recursive=False)
        self._observer.start()
        logger.info(f"Watching inbox: {self._inbox}")

    def run_forever(self):
        """Start watching and block until interrupted."""
        self.start()
        try:
            while True:
                self._handler.check_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the watcher."""
        self._observer.stop()
        self._observer.join()
        logger.info("Inbox watcher stopped")

    def process_existing(self):
        """Process any files already in the inbox."""
        for filepath in sorted(self._inbox.iterdir()):
            if filepath.is_file() and not filepath.name.startswith("."):
                try:
                    self._handler._process(filepath)
                except Exception as e:
                    logger.error(f"Failed to process {filepath.name}: {e}")


# ── CLI Entry Point ──────────────────────────────────────

def main():
    """Run the inbox watcher as a standalone daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    watcher = InboxWatcher()

    # Process any existing files first
    watcher.process_existing()

    # Then watch for new ones
    logger.info("Starting inbox watcher daemon...")
    watcher.run_forever()


if __name__ == "__main__":
    main()
