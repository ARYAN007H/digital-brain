"""
Antigravity IDE auto-ingestion connector.

Watches ~/.gemini/antigravity/brain/ for conversation logs
and auto-ingests them as memory + atomic neurons.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from brain.config import Paths
from brain.models import AtomicNeuron, MemoryNeuron, NeuronSource
from brain.vault import Vault

logger = logging.getLogger(__name__)

ANTIGRAVITY_DIR = Path.home() / ".gemini" / "antigravity" / "brain"


class IDEConversationParser:
    """Parses Antigravity IDE conversation logs."""

    @staticmethod
    def parse_overview(filepath: Path) -> dict:
        """Parse overview.txt → {title, themes, total_exchanges, raw_content}."""
        if not filepath.exists():
            return {}
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        lines = content.strip().split("\n")

        user_msgs = [l[5:].strip() for l in lines if l.strip().startswith("USER:")]
        title = user_msgs[0][:100] if user_msgs else "IDE Session"

        code_patterns = {
            "debugging": ["error", "fix", "bug", "fail"],
            "architecture": ["refactor", "design", "pattern"],
            "feature": ["add", "implement", "create", "build"],
            "optimization": ["optimize", "performance", "speed"],
        }
        themes = []
        cl = content.lower()
        for theme, kws in code_patterns.items():
            if any(k in cl for k in kws):
                themes.append(theme)

        return {
            "title": title,
            "themes": themes,
            "total_exchanges": len(user_msgs),
            "raw_content": content,
        }

    @staticmethod
    def extract_decisions(content: str) -> list[str]:
        """Extract technical decisions from conversation."""
        decisions = []
        patterns = [
            r"(?:decided|choosing|going with|will use|using)\s+(.{10,80})",
            r"(?:the fix|solution|approach)\s+(?:is|was)\s+(.{10,80})",
        ]
        for p in patterns:
            decisions.extend(re.findall(p, content, re.IGNORECASE)[:3])
        return decisions[:5]


class IDEHandler(FileSystemEventHandler):
    """Handles new conversation logs from Antigravity IDE."""

    STABILITY_SECONDS = 5

    def __init__(self, vault: Optional[Vault] = None):
        super().__init__()
        self.vault = vault or Vault()
        self.parser = IDEConversationParser()
        self._pending: dict[str, float] = {}
        self._processed: set[str] = set()

    def on_created(self, event):
        if event.is_directory:
            return
        fp = Path(event.src_path)
        if fp.name == "overview.txt":
            self._pending[str(fp)] = time.time()

    def on_modified(self, event):
        if event.is_directory:
            return
        fp = str(event.src_path)
        if fp in self._pending:
            self._pending[fp] = time.time()

    def check_pending(self):
        now = time.time()
        ready = [f for f, t in self._pending.items() if now - t >= self.STABILITY_SECONDS]
        for fp in ready:
            path = Path(fp)
            del self._pending[fp]
            if str(path) in self._processed or not path.exists():
                continue
            try:
                self._ingest_conversation(path)
                self._processed.add(str(path))
            except Exception as e:
                logger.error(f"IDE ingest failed: {e}")

    def _ingest_conversation(self, overview_path: Path):
        parsed = self.parser.parse_overview(overview_path)
        if not parsed or parsed.get("total_exchanges", 0) < 2:
            return

        memory = MemoryNeuron(
            title=f"IDE: {parsed['title']}",
            source=NeuronSource.ANTIGRAVITY_IDE.value,
            raw_log_path=str(overview_path),
            key_themes=parsed.get("themes", []),
            body=f"IDE session: {parsed['title']}. "
                 f"{parsed['total_exchanges']} exchanges. "
                 f"Themes: {', '.join(parsed.get('themes', ['general']))}",
        )
        self.vault.write_neuron(memory)

        decisions = self.parser.extract_decisions(parsed.get("raw_content", ""))
        for d in decisions:
            n = AtomicNeuron(
                region="prefrontal", title=d[:100],
                source=NeuronSource.ANTIGRAVITY_IDE.value,
                tags=["ide-decision"], body=d,
            )
            self.vault.write_neuron(n)
            memory.atomic_children.append(n.id)

        if memory.atomic_children:
            self.vault.write_neuron(memory)

        logger.info(f"IDE ingested: {memory.id}, decisions={len(decisions)}")


class IDEWatcher:
    """Watches Antigravity IDE directory for new conversations."""

    def __init__(self, ide_dir: Optional[Path] = None, vault: Optional[Vault] = None):
        self._ide_dir = ide_dir or ANTIGRAVITY_DIR
        self._handler = IDEHandler(vault)
        self._observer = Observer()

    def start(self):
        if not self._ide_dir.exists():
            logger.warning(f"IDE dir not found: {self._ide_dir}")
            return
        self._observer.schedule(self._handler, str(self._ide_dir), recursive=True)
        self._observer.start()
        logger.info(f"Watching IDE: {self._ide_dir}")

    def run_forever(self):
        self.start()
        try:
            while True:
                self._handler.check_pending()
                time.sleep(2)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._observer.stop()
        self._observer.join()

    def ingest_existing(self, since_days: int = 7):
        """Ingest recent existing conversations."""
        if not self._ide_dir.exists():
            return
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=since_days)
        count = 0
        for d in sorted(self._ide_dir.iterdir()):
            if not d.is_dir():
                continue
            for ov in [d / ".system_generated" / "logs" / "overview.txt", d / "overview.txt"]:
                if ov.exists() and datetime.fromtimestamp(ov.stat().st_mtime) >= cutoff:
                    try:
                        self._handler._ingest_conversation(ov)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed: {d.name}: {e}")
                    break
        logger.info(f"Ingested {count} IDE conversations")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    IDEWatcher().run_forever()
