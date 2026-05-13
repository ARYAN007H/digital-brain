"""
LLM-powered ingestion processor.

Takes chunked content, sends it through qwen2.5:3b to:
1. Detect cortex region
2. Extract atomic ideas → AtomicNeuron
3. Summarize session → MemoryNeuron
4. Extract tasks → executive neurons
5. Detect emotional tone → amygdala metadata
6. Generate embeddings → ChromaDB
7. Suggest synapse candidates → PocketBase
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Optional

from brain.config import Brain, Paths
from brain.dedup import DedupEngine
from brain.ingestion.parser import chunk_content, detect_source, extract_text
from brain.models import AtomicNeuron, MemoryNeuron
from brain.queue import WriteQueue
from brain.router import Router
from brain.sync.git_sync import GitSync
from brain.synapses import SynapseManager
from brain.vault import Vault
from brain.vectors import VectorStore
from brain.wikilink_scanner import WikilinkScanner

logger = logging.getLogger(__name__)

# ── LLM Prompts ──────────────────────────────────────────

REGION_DETECT_PROMPT = """Classify this text into exactly ONE brain region:
- prefrontal: planning, decisions, reasoning, goals
- hippocampus: facts, memories, knowledge, chat summaries
- creative: ideas, brainstorms, what-ifs, sparks
- predictive: patterns, trends, forecasts, signals
- amygdala: emotional content, urgency, core values
- executive: tasks, projects, commitments, action items

Text: {text}

Reply with ONLY the region name (one word, lowercase)."""

ATOMIC_EXTRACT_PROMPT = """Extract atomic ideas from this text. Each idea should be:
- One single concept
- 2-5 sentences
- Self-contained

Text: {text}

Output as JSON array of objects: [{{"title": "...", "body": "...", "tags": ["..."]}}]
Return ONLY valid JSON, no other text."""

SUMMARY_PROMPT = """Summarize this content in 3-7 sentences.
Focus on: what happened, what was decided, what was learned.

Text: {text}

Key themes (as comma-separated list):"""

EMOTION_DETECT_PROMPT = """Analyze the emotional tone and urgency of this text.
Reply as JSON: {{"tone": "neutral|positive|negative|charged", "urgency": "low|medium|high|critical"}}
Return ONLY valid JSON.

Text: {text}"""

TASK_EXTRACT_PROMPT = """Extract actionable tasks from this text.
Reply as JSON array: [{{"action": "...", "due_date": ""}}]
Return ONLY valid JSON. If no tasks, return [].

Text: {text}"""


class Processor:
    """LLM-powered ingestion processor."""

    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
    MAX_TEXT_CHARS = 500_000
    MAX_CHUNKS = 500

    def __init__(
        self,
        router: Optional[Router] = None,
        vault: Optional[Vault] = None,
        vectors: Optional[VectorStore] = None,
        synapses: Optional[SynapseManager] = None,
        queue: Optional[WriteQueue] = None,
        git: Optional[GitSync] = None,
        dedup: Optional[DedupEngine] = None,
        wikilinks: Optional[WikilinkScanner] = None,
    ):
        self.router = router or Router()
        self.vault = vault or Vault()
        self.vectors = vectors or VectorStore()
        self.synapses = synapses or SynapseManager()
        self.queue = queue or WriteQueue()
        self.git = git or GitSync()
        self.dedup = dedup or DedupEngine()
        self.wikilinks = wikilinks or WikilinkScanner(vault=self.vault, synapses=self.synapses)
        self._metrics_file = Paths.META / "ingestion-health.json"

        # Neural subsystems (lazy init)
        self._emotional_gate = None
        self._associative = None
        self._bus = None
        try:
            from brain.eventbus import EventBus
            from brain.emotional_gate import EmotionalGate
            from brain.associative import AssociativeRecallEngine
            self._bus = EventBus.get()
            self._emotional_gate = EmotionalGate()
            self._associative = AssociativeRecallEngine(vault=self.vault)
        except Exception:
            pass

    def _load_metrics(self) -> dict:
        if self._metrics_file.exists():
            try:
                return json.loads(self._metrics_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"rejected": 0, "quarantined": 0, "reasons": {}}

    def _save_metrics(self, metrics: dict):
        self.vault.write_meta("ingestion-health.json", json.dumps(metrics, indent=2))

    def _inc_metric(self, bucket: str, reason: str):
        metrics = self._load_metrics()
        metrics[bucket] = metrics.get(bucket, 0) + 1
        reasons = metrics.setdefault("reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1
        self._save_metrics(metrics)

    def _quarantine(self, filepath: Path, reason: str):
        quarantine_dir = Paths.RAW_LOGS / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / filepath.name
        shutil.move(str(filepath), str(target))
        logger.warning(
            "File quarantined",
            extra={"event": "file_quarantined", "file": str(filepath), "target": str(target), "reason": reason},
        )

    def process_file(self, filepath: Path, force: bool = False) -> dict:
        """Full ingestion pipeline for a single file.

        Args:
            filepath: File to ingest.
            force: If True, bypass dedup and re-ingest.

        Returns summary dict with counts of created neurons.
        """
        logger.info(f"Processing: {filepath.name}")

        size_bytes = filepath.stat().st_size
        if size_bytes > self.MAX_FILE_SIZE_BYTES:
            self._inc_metric("rejected", "max_file_size_exceeded")
            self._quarantine(filepath, "max_file_size_exceeded")
            return {"status": "rejected", "reason": "max_file_size_exceeded", "size_bytes": size_bytes}

        # Deduplication check
        is_new, content_hash = self.dedup.check_and_record(filepath, force=force)
        if not is_new:
            return {"status": "skipped", "reason": "duplicate"}

        source = detect_source(filepath)
        extraction = extract_text(filepath)
        if extraction.status != "ok":
            reason = extraction.reason or "extract_failed"
            self._inc_metric("quarantined", reason)
            self._quarantine(filepath, reason)
            logger.warning(
                "Extraction failed",
                extra={
                    "event": "extract_failure",
                    "file": str(filepath),
                    "status": extraction.status,
                    "reason": reason,
                    "exit_cause": extraction.exit_cause,
                },
            )
            return {"status": "quarantined", "reason": reason, "exit_cause": extraction.exit_cause}

        text = extraction.text
        if len(text) > self.MAX_TEXT_CHARS:
            self._inc_metric("rejected", "max_text_chars_exceeded")
            self._quarantine(filepath, "max_text_chars_exceeded")
            return {"status": "rejected", "reason": "max_text_chars_exceeded", "text_chars": len(text)}

        if not text.strip():
            logger.warning(f"Empty file, skipping: {filepath}")
            return {"status": "skipped", "reason": "empty"}

        chunks = chunk_content(text)
        if len(chunks) > self.MAX_CHUNKS:
            self._inc_metric("rejected", "max_chunks_exceeded")
            self._quarantine(filepath, "max_chunks_exceeded")
            return {"status": "rejected", "reason": "max_chunks_exceeded", "chunks": len(chunks)}
        all_atomic: list[AtomicNeuron] = []
        all_tasks: list[dict] = []

        # Process each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i+1}/{len(chunks)}")

            # 1. Detect region
            region = self._detect_region(chunk)

            # 2. Extract atomic ideas
            atoms = self._extract_atoms(chunk, region, source)
            all_atomic.extend(atoms)

            # 3. Extract tasks
            tasks = self._extract_tasks(chunk)
            all_tasks.extend(tasks)

            # 4. Detect emotion (for amygdala metadata)
            self._detect_emotion(chunk, [a.id for a in atoms])

        # 5. Create memory neuron (session summary)
        memory = self._create_memory(text, source, filepath, all_atomic)

        # 6. Generate embeddings and store vectors
        self._store_vectors(all_atomic, memory)

        # 7. Suggest synapse candidates
        self._suggest_synapses(all_atomic)

        # 8. Create executive neurons from tasks
        task_neurons = self._create_task_neurons(all_tasks, source)

        # 9. Scan for wikilinks in new neurons
        new_ids = [n.id for n in all_atomic + task_neurons]
        if memory:
            new_ids.append(memory.id)
        self.wikilinks.scan_neurons(new_ids)

        # 10. Queue cloud sync
        self._queue_sync(all_atomic, memory, task_neurons)

        # 11. Record neuron IDs in dedup table
        self.dedup.record_neurons(content_hash, new_ids)

        # 12. Git commit
        self.git.auto_commit(source=source)

        result = {
            "status": "success",
            "source": source,
            "atomic_count": len(all_atomic),
            "memory_id": memory.id if memory else None,
            "task_count": len(task_neurons),
            "chunks_processed": len(chunks),
        }
        logger.info(f"Ingestion complete: {result}")
        return result

    def _detect_region(self, text: str) -> str:
        """Use LLM to classify text into a cortex region."""
        prompt = REGION_DETECT_PROMPT.format(text=text[:1500])
        response = self.router.ask_local(prompt).strip().lower()

        valid_regions = {"prefrontal", "hippocampus", "creative",
                         "predictive", "amygdala", "executive"}
        if response in valid_regions:
            return response

        # Fallback: hippocampus (general memory)
        return "hippocampus"

    def _extract_atoms(self, text: str, region: str, source: str) -> list[AtomicNeuron]:
        """Extract atomic ideas from a chunk."""
        prompt = ATOMIC_EXTRACT_PROMPT.format(text=text[:2000])
        response = self.router.ask_local(prompt)

        atoms = []
        try:
            ideas = json.loads(self._clean_json(response))
            if not isinstance(ideas, list):
                ideas = [ideas]

            for idea in ideas:
                neuron = AtomicNeuron(
                    region=region,
                    title=idea.get("title", "Untitled idea"),
                    source=source,
                    tags=idea.get("tags", []),
                    body=idea.get("body", ""),
                )
                self.vault.write_neuron(neuron)
                atoms.append(neuron)

                # Index for associative recall
                if self._associative:
                    self._associative.index_neuron(
                        neuron.id, neuron.title, neuron.body, neuron.region,
                        neuron.tags,
                    )
                # Emit neuron.created event
                if self._bus:
                    self._bus.emit("neuron.created", {
                        "neuron_id": neuron.id, "region": region,
                        "title": neuron.title, "source": source,
                    })

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse atomic ideas: {e}")
            # Fallback: create one neuron from the whole chunk
            neuron = AtomicNeuron(
                region=region,
                title=text[:80].replace("\n", " "),
                source=source,
                body=text[:500],
            )
            self.vault.write_neuron(neuron)
            atoms.append(neuron)

        return atoms

    def _create_memory(
        self, text: str, source: str, filepath: Path,
        children: list[AtomicNeuron],
    ) -> Optional[MemoryNeuron]:
        """Create a memory neuron summarizing the session."""
        prompt = SUMMARY_PROMPT.format(text=text[:3000])
        response = self.router.ask_local(prompt)

        # Parse summary and themes
        lines = response.strip().split("\n")
        summary = "\n".join(lines[:-1]) if len(lines) > 1 else response
        themes = []
        if lines:
            last = lines[-1]
            if "," in last:
                themes = [t.strip() for t in last.split(",") if t.strip()]

        memory = MemoryNeuron(
            title=f"Session: {filepath.stem}",
            source=source,
            raw_log_path=str(filepath.relative_to(filepath.parent.parent.parent)),
            atomic_children=[c.id for c in children],
            key_themes=themes,
            body=summary,
        )
        self.vault.write_neuron(memory)
        return memory

    def _detect_emotion(self, text: str, neuron_ids: list[str]):
        """Detect emotional tone and urgency, store in PocketBase."""
        prompt = EMOTION_DETECT_PROMPT.format(text=text[:1000])
        response = self.router.ask_local(prompt)

        try:
            data = json.loads(self._clean_json(response))
            tone = data.get("tone", "neutral")
            urgency = data.get("urgency", "low")

            for nid in neuron_ids:
                self.synapses.set_emotional_tag(nid, tone, urgency)
        except (json.JSONDecodeError, KeyError):
            pass

    def _extract_tasks(self, text: str) -> list[dict]:
        """Extract action items from text."""
        prompt = TASK_EXTRACT_PROMPT.format(text=text[:1500])
        response = self.router.ask_local(prompt)

        try:
            tasks = json.loads(self._clean_json(response))
            return tasks if isinstance(tasks, list) else []
        except json.JSONDecodeError:
            return []

    def _create_task_neurons(self, tasks: list[dict], source: str) -> list[AtomicNeuron]:
        """Create executive neurons from extracted tasks."""
        neurons = []
        for task in tasks:
            action = task.get("action", "").strip()
            if not action:
                continue

            neuron = AtomicNeuron(
                region="executive",
                title=action[:100],
                source=source,
                urgency="medium",
                body=action,
            )
            self.vault.write_neuron(neuron)
            neurons.append(neuron)

            self.synapses.create_task_event(
                neuron.id, action, task.get("due_date", ""),
            )
        return neurons

    def _store_vectors(self, atoms: list[AtomicNeuron], memory: Optional[MemoryNeuron]):
        """Generate embeddings and store in ChromaDB."""
        all_neurons = list(atoms)
        if memory:
            all_neurons.append(memory)

        for neuron in all_neurons:
            if neuron.region == "amygdala":
                continue  # amygdala is metadata-only

            text = f"{neuron.title}\n{neuron.body}"
            embedding = self.router.get_embedding(text)

            if embedding:
                chroma_id = self.vectors.store_neuron(
                    neuron_id=neuron.id,
                    region=neuron.region,
                    embedding=embedding,
                    document=text,
                    metadata={
                        "title": neuron.title,
                        "source": neuron.source,
                        "created": neuron.created,
                    },
                )
                neuron.chroma_id = chroma_id

    def _suggest_synapses(self, atoms: list[AtomicNeuron]):
        """Find and create synapse candidates between new neurons.

        Applies emotional encoding boost for charged/urgent content.
        """
        for neuron in atoms:
            if neuron.region == "amygdala":
                continue

            similar = self.vectors.get_similar(neuron.id, neuron.region)
            for match in similar:
                base_score = Brain.SYNAPSE_SEMANTIC_SIM

                # Emotional encoding boost
                if self._emotional_gate:
                    mult = self._emotional_gate.encoding_multiplier(
                        neuron.emotional_weight, neuron.urgency,
                    )
                    base_score = max(1, int(round(base_score * mult)))

                self.synapses.reinforce(
                    neuron.id,
                    match["id"],
                    "semantic-sim",
                    base_score,
                )

    def _queue_sync(self, atoms, memory, task_neurons):
        """Queue all new data for Supabase sync."""
        for neuron in atoms + task_neurons + ([memory] if memory else []):
            self.queue.enqueue("supabase", {
                "type": "neuron",
                "id": neuron.id,
                "region": neuron.region,
                "data": neuron.to_markdown(),
            })

    @staticmethod
    def _clean_json(text: str) -> str:
        """Extract JSON from LLM response that may contain extra text."""
        # Try to find JSON array or object
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            import re
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group()
        return text
