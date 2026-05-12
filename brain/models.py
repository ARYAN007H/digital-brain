"""
Neuron and Synapse data models.

Every note = one neuron. Two types:
- AtomicNeuron: one idea, smallest unit
- MemoryNeuron: compressed conversation/session

Synapse = connection between two neurons with a strength score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

import yaml


# ── Enums ─────────────────────────────────────────────────

class Region(str, Enum):
    PREFRONTAL = "prefrontal"
    HIPPOCAMPUS = "hippocampus"
    CREATIVE = "creative"
    PREDICTIVE = "predictive"
    AMYGDALA = "amygdala"
    EXECUTIVE = "executive"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmotionalWeight(str, Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CHARGED = "charged"


class NeuronSource(str, Enum):
    ANTIGRAVITY_IDE = "antigravity-ide"
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    CURSOR = "cursor"
    BROWSER = "browser"
    FILE = "file"
    CODE_REPO = "code-repo"
    VOICE = "voice"


# ── ID Generators ────────────────────────────────────────

class IDGenerator:
    """Thread-safe neuron ID generation."""

    _atomic_counter: int = 0
    _memory_counter: int = 0
    _last_date: str = ""

    @classmethod
    def _reset_if_new_day(cls):
        today = date.today().strftime("%Y%m%d")
        if today != cls._last_date:
            cls._last_date = today
            cls._atomic_counter = 0
            cls._memory_counter = 0

    @classmethod
    def next_atomic(cls) -> str:
        """Generate NRN-{YYYYMMDD}-{4digit} ID."""
        cls._reset_if_new_day()
        cls._atomic_counter += 1
        return f"NRN-{cls._last_date}-{cls._atomic_counter:04d}"

    @classmethod
    def next_memory(cls) -> str:
        """Generate MEM-{YYYYMMDD}-{3digit} ID."""
        cls._reset_if_new_day()
        cls._memory_counter += 1
        return f"MEM-{cls._last_date}-{cls._memory_counter:03d}"

    @classmethod
    def set_counters(cls, atomic: int = 0, memory: int = 0):
        """Set counters explicitly (used when loading existing vault state)."""
        cls._atomic_counter = atomic
        cls._memory_counter = memory


# ── YAML Frontmatter Helpers ─────────────────────────────

def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from a markdown string.

    Returns (frontmatter_dict, body_text).
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not match:
        return {}, content

    fm_str, body = match.groups()
    try:
        fm = yaml.safe_load(fm_str) or {}
    except yaml.YAMLError:
        fm = {}

    return fm, body.strip()


def _render_frontmatter(data: dict, body: str) -> str:
    """Render a dict + body into YAML-frontmatter markdown."""
    # Use block style for cleaner output, default_flow_style for inline lists
    fm = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return f"---\n{fm}---\n\n{body}\n"


# ── Neuron Dataclasses ───────────────────────────────────

@dataclass
class AtomicNeuron:
    """One idea, smallest unit of knowledge."""

    id: str = ""
    type: str = "atomic-note"
    region: str = Region.HIPPOCAMPUS.value
    title: str = ""
    created: str = ""
    source: str = NeuronSource.FILE.value
    tags: list[str] = field(default_factory=list)
    urgency: str = Urgency.LOW.value
    emotional_weight: str = EmotionalWeight.NEUTRAL.value
    chroma_id: str = ""
    supabase_synced: bool = False
    body: str = ""  # The actual content (2-5 sentences)

    def __post_init__(self):
        if not self.id:
            self.id = IDGenerator.next_atomic()
        if not self.created:
            self.created = date.today().isoformat()

    def to_markdown(self) -> str:
        """Serialize to Obsidian markdown with YAML frontmatter."""
        fm = {
            "id": self.id,
            "type": self.type,
            "region": self.region,
            "title": self.title,
            "created": self.created,
            "source": self.source,
            "tags": self.tags,
            "urgency": self.urgency,
            "emotional_weight": self.emotional_weight,
            "chroma_id": self.chroma_id,
            "supabase_synced": self.supabase_synced,
        }
        return _render_frontmatter(fm, self.body)

    @classmethod
    def from_markdown(cls, content: str) -> AtomicNeuron:
        """Deserialize from Obsidian markdown."""
        fm, body = _parse_frontmatter(content)
        return cls(
            id=fm.get("id", ""),
            type=fm.get("type", "atomic-note"),
            region=fm.get("region", Region.HIPPOCAMPUS.value),
            title=fm.get("title", ""),
            created=str(fm.get("created", "")),
            source=fm.get("source", NeuronSource.FILE.value),
            tags=fm.get("tags", []) or [],
            urgency=fm.get("urgency", Urgency.LOW.value),
            emotional_weight=fm.get("emotional_weight", EmotionalWeight.NEUTRAL.value),
            chroma_id=fm.get("chroma_id", ""),
            supabase_synced=fm.get("supabase_synced", False),
            body=body,
        )

    @property
    def filename(self) -> str:
        """Filename for this neuron: {id}.md"""
        return f"{self.id}.md"


@dataclass
class MemoryNeuron:
    """Compressed conversation/session summary."""

    id: str = ""
    type: str = "conversation-summary"
    region: str = Region.HIPPOCAMPUS.value
    title: str = ""
    created: str = ""
    source: str = NeuronSource.FILE.value
    raw_log_path: str = ""
    atomic_children: list[str] = field(default_factory=list)
    prefrontal_links: list[str] = field(default_factory=list)
    executive_links: list[str] = field(default_factory=list)
    key_themes: list[str] = field(default_factory=list)
    chroma_id: str = ""
    supabase_synced: bool = False
    body: str = ""  # 3-7 sentences summary

    def __post_init__(self):
        if not self.id:
            self.id = IDGenerator.next_memory()
        if not self.created:
            self.created = date.today().isoformat()

    def to_markdown(self) -> str:
        """Serialize to Obsidian markdown with YAML frontmatter."""
        fm = {
            "id": self.id,
            "type": self.type,
            "region": self.region,
            "title": self.title,
            "created": self.created,
            "source": self.source,
            "raw_log_path": self.raw_log_path,
            "atomic_children": self.atomic_children,
            "prefrontal_links": self.prefrontal_links,
            "executive_links": self.executive_links,
            "key_themes": self.key_themes,
            "chroma_id": self.chroma_id,
            "supabase_synced": self.supabase_synced,
        }
        return _render_frontmatter(fm, self.body)

    @classmethod
    def from_markdown(cls, content: str) -> MemoryNeuron:
        """Deserialize from Obsidian markdown."""
        fm, body = _parse_frontmatter(content)
        return cls(
            id=fm.get("id", ""),
            type=fm.get("type", "conversation-summary"),
            region=fm.get("region", Region.HIPPOCAMPUS.value),
            title=fm.get("title", ""),
            created=str(fm.get("created", "")),
            source=fm.get("source", NeuronSource.FILE.value),
            raw_log_path=fm.get("raw_log_path", ""),
            atomic_children=fm.get("atomic_children", []) or [],
            prefrontal_links=fm.get("prefrontal_links", []) or [],
            executive_links=fm.get("executive_links", []) or [],
            key_themes=fm.get("key_themes", []) or [],
            chroma_id=fm.get("chroma_id", ""),
            supabase_synced=fm.get("supabase_synced", False),
            body=body,
        )

    @property
    def filename(self) -> str:
        """Filename for this neuron: {id}.md"""
        return f"{self.id}.md"


# ── Synapse Dataclass ────────────────────────────────────

@dataclass
class Synapse:
    """Connection between two neurons with accumulated strength."""

    source_id: str = ""
    target_id: str = ""
    strength: int = 0
    last_reinforced: str = ""
    reinforcement_log: list[str] = field(default_factory=list)
    supabase_synced: bool = False

    def reinforce(self, event: str, score: int):
        """Add strength from a reinforcement event."""
        self.strength += score
        self.last_reinforced = date.today().isoformat()
        self.reinforcement_log.append(event)
        self.supabase_synced = False  # needs re-sync

    def to_dict(self) -> dict:
        """Serialize for PocketBase / SQLite queue."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "strength": self.strength,
            "last_reinforced": self.last_reinforced,
            "reinforcement_log": self.reinforcement_log,
            "supabase_synced": self.supabase_synced,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Synapse:
        """Deserialize from dict."""
        return cls(
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            strength=data.get("strength", 0),
            last_reinforced=data.get("last_reinforced", ""),
            reinforcement_log=data.get("reinforcement_log", []) or [],
            supabase_synced=data.get("supabase_synced", False),
        )


# ── Type alias for either neuron type ────────────────────
Neuron = AtomicNeuron | MemoryNeuron
