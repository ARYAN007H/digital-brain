"""
Source-type detection and content chunking.

Handles: markdown, text, PDF, AI chat exports, browser bookmarks,
code repos, and voice transcripts.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Source Type Detection ────────────────────────────────

SOURCE_PATTERNS = {
    "claude": [r"Human:", r"Assistant:", r"claude"],
    "chatgpt": [r"ChatGPT", r"user:", r"assistant:"],
    "voice": [r"voice-transcript", r"whisper", r"transcri"],
    "browser": [r"<!DOCTYPE NETSCAPE-Bookmark", r"<DL>", r"bookmark"],
    "code-repo": [r"```", r"#brain", r"commit [a-f0-9]"],
}


def detect_source(filepath: Path) -> str:
    """Detect the source type of a file based on content and extension.

    Returns one of: claude, chatgpt, voice, browser, code-repo, file.
    """
    ext = filepath.suffix.lower()

    # Extension-based hints
    if ext == ".pdf":
        return "file"
    if ext in (".html", ".htm"):
        content = filepath.read_text(encoding="utf-8", errors="ignore")[:2000]
        if any(re.search(p, content, re.IGNORECASE) for p in SOURCE_PATTERNS["browser"]):
            return "browser"
        return "file"

    # Content-based detection
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")[:5000]
    except Exception:
        return "file"

    for source, patterns in SOURCE_PATTERNS.items():
        if any(re.search(p, content, re.IGNORECASE) for p in patterns):
            return source

    return "file"


def extract_text(filepath: Path) -> str:
    """Extract text content from a file.

    Supports: .md, .txt, .json, .py, .html, .pdf (basic).
    """
    ext = filepath.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(filepath)
    elif ext in (".html", ".htm"):
        return _extract_html(filepath)
    else:
        # Plain text / markdown / code
        try:
            return filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return ""


def _extract_pdf(filepath: Path) -> str:
    """Basic PDF text extraction. Falls back to empty string."""
    try:
        import subprocess
        result = subprocess.run(
            ["pdftotext", str(filepath), "-"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except FileNotFoundError:
        logger.warning("pdftotext not found. Install poppler-utils for PDF support.")
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
    return ""


def _extract_html(filepath: Path) -> str:
    """Strip HTML tags for plain text."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    # Simple tag stripping
    clean = re.sub(r"<[^>]+>", " ", content)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


# ── Content Chunking ─────────────────────────────────────

def chunk_content(text: str, max_chars: int = 3000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for LLM processing.

    Uses paragraph boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = re.split(r"\n\s*\n", text)
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk with overlap from previous
            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para

            # Handle paragraphs longer than max_chars
            while len(current_chunk) > max_chars:
                split_point = current_chunk.rfind(" ", 0, max_chars)
                if split_point == -1:
                    split_point = max_chars
                chunks.append(current_chunk[:split_point])
                current_chunk = current_chunk[split_point:].strip()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
