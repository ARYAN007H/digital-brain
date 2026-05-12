"""
Source-type detection and content chunking.

Handles: markdown, text, PDF, AI chat exports, browser bookmarks,
code repos, and voice transcripts.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"


@dataclass
class ExtractionResult:
    text: str
    status: str = "ok"
    reason: str | None = None
    exit_cause: str | None = None

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


def extract_text(filepath: Path, timeout_seconds: int = 30) -> ExtractionResult:
    """Extract text content from a file.

    Supports: .md, .txt, .json, .py, .html, .pdf (basic).
    """
    ext = filepath.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(filepath, timeout_seconds=timeout_seconds)
    elif ext in (".html", ".htm"):
        return ExtractionResult(text=_extract_html(filepath))
    else:
        # Plain text / markdown / code
        try:
            return ExtractionResult(text=filepath.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return ExtractionResult(text="", status="error", reason="read_failed")


def _extract_pdf(filepath: Path, timeout_seconds: int = 30) -> ExtractionResult:
    """Basic PDF text extraction. Falls back to empty string."""
    try:
        with filepath.open("rb") as f:
            sig = f.read(len(PDF_SIGNATURE))
        if sig != PDF_SIGNATURE:
            logger.warning("Rejecting PDF extraction due to invalid MIME signature", extra={"file": str(filepath)})
            return ExtractionResult(text="", status="rejected", reason="invalid_pdf_signature")

        result = subprocess.run(
            ["pdftotext", str(filepath), "-"],
            capture_output=True, text=True, timeout=timeout_seconds,
        )
        if result.returncode == 0:
            return ExtractionResult(text=result.stdout)
        stderr = (result.stderr or "").lower()
        if "syntax" in stderr or "damaged" in stderr or "corrupt" in stderr:
            cause = "corrupt_pdf"
        elif "permission" in stderr or "encrypted" in stderr:
            cause = "pdf_permission_or_encryption"
        else:
            cause = "extractor_non_zero_exit"
        return ExtractionResult(text="", status="error", reason="pdf_extract_failed", exit_cause=cause)
    except FileNotFoundError:
        logger.warning("pdftotext not found. Install poppler-utils for PDF support.")
        return ExtractionResult(text="", status="error", reason="pdftotext_missing", exit_cause="missing_dependency")
    except subprocess.TimeoutExpired:
        return ExtractionResult(text="", status="error", reason="pdf_extract_timeout", exit_cause="timeout")
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")
        return ExtractionResult(text="", status="error", reason="pdf_extract_exception", exit_cause="exception")


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
