"""
Text-to-speech via Piper TTS.

Offline, single binary, ~50MB RAM. Pipes through aplay for playback.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from brain.config import Paths

logger = logging.getLogger(__name__)


def speak(text: str, output_file: Optional[str] = None):
    """Speak text aloud using Piper TTS.

    If output_file is provided, saves to WAV instead of playing.
    """
    piper_bin = Paths.PIPER_BIN
    piper_model = Paths.PIPER_MODEL

    if not piper_bin.exists():
        logger.error(f"Piper binary not found: {piper_bin}")
        logger.info("Download from: https://github.com/rhasspy/piper/releases")
        return

    if not piper_model.exists():
        logger.error(f"Piper model not found: {piper_model}")
        return

    # Strip markdown formatting for natural speech
    clean = _clean_for_speech(text)

    if not clean.strip():
        return

    if output_file:
        # Save to file
        cmd = (
            f'echo "{_escape_shell(clean)}" | '
            f'{piper_bin} --model {piper_model} --output_file {output_file}'
        )
    else:
        # Play directly via aplay
        cmd = (
            f'echo "{_escape_shell(clean)}" | '
            f'{piper_bin} --model {piper_model} --output_raw | '
            f'aplay -r 22050 -f S16_LE -t raw -q'
        )

    try:
        subprocess.run(cmd, shell=True, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        logger.warning("TTS playback timed out")
    except Exception as e:
        logger.error(f"TTS failed: {e}")


def _clean_for_speech(text: str) -> str:
    """Remove markdown formatting for natural speech output."""
    clean = text
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)  # bold
    clean = re.sub(r"\*(.+?)\*", r"\1", clean)       # italic
    clean = re.sub(r"`(.+?)`", r"\1", clean)          # inline code
    clean = re.sub(r"```[\s\S]*?```", "", clean)       # code blocks
    clean = re.sub(r"^#+\s*", "", clean, flags=re.MULTILINE)  # headers
    clean = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", clean)  # links
    clean = re.sub(r"^\s*[-*]\s+", "", clean, flags=re.MULTILINE)  # bullets
    clean = re.sub(r"\n{3,}", "\n\n", clean)           # excess newlines
    return clean.strip()


def _escape_shell(text: str) -> str:
    """Escape text for shell echo command."""
    return text.replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
