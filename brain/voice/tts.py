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
_TTS_TIMEOUT_SECONDS = 60


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

    try:
        if output_file:
            # Save to file
            subprocess.run(
                [
                    str(piper_bin),
                    "--model",
                    str(piper_model),
                    "--output_file",
                    output_file,
                ],
                input=clean.encode("utf-8"),
                timeout=_TTS_TIMEOUT_SECONDS,
                check=False,
            )
        else:
            # Play directly via aplay
            piper_proc = subprocess.Popen(
                [str(piper_bin), "--model", str(piper_model), "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            aplay_proc = subprocess.Popen(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q"],
                stdin=piper_proc.stdout,
            )

            if piper_proc.stdout is not None:
                piper_proc.stdout.close()

            piper_proc.communicate(clean.encode("utf-8"), timeout=_TTS_TIMEOUT_SECONDS)
            aplay_proc.wait(timeout=_TTS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_process(piper_proc if "piper_proc" in locals() else None)
        _terminate_process(aplay_proc if "aplay_proc" in locals() else None)
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


def _terminate_process(proc: Optional[subprocess.Popen]) -> None:
    """Best-effort process teardown for timeout/error handling."""
    if proc is None:
        return

    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:
        return
