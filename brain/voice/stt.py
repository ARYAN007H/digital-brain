"""
Speech-to-text via faster-whisper.

CPU-only, int8 quantized base model (~300MB RAM).
Model loaded on demand, kept alive during voice sessions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.config import Hardware

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None


def _get_model():
    """Lazy-load the Whisper model."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper base model (CPU, int8)...")
        _model = WhisperModel(
            "base",
            device=Hardware.DEVICE,
            compute_type=Hardware.COMPUTE_TYPE,
        )
        logger.info("Whisper model loaded")
    return _model


def transcribe(audio_path: str | Path) -> str:
    """Transcribe an audio file to text.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)

    Returns:
        Transcribed text string.
    """
    model = _get_model()
    segments, info = model.transcribe(str(audio_path), beam_size=5)

    text = " ".join(segment.text for segment in segments).strip()
    logger.info(
        f"Transcribed {Path(audio_path).name}: "
        f"lang={info.language} prob={info.language_probability:.2f} "
        f"len={len(text)} chars"
    )
    return text


def unload():
    """Unload the model to free ~300MB RAM."""
    global _model
    if _model is not None:
        _model = None
        logger.info("Whisper model unloaded")
