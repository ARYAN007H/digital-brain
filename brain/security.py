"""Security helpers for redaction and startup checks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from brain.config import API, Paths


PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CARD]"),
]


def redact_pii(text: str) -> str:
    """Redact common PII-like patterns from text."""
    out = text
    for pattern, replacement in PII_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_startup_checks() -> list[CheckResult]:
    """Run configuration checks useful before first use."""
    results: list[CheckResult] = []
    results.append(CheckResult("vault_path", Paths.VAULT.exists(), str(Paths.VAULT)))
    results.append(CheckResult("data_path", Paths.DATA.exists(), str(Paths.DATA)))
    results.append(CheckResult("ollama_host", API.OLLAMA_HOST.startswith("http"), API.OLLAMA_HOST))
    results.append(CheckResult("piper_bin", Paths.PIPER_BIN.exists(), str(Paths.PIPER_BIN)))
    results.append(CheckResult("piper_model", Paths.PIPER_MODEL.exists(), str(Paths.PIPER_MODEL)))
    results.append(CheckResult("groq_key", API.has_groq(), "configured" if API.has_groq() else "missing"))
    results.append(CheckResult("gemini_key", API.has_gemini(), "configured" if API.has_gemini() else "missing"))
    results.append(CheckResult("supabase", API.has_supabase(), "configured" if API.has_supabase() else "optional/missing"))

    keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "")
    results.append(
        CheckResult(
            "ollama_keep_alive",
            keep_alive == "0",
            keep_alive or "unset",
        )
    )
    return results
