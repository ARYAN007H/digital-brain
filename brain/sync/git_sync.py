"""
Git sync for the Obsidian vault.

Auto-commits after ingestion. Never pushes automatically —
the user controls when to push to remote.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from brain.config import Paths

logger = logging.getLogger(__name__)


class GitSync:
    """Git operations for the vault."""

    def __init__(self, repo_root: Optional[Path] = None):
        self._repo_root = repo_root or Paths.ROOT
        self._repo = None

    def _get_repo(self):
        """Lazy-load the git repo."""
        if self._repo is None:
            try:
                from git import Repo
                self._repo = Repo(self._repo_root)
            except Exception as e:
                logger.error(f"Failed to open git repo at {self._repo_root}: {e}")
                raise
        return self._repo

    def auto_commit(self, source: str = "unknown", message: Optional[str] = None):
        """Stage all vault changes and commit.

        Default message format: brain-ingest: {source} {date}
        """
        try:
            repo = self._get_repo()
            repo.index.add(["vault/"])

            if not repo.index.diff("HEAD") and not repo.untracked_files:
                logger.info("No changes to commit")
                return

            commit_msg = message or f"brain-ingest: {source} {date.today().isoformat()}"
            repo.index.commit(commit_msg)
            logger.info(f"Git commit: {commit_msg}")
        except Exception as e:
            logger.warning(f"Git auto-commit failed (non-fatal): {e}")

    def has_changes(self) -> bool:
        """Check if there are uncommitted changes in the vault."""
        try:
            repo = self._get_repo()
            return repo.is_dirty(path="vault/") or bool(repo.untracked_files)
        except Exception:
            return False

    def last_commit_time(self) -> Optional[str]:
        """Get the timestamp of the last commit."""
        try:
            repo = self._get_repo()
            if repo.head.is_valid():
                return repo.head.commit.committed_datetime.isoformat()
        except Exception:
            pass
        return None
