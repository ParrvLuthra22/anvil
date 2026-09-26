"""Detect a repository's languages, install/test commands and produce a compact map of it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RepoProfile:
    """What the harness needs to know to build and test a repo."""

    languages: list[str]
    primary_language: str
    install_cmd: str | None
    test_cmd: str | None
    test_framework: str | None
    notes: str = ""


def profile_repo(root: Path) -> RepoProfile:
    """Inspect ``root`` and infer languages, install command and test command."""
    raise NotImplementedError


def repo_map(root: Path, max_chars: int = 6000) -> str:
    """Return a compact file tree with top-level symbols, truncated to ``max_chars``."""
    raise NotImplementedError
