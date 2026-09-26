"""Sandbox contract: the only way the agent touches the target repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ExecResult:
    """Outcome of one shell command run inside a sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration: float


class Sandbox(Protocol):
    """A working copy of the repo plus a way to run commands, checkpoint and diff it."""

    root: Path

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        """Run ``cmd`` in the repo root, killing it after ``timeout`` seconds."""
        ...

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read a file, optionally restricted to a 1-indexed inclusive line range."""
        ...

    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a file with ``content``."""
        ...

    def diff(self) -> str:
        """Return a unified diff of all changes versus the baseline."""
        ...

    def checkpoint(self, label: str) -> str:
        """Snapshot the current state and return an opaque ref id."""
        ...

    def rollback(self, ref: str) -> None:
        """Restore the state captured by ``checkpoint``."""
        ...

    def close(self) -> None:
        """Release any resources held by the sandbox."""
        ...
