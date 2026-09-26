"""JSONL trace of a run, used for reports and offline replay."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from anvil.events import AgentEvent


class TraceRecorder:
    """Appends every event of a run to a JSON-lines file."""

    def __init__(self, path: Path):
        """Prepare to write events to ``path``."""
        raise NotImplementedError

    def record(self, event: AgentEvent) -> None:
        """Append one JSON line for ``event`` and flush."""
        raise NotImplementedError

    @staticmethod
    def load(path: Path) -> Iterator[AgentEvent]:
        """Yield the events stored in the trace file at ``path``."""
        raise NotImplementedError
