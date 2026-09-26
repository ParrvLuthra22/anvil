"""JSONL trace recorder and loader for a single run.

Each event is written as one JSON line and flushed immediately so the file is
always readable even if the process is killed mid-run.  The loader tolerates a
corrupted (truncated) last line, which happens when the process is killed.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

from anvil.events import AgentEvent, Phase


def _event_to_dict(event: AgentEvent) -> dict:
    """Serialise an :class:`AgentEvent` to a plain dict."""
    return {
        "ts": event.ts,
        "type": event.type,
        "phase": event.phase.value if isinstance(event.phase, Phase) else event.phase,
        "data": event.data,
    }


def _dict_to_event(raw: dict) -> AgentEvent:
    """Deserialise a plain dict into an :class:`AgentEvent`."""
    phase_raw = raw.get("phase")
    phase: Phase | None = None
    if phase_raw is not None:
        try:
            phase = Phase(phase_raw)
        except ValueError:
            phase = None  # tolerate unknown future phase names

    return AgentEvent(
        ts=float(raw.get("ts", 0.0)),
        type=str(raw.get("type", "unknown")),
        phase=phase,
        data=raw.get("data") or {},
    )


class TraceRecorder:
    """Appends every event of a run to a JSON-lines file.

    Usage::

        recorder = TraceRecorder(Path("output/trace.jsonl"))
        recorder.record(event)          # call for every AgentEvent
        # later, for replay:
        for ev in TraceRecorder.load(Path("output/trace.jsonl")):
            ...
    """

    def __init__(self, path: Path) -> None:
        """Open *path* for appending (creates parent dirs and the file if needed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Open in text-append mode; we flush after every write.
        self._path = path
        self._fh = path.open("a", encoding="utf-8")

    def record(self, event: AgentEvent) -> None:
        """Append one JSON line for *event* and flush to disk immediately."""
        line = json.dumps(_event_to_dict(event), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        self._fh.flush()
        self._fh.close()

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def load(path: Path) -> Iterator[AgentEvent]:
        """Yield :class:`AgentEvent` objects stored in the JSONL trace at *path*.

        Skips blank lines.  If the **last** line is corrupt (e.g. truncated by a
        crash), it is silently dropped so the rest of the trace is still usable.
        """
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                yield _dict_to_event(raw)
            except (json.JSONDecodeError, KeyError, TypeError):
                if idx == len(lines) - 1:
                    # Tolerate a corrupted last line (process killed mid-write).
                    return
                # Mid-file corruption: skip and continue so we recover as much
                # of the trace as possible.
                continue
