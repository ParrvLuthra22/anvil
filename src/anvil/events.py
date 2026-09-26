"""Event types and the in-process event bus shared by the agent, TUI and trace recorder."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    """Stages of the agent cycle, in execution order."""

    INGEST = "ingest"
    PROFILE = "profile"
    UNDERSTAND = "understand"
    LOCALIZE = "localize"
    REPRODUCE = "reproduce"
    PATCH = "patch"
    VERIFY = "verify"
    REVIEW = "review"
    FINALIZE = "finalize"


@dataclass
class AgentEvent:
    """One observable step of a run.

    ``type`` is one of "phase" | "message" | "tool_call" | "tool_result" |
    "llm_usage" | "error" | "done"; ``data`` carries the per-type payload
    documented in CLAUDE.md.
    """

    ts: float
    type: str
    phase: Phase | None
    data: dict


class EventBus:
    """Fan-out channel: the agent emits events, the TUI and recorder subscribe."""

    def emit(self, event: AgentEvent) -> None:
        """Publish ``event`` to every subscriber."""
        raise NotImplementedError

    def subscribe(self) -> asyncio.Queue[AgentEvent]:
        """Return a new queue that receives every event emitted from now on."""
        raise NotImplementedError
