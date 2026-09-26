"""Event types and the in-process event bus shared by the agent, TUI and trace recorder."""

from __future__ import annotations

import asyncio
import threading
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
    """Fan-out channel: the agent emits events, the TUI and recorder subscribe.

    ``emit`` is safe to call from any thread, so the agent can run in a worker
    thread while the TUI consumes on the asyncio loop. For that to work,
    ``subscribe`` must be called from the consumer's event-loop thread (as the
    TUI does): the queue is then fed with ``call_soon_threadsafe``. A queue
    created outside any running loop is fed directly, which is right whenever
    emitter and consumer share a thread (tests, headless runs).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[tuple[asyncio.Queue[AgentEvent], asyncio.AbstractEventLoop | None]] = []

    def emit(self, event: AgentEvent) -> None:
        """Publish ``event`` to every subscriber."""
        with self._lock:
            subscribers = list(self._subscribers)
        for queue, loop in subscribers:
            _deliver(queue, loop, event)

    def subscribe(self) -> asyncio.Queue[AgentEvent]:
        """Return a new queue that receives every event emitted from now on."""
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        with self._lock:
            self._subscribers.append((queue, _running_loop()))
        return queue


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _deliver(
    queue: asyncio.Queue[AgentEvent], loop: asyncio.AbstractEventLoop | None, event: AgentEvent
) -> None:
    if loop is None or _running_loop() is loop:
        queue.put_nowait(event)
        return
    try:
        loop.call_soon_threadsafe(queue.put_nowait, event)
    except RuntimeError:
        pass  # the subscriber's loop is closed: nobody is listening any more
