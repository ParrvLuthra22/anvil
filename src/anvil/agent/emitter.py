"""Builds and publishes the ``AgentEvent`` stream for one run."""

from __future__ import annotations

import logging
import time
from typing import Callable

from anvil.events import AgentEvent, EventBus, Phase

logger = logging.getLogger("anvil.agent")
logger.addHandler(logging.NullHandler())

MESSAGE_EVENT_CHARS = 4000
PREVIEW_CHARS = 500


class Emitter:
    """Typed helpers over ``EventBus.emit`` that tag every event with the current phase.

    A misbehaving subscriber must never break a run, so publishing errors are
    logged and swallowed.
    """

    def __init__(self, bus: EventBus, clock: Callable[[], float] = time.time) -> None:
        self._bus = bus
        self._clock = clock
        self.phase: Phase | None = None

    def set_phase(self, phase: Phase) -> None:
        """Enter ``phase`` and announce it."""
        self.phase = phase
        self._emit("phase", {"name": phase.value})

    def message(self, role: str, text: str) -> None:
        """Publish a conversation message (long text is truncated for the event only)."""
        self._emit("message", {"role": role, "text": _clip(text, MESSAGE_EVENT_CHARS)})

    def tool_call(self, tool: str, args: dict) -> None:
        """Publish that ``tool`` is about to run with ``args``."""
        self._emit("tool_call", {"tool": tool, "args": args})

    def tool_result(self, tool: str, ok: bool, output: str) -> None:
        """Publish a tool's outcome with a short preview of its output."""
        self._emit("tool_result", {"tool": tool, "ok": ok, "output_preview": _clip(output, PREVIEW_CHARS)})

    def usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int, cost_estimate: float) -> None:
        """Publish the token usage of one model call."""
        self._emit(
            "llm_usage",
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_estimate": cost_estimate,
            },
        )

    def error(self, kind: str, message: str) -> None:
        """Publish a non-fatal or fatal problem (the run's fate is decided elsewhere)."""
        self._emit("error", {"kind": kind, "message": message})

    def done(self, **data: object) -> None:
        """Publish the final event of the run."""
        self._emit("done", dict(data))

    def _emit(self, event_type: str, data: dict) -> None:
        try:
            self._bus.emit(AgentEvent(ts=self._clock(), type=event_type, phase=self.phase, data=data))
        except Exception:  # noqa: BLE001 - see class docstring
            logger.exception("event subscriber failed while handling a %s event", event_type)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"
