"""Global run budgets: LLM calls, tokens and wall-clock time."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from anvil.agent.settings import AgentSettings


class BudgetExceeded(Exception):
    """A global budget is used up; the run must wrap up with what it has.

    ``kind`` is ``"steps"``, ``"tokens"`` or ``"wall_clock"``.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class PhaseUsage:
    """What the model calls made during one phase cost."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Budget:
    """Counts LLM calls ("steps"), tokens and elapsed time against the configured limits."""

    def __init__(self, settings: AgentSettings, clock: Callable[[], float] = time.monotonic) -> None:
        self._max_steps = settings.max_total_steps
        self._max_tokens = settings.max_tokens_total
        self._max_seconds = settings.wall_clock_seconds
        self._clock = clock
        self._started = clock()
        self.steps = 0
        self.tokens = 0
        self.by_phase: dict[str, PhaseUsage] = {}  # in the order the phases were first charged

    @property
    def elapsed(self) -> float:
        """Seconds since the run started."""
        return self._clock() - self._started

    def check(self) -> None:
        """Raise ``BudgetExceeded`` if any limit has been reached."""
        if self.steps >= self._max_steps:
            raise BudgetExceeded("steps", f"step budget exhausted ({self._max_steps} LLM calls)")
        if self.tokens >= self._max_tokens:
            raise BudgetExceeded("tokens", f"token budget exhausted ({self.tokens} of {self._max_tokens} tokens)")
        if self.elapsed >= self._max_seconds:
            raise BudgetExceeded("wall_clock", f"wall-clock budget exhausted ({self._max_seconds:g}s)")

    def charge_step(self) -> None:
        """Account for one LLM call about to be made; raises if the budget does not allow it."""
        self.check()
        self.steps += 1

    def add_tokens(self, count: int) -> None:
        """Record tokens consumed by a completed call."""
        self.tokens += max(0, count)

    def add_phase_usage(self, phase: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Attribute one completed call to ``phase`` (for the report's tokens-by-phase table)."""
        usage = self.by_phase.setdefault(phase, PhaseUsage())
        usage.calls += 1
        usage.prompt_tokens += max(0, prompt_tokens)
        usage.completion_tokens += max(0, completion_tokens)
