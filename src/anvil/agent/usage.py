"""Accounting for the token usage of one LLM call, shared by every place the agent calls the model."""

from __future__ import annotations

from anvil.agent.budget import Budget
from anvil.agent.emitter import Emitter
from anvil.agent.settings import AgentSettings
from anvil.llm.client import LLMResponse


def record_usage(response: LLMResponse, budget: Budget, emitter: Emitter, settings: AgentSettings) -> None:
    """Charge ``response``'s tokens to the budget and publish an ``llm_usage`` event."""
    usage = response.usage or {}
    prompt, completion = _count(usage, "prompt_tokens"), _count(usage, "completion_tokens")
    total = _count(usage, "total_tokens") or prompt + completion
    budget.add_tokens(total)
    budget.add_phase_usage(emitter.phase.value if emitter.phase else "none", prompt, completion)
    emitter.usage(prompt, completion, total, settings.cost_estimate(prompt, completion))


def _count(usage: dict, key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
