"""The LLM call the context manager makes when the history outgrows its budget."""

from __future__ import annotations

from typing import Callable

from anvil.agent.budget import Budget
from anvil.agent.emitter import Emitter
from anvil.agent.prompts import SUMMARIZER_PROMPT
from anvil.agent.settings import AgentSettings
from anvil.agent.usage import record_usage
from anvil.llm.client import LLMClient
from anvil.llm.errors import LLMError


class HistorySummarizer:
    """Condenses a transcript of old agent history into a short summary with one LLM call.

    It is a ``ContextManager`` summariser: it may raise, and the manager then falls
    back to pruning. The call counts as a step and its tokens are charged to the run
    budget like any other model call. ``get_llm`` is looked up on every call because the
    orchestrator creates its client lazily.
    """

    def __init__(
        self,
        get_llm: Callable[[], LLMClient | None],
        budget: Budget,
        emitter: Emitter,
        settings: AgentSettings,
    ) -> None:
        self._get_llm = get_llm
        self._budget = budget
        self._emitter = emitter
        self._settings = settings

    def __call__(self, transcript: str) -> str:
        """Return the summary of ``transcript``; raises if the budget is spent or the LLM call fails."""
        llm = self._get_llm()
        if llm is None:
            raise RuntimeError("no LLM client to summarise with")
        self._budget.charge_step()
        self._emitter.message("system", "The context window is filling up: summarising older history.")
        messages = [
            {"role": "system", "content": SUMMARIZER_PROMPT},
            {"role": "user", "content": f"<history>\n{transcript}\n</history>"},
        ]
        try:
            response = llm.chat(messages)
        except LLMError as exc:
            self._emitter.error("context", f"could not summarise old history, pruning it instead: {exc}")
            raise
        record_usage(response, self._budget, self._emitter, self._settings)
        return response.text
