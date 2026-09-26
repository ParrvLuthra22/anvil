"""Scripted LLM double so tests never touch the network."""

from __future__ import annotations

from anvil.llm.client import LLMClient, LLMResponse


class MockLLM(LLMClient):
    """Replays a scripted list of ``LLMResponse`` objects, one per ``chat`` call, in order."""

    def __init__(self, script: list[LLMResponse]):
        self._script = list(script)
        self._next = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Return the next scripted response; raise if the script has run out."""
        if self._next >= len(self._script):
            raise RuntimeError(f"MockLLM script exhausted after {len(self._script)} responses")
        response = self._script[self._next]
        self._next += 1
        return response
