"""LLM client contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """A single model reply: free text, requested tool calls, and token usage."""

    text: str
    tool_calls: list[dict]
    usage: dict


class LLMClient(Protocol):
    """Anything that can answer a chat request; implemented by the real client and by MockLLM."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Send ``messages`` (and optional tool schemas) and return the model's reply."""
        ...
