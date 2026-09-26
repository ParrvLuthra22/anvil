"""Model-agnostic token estimation: about four characters per token."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4  # role and framing that every chat message costs


def estimate_tokens(text: str) -> int:
    """Estimated tokens in ``text`` (chars / 4, rounded up)."""
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: Mapping[str, Any]) -> int:
    """Estimated tokens of one chat message, counting the arguments of any tool calls it makes."""
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(str(message.get("content") or ""))
    calls = message.get("tool_calls")
    if calls:
        tokens += estimate_tokens(json.dumps(calls, default=str))
    return tokens


def estimate_messages_tokens(messages: Iterable[Mapping[str, Any]]) -> int:
    """Estimated tokens of a whole prompt."""
    return sum(estimate_message_tokens(m) for m in messages)
