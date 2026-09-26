"""Conversation history for the agent: every message the model sees goes through here.

This is the pass-through version: it keeps the full history and sends all of it.
A later step replaces the internals with compaction (dropping stale tool output,
keeping ``pinned`` messages) but must keep this public surface, including the
extra message fields accepted by ``add_message``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _Entry:
    message: dict[str, Any]
    pinned: bool


class ContextManager:
    """Ordered message history plus the logic that turns it into a model prompt."""

    def __init__(self) -> None:
        self._entries: list[_Entry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add_message(self, role: str, content: str, pinned: bool = False, **fields: Any) -> None:
        """Append a message in OpenAI chat format.

        ``pinned`` marks messages that must survive compaction (the issue, phase
        summaries). ``fields`` carries the extra keys tool calling needs:
        ``tool_calls`` on an assistant message, ``tool_call_id`` on a ``tool``
        message.
        """
        self._entries.append(_Entry({"role": role, "content": content, **fields}, pinned))

    def build_messages(self, phase_goal: str) -> list[dict[str, Any]]:
        """Return the prompt for the next model call.

        ``phase_goal`` is the current phase's instructions; it leads the prompt
        as the system message. The returned dicts are copies, so callers (and
        the LLM client) can never corrupt the stored history.
        """
        return [{"role": "system", "content": phase_goal}, *(dict(e.message) for e in self._entries)]
