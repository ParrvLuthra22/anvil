"""Separating a reasoning model's thinking from its answer.

DeepSeek-R1 and Qwen (QwQ, Qwen3 in thinking mode) reason before they answer. Providers deliver that in one
of three ways, all handled here:

* ``<think>...</think>`` (or ``<thinking>``) inside ``message.content``;
* only the closing tag: the chat template already opened the block in the prompt, so the reply is
  ``reasoning...</think>answer``;
* a separate ``reasoning_content`` (DeepSeek, vLLM, DashScope) or ``reasoning`` (OpenRouter) field.

The reasoning must never reach the tool-call parser (it may quote example calls), the message history (it
would eat the context window) or the screen. It is still counted in the token usage, because the provider bills it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_BLOCK = re.compile(r"<(think|thinking)>(.*?)</\1\s*>", re.DOTALL | re.IGNORECASE)
_CLOSE = re.compile(r"</(?:think|thinking)\s*>", re.IGNORECASE)
_OPEN = re.compile(r"<(?:think|thinking)>", re.IGNORECASE)
_REASONING_FIELDS = ("reasoning_content", "reasoning")


@dataclass(frozen=True)
class Reply:
    """A model reply split into what to use and what to drop."""

    text: str
    """The answer, with all reasoning removed (unchanged, not even trimmed, when there was none)."""
    reasoning: str = ""
    """The reasoning that was removed, from tags and from reasoning fields, joined."""
    found: bool = False
    """Whether any reasoning was present (tags or a field), even if it was empty."""


def content_text(content: Any) -> str:
    """``message.content`` as a string: a string as is, a list of parts as the join of its text parts, else ``""``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def split_reasoning(text: str) -> Reply:
    """Remove ``<think>`` blocks from ``text``; handles complete, closing-tag-only and unclosed (cut off) blocks."""
    if "think" not in text.lower():
        return Reply(text)
    removed: list[str] = []
    found = False

    def take(match: re.Match[str]) -> str:
        nonlocal found
        found = True
        removed.append(match.group(2).strip())
        return ""

    text = _BLOCK.sub(take, text)
    closing = _CLOSE.search(text)
    if closing:  # the opening tag was in the prompt: everything before the closing tag is reasoning
        found = True
        removed.append(text[: closing.start()].strip())
        text = text[closing.end():]
    opening = _OPEN.search(text)
    if opening:  # never closed: the reply was cut off while thinking; nothing after the tag is an answer
        found = True
        removed.append(text[opening.end():].strip())
        text = text[: opening.start()]
    if not found:
        return Reply(text)
    return Reply(text.strip(), "\n\n".join(part for part in removed if part), True)


def split_message(message: dict[str, Any], *, strip: bool = True) -> Reply:
    """The usable text and the reasoning of a chat-completions ``message``.

    A ``reasoning_content`` / ``reasoning`` field is always kept out of the text. ``strip=False`` leaves ``<think>``
    tags in the text (for debugging a provider); the fields are still dropped, since they never were part of it.
    """
    text = content_text(message.get("content"))
    extra = [value.strip() for field in _REASONING_FIELDS if isinstance(value := message.get(field), str)]
    reasoning_field = "\n\n".join(part for part in extra if part)
    found_field = any(isinstance(message.get(field), str) and message[field].strip() for field in _REASONING_FIELDS)
    if not strip:
        return Reply(text, reasoning_field, found_field)
    inline = split_reasoning(text)
    reasoning = "\n\n".join(part for part in (reasoning_field, inline.reasoning) if part)
    return Reply(inline.text, reasoning, inline.found or found_field)
