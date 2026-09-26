"""Making a chat history acceptable to endpoints that are stricter than OpenAI's.

Some OpenAI-compatible servers (vLLM with Qwen, Mistral and Gemma chat templates, several hosted gateways) reject
what OpenAI itself allows: a ``system`` message that is not first, two messages in a row from the same role, a
``tool`` role when the model has no tool support, a conversation that starts with the assistant, an empty message.
The agent's history has all of these legitimately (a brief, a repo map and a kickoff are three user messages in a
row; one assistant reply can call two tools, giving two results in a row).

``normalize_messages`` rewrites a history into strictly alternating ``[system] user assistant user ...`` form
without losing any text. It never changes the input.
"""

from __future__ import annotations

from typing import Any

EMPTY = "(empty)"
_START = "(start of conversation)"
_SEPARATOR = "\n\n"


def normalize_messages(messages: list[dict[str, Any]], *, native: bool = False) -> list[dict[str, Any]]:
    """A copy of ``messages`` that strict endpoints accept.

    * one ``system`` message, first: consecutive leading ones are merged, later ones become user messages;
    * consecutive messages of the same role are merged (text joined by a blank line);
    * an empty user or assistant message gets a placeholder, since some servers reject empty content;
    * the first non-system message is a user message (a placeholder is inserted if needed).

    ``native=False`` (text-mode tool calling) also turns any leftover ``tool`` message into a user message and drops
    tool-call fields. ``native=True`` keeps assistant tool calls and ``tool`` results exactly as they are (their pairing
    must survive) and only merges around them.
    """
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system" and not turns:
            system.append(_text(message))
            continue
        if role == "system":
            role = "user"
        if role == "tool" and not native:
            turns.append({"role": "user", "content": f"[result of {message.get('name') or 'tool'}]\n{_text(message)}"})
        elif role in ("tool",) or (native and message.get("tool_calls")):
            turns.append(dict(message))
        else:
            turns.append({"role": role, "content": _text(message)})
    merged = _merge(turns)
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": _START})
    out: list[dict[str, Any]] = []
    if system := [text for text in system if text.strip()]:
        out.append({"role": "system", "content": _SEPARATOR.join(system)})
    out.extend(merged)
    return out


def _merge(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge neighbours with the same role, except messages that carry or answer tool calls."""
    merged: list[dict[str, Any]] = []
    for turn in turns:
        fixed = "tool_calls" in turn or turn["role"] == "tool"
        if merged and not fixed and merged[-1]["role"] == turn["role"] and "tool_calls" not in merged[-1]:
            merged[-1]["content"] = _SEPARATOR.join(
                part for part in (merged[-1]["content"], turn["content"]) if isinstance(part, str) and part.strip()
            )
        else:
            merged.append(dict(turn))
    for turn in merged:
        if turn["role"] in ("user", "assistant") and "tool_calls" not in turn:
            if not isinstance(turn.get("content"), str) or not turn["content"].strip():
                turn["content"] = EMPTY
    return merged


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)
