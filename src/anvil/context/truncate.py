"""Shortening text for the model: capped tool output and one-line summaries of past tool calls."""

from __future__ import annotations

from typing import Any

_MARKER_ROOM = 40  # characters reserved for the omission marker, so the result stays within the cap
_VALUE_CHARS = 40
_CALL_CHARS = 120
_LINE_CHARS = 120


def truncate_output(text: str, cap: int) -> str:
    """Shorten ``text`` to at most about ``cap`` characters, keeping its head and its tail.

    Whole lines are dropped from the middle and replaced by ``[N lines omitted]``.
    Failures announce themselves at the start (the command) and at the end (the
    traceback or summary), so those are the parts that survive. Text whose lines
    are too long to drop cleanly (minified files) is cut by characters instead and
    marked ``[N chars omitted]``.
    """
    if len(text) <= cap:
        return text
    budget = max(cap - _MARKER_ROOM, cap // 2, 1)
    head_budget = budget // 2
    lines = text.splitlines(keepends=True)
    head = _take(lines, head_budget)
    tail = _take(lines[len(head):][::-1], budget - head_budget)[::-1]
    omitted = len(lines) - len(head) - len(tail)
    if not head or not tail or omitted < 1:
        return _cut_chars(text, budget)
    noun = "line" if omitted == 1 else "lines"
    return f"{''.join(head)}[{omitted} {noun} omitted]\n{''.join(tail)}"


def _take(lines: list[str], budget: int) -> list[str]:
    """The longest prefix of ``lines`` that fits in ``budget`` characters."""
    taken: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) > budget:
            break
        taken.append(line)
        used += len(line)
    return taken


def _cut_chars(text: str, budget: int) -> str:
    half = max(budget // 2, 1)
    return f"{text[:half]}\n[{len(text) - 2 * half} chars omitted]\n{text[-half:]}"


def first_line(text: str, limit: int = _LINE_CHARS) -> str:
    """The first non-blank line of ``text``, trimmed, and clipped to ``limit`` characters."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def format_call(name: str, args: Any, limit: int = _CALL_CHARS) -> str:
    """A compact one-line rendering of a tool call, for example ``grep(pattern='add', path='src')``."""
    if isinstance(args, dict):
        inner = ", ".join(f"{key}={_short(value)}" for key, value in args.items())
    else:
        inner = _short(args) if args else ""
    call = f"{name}({inner})"
    return call if len(call) <= limit else call[: limit - 2] + "…)"


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _VALUE_CHARS else text[: _VALUE_CHARS - 1] + "…"
