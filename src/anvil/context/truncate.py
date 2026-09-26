"""Shortening text for the model: capped tool output and one-line summaries of past tool calls."""

from __future__ import annotations

import re
from typing import Any

_MARKER_ROOM = 40  # characters reserved for the omission marker, so the result stays within the cap
_VALUE_CHARS = 40
_CALL_CHARS = 120
_LINE_CHARS = 120
_NUMBERED_LINE = re.compile(r"^\s*(\d+)  (.*)$")  # read_file prints "%6d  text"
_DEFINITION = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?:def|class|function|func|fn|interface|struct|enum|trait|impl|type|module|object)\b"
)
_SYNTAX = re.compile(r"[(){}:<]")
_MIN_HEAD_LINES = 10
_OUTLINE_MAX = 80
_OUTLINE_LINE_CHARS = 100


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


def clip_lines(text: str, limit: int) -> str:
    """Keep the leading whole lines of ``text`` that fit in ``limit`` characters, noting how many were left out."""
    if len(text) <= limit:
        return text
    lines = text.splitlines(keepends=True)
    kept = _take(lines, limit)
    left = len(lines) - len(kept)
    return f"{''.join(kept)}[{left} more {'line' if left == 1 else 'lines'} omitted]"


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


def _is_definition(code: str) -> bool:
    """A line that declares something (``def f(``, ``class A:``, ``func F() {``), not prose that starts with such a word."""
    return bool(_DEFINITION.match(code) and _SYNTAX.search(code) and not code.rstrip().endswith("."))


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


def cap_read_output(text: str, args: Any, max_lines: int, head_lines: int, char_budget: int) -> str:
    """Shorten a ``read_file`` result of more than ``max_lines`` lines to its first lines plus an outline.

    A model that reads a whole large file pays for it in every later prompt until the observation is pruned. What it
    usually needs is where things are, so the result becomes the first ``head_lines`` lines, an outline of the
    definitions in the rest of the file (``line: signature``), and a hint to read a line range for anything else.
    Text of ``max_lines`` lines or fewer, and anything that is not numbered file text, is returned unchanged. The
    result stays within ``char_budget`` characters: the head shrinks first, then the outline.
    """
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    parsed = [_NUMBERED_LINE.match(line) for line in lines]
    numbered = sum(1 for m in parsed if m)
    if numbered < 0.8 * len(lines):
        return text  # not file text (a listing, an error): the ordinary output cap deals with it
    numbers = [int(m.group(1)) for m in parsed if m]
    path = args.get("path") if isinstance(args, dict) else None
    where = f"path={path!r}, " if path else ""

    def build(head: int, outline_limit: int) -> str:
        outline = [
            f"  {m.group(1)}: {m.group(2).strip()[:_OUTLINE_LINE_CHARS]}"
            for line, m in zip(lines[head:], parsed[head:])
            if m and _is_definition(m.group(2))
        ]
        shown, left = outline[:outline_limit], max(len(outline) - outline_limit, 0)
        notice = (
            f"[read_file shortened: {len(lines)} lines (numbered {numbers[0]} to {numbers[-1]}) is more than the "
            f"{max_lines} shown at once. Here are the first {head} lines and an outline of the rest. To read a part, "
            f"call read_file({where}start=<first line>, end=<last line>) with a range of at most {max_lines} lines.]"
        )
        parts = [notice, *lines[:head]]
        if shown:
            parts.append(f"[outline of lines {numbers[head] if head < len(numbers) else numbers[-1]} to {numbers[-1]}, as line: definition]")
            parts.extend(shown)
        if left:
            parts.append(f"[{left} more definitions not listed]")
        return "\n".join(parts) + "\n"

    head = min(head_lines, max_lines, len(lines))
    limit = _OUTLINE_MAX
    result = build(head, limit)
    while len(result) > char_budget and head > _MIN_HEAD_LINES:
        head = max(_MIN_HEAD_LINES, head - 10)
        result = build(head, limit)
    while len(result) > char_budget and limit > 0:
        limit //= 2
        result = build(head, limit)
    return result
