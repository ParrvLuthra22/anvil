"""Text-mode tool calling for models without a usable native tool API.

Outgoing: ``adapt_messages_for_text_mode`` describes the tools in the system
prompt and rewrites OpenAI-style tool history (assistant ``tool_calls`` and
``tool`` role messages) into plain assistant/user turns, so the agent can keep
one canonical history format whichever mode is active.

Incoming: ``parse_text_tool_call`` finds the model's ```json {"tool": ..., "args": ...}```
block, tolerating trailing prose, several blocks, bare unfenced JSON and the
usual malformed-JSON habits (see ``loads_lenient``).
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
# Only a leading "tool" key marks a call, so unrelated JSON in prose (a package.json snippet) is ignored.
_ANCHOR_RE = re.compile(r"\{\s*[\"']tool[\"']\s*:")
_VALID_ESCAPES = frozenset('"\\/bfnrtu')
_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}

_INSTRUCTIONS = """\
You can call tools. Available tools:

{tools}

To call a tool, reply with exactly one fenced JSON block, and nothing after it:

```json
{{"tool": "<tool name>", "args": {{"<parameter>": <value>}}}}
```

Rules:
- Call at most one tool per reply. Its result arrives in the next user message.
- The block must be valid JSON: double quotes, no comments, no trailing commas, newlines inside strings written as \\n.
- When you do not need a tool (for example to give your final answer), reply in plain text with no JSON block."""


def adapt_messages_for_text_mode(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """Return a copy of ``messages`` fit for a provider that only sees plain chat turns.

    Assistant ``tool_calls`` become fenced JSON blocks in the assistant text and
    ``tool`` messages become user messages naming the tool. When ``tools`` is
    given, the tool descriptions and calling rules are appended to the first
    system message (one is created if absent). The input is never mutated.
    """
    converted: list[dict] = []
    names_by_id: dict[str, str] = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            blocks = []
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") or {}
                name = str(fn.get("name", ""))
                names_by_id[str(call.get("id", ""))] = name
                blocks.append(render_call(name, _decode_arguments(fn.get("arguments"))))
            text = msg.get("content") or ""
            converted.append({"role": "assistant", "content": "\n\n".join(p for p in (text, *blocks) if p)})
        elif role == "tool":
            name = msg.get("name") or names_by_id.get(str(msg.get("tool_call_id", "")), "tool")
            converted.append({"role": "user", "content": f"[result of {name}]\n{msg.get('content') or ''}"})
        else:
            converted.append(dict(msg))

    if tools:
        instructions = _INSTRUCTIONS.format(tools=_describe_tools(tools))
        if converted and converted[0].get("role") == "system":
            existing = converted[0].get("content") or ""
            converted[0]["content"] = f"{existing}\n\n{instructions}" if existing else instructions
        else:
            converted.insert(0, {"role": "system", "content": instructions})
    return converted


def render_call(name: str, args: dict[str, Any]) -> str:
    """Render one tool call as the fenced JSON block the model is told to emit."""
    return "```json\n" + json.dumps({"tool": name, "args": args}, ensure_ascii=False) + "\n```"


def parse_text_tool_call(text: str) -> tuple[dict | None, str]:
    """Extract the model's tool call from ``text``.

    Returns ``({"tool": name, "args": {...}}, remaining_text)`` where
    ``remaining_text`` is ``text`` without the call. Fenced blocks are tried in
    order, then bare JSON objects; the first that yields a call wins. If none
    does (no block, or malformed JSON that cannot be repaired) returns
    ``(None, text)`` with the text untouched.
    """
    for fence in _FENCE_RE.finditer(text):
        found = _call_in_block(fence.group(1))
        if found is not None:
            return found, _without(text, fence.start(), fence.end())
    found_span = _first_call(text)
    if found_span is not None:
        call, start, end = found_span
        return call, _without(text, start, end)
    return None, text


def loads_lenient(text: str) -> Any:
    """Parse JSON, repairing the mistakes models commonly make.

    Repairs: raw newlines/tabs inside strings, single-quoted strings, Python
    ``True/False/None``, trailing commas, stray backslashes (``\\d``) and
    missing closing brackets. A value cut off inside a string is never
    completed, since guessing could execute a half-written edit. Raises
    ``ValueError`` when the text cannot be parsed.
    """
    try:
        return json.loads(text)
    except ValueError:
        pass
    repaired = _repair(text)
    if repaired is None:
        raise ValueError("unrepairable JSON: cut off inside a string")
    return json.loads(repaired)


def _describe_tools(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        spec = tool.get("function", tool)
        params = json.dumps(spec.get("parameters") or {}, separators=(",", ":"))
        description = (spec.get("description") or "").strip()
        lines.append(f"- {spec.get('name')}: {description}\n  parameters (JSON Schema): {params}")
    return "\n".join(lines)


def _decode_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = loads_lenient(raw) if isinstance(raw, str) and raw.strip() else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _call_in_block(block: str) -> dict | None:
    """Find a tool call in one fenced block: whole-block JSON first, then any embedded object."""
    try:
        call = _as_call(loads_lenient(block.strip()))
    except ValueError:
        call = None
    if call is not None:
        return call
    found = _first_call(block)
    return found[0] if found else None


def _first_call(text: str) -> tuple[dict, int, int] | None:
    """Return (call, start, end) for the first ``{"tool": ...}`` object in ``text``."""
    for anchor in _ANCHOR_RE.finditer(text):
        end = _object_end(text, anchor.start())
        try:
            call = _as_call(loads_lenient(text[anchor.start():end]))
        except ValueError:
            continue
        if call is not None:
            return call, anchor.start(), end
    return None


def _as_call(obj: Any) -> dict | None:
    """Validate a parsed value as ``{"tool": str, "args": dict}`` (``arguments`` accepted for ``args``)."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("args", obj.get("arguments"))
    if args is None:
        args = {}
    elif isinstance(args, str):
        try:
            args = loads_lenient(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"tool": name.strip(), "args": args}


def _object_end(text: str, start: int) -> int:
    """Index just past the bracket-balanced value opening at ``start`` (``len(text)`` if unbalanced)."""
    depth, quote, i = 0, None, start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def _without(text: str, start: int, end: int) -> str:
    before, after = text[:start].strip(), text[end:].strip()
    return "\n\n".join(part for part in (before, after) if part)


def _repair(text: str) -> str | None:
    """One string-aware pass that turns near-JSON into JSON; ``None`` if it ends inside a string."""
    out: list[str] = []
    closers: list[str] = []
    quote: str | None = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\":
                nxt = text[i + 1] if i + 1 < n else ""
                if nxt == "'":
                    out.append("'")
                    i += 2
                elif nxt and nxt in _VALID_ESCAPES:
                    out.append(ch + nxt)
                    i += 2
                else:
                    out.append("\\\\")
                    i += 1
                continue
            if ch == quote:
                out.append('"')
                quote = None
            elif ch == '"':
                out.append('\\"')
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
            else:
                out.append(ch)
            i += 1
            continue

        if ch in "\"'":
            quote = ch
            out.append('"')
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
            out.append(ch)
        elif ch in "}]":
            if closers and closers[-1] == ch:
                closers.pop()
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if not (j < n and text[j] in "}]"):
                out.append(ch)
        elif ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            out.append(_PY_LITERALS.get(word, word))
            i = j
            continue
        else:
            out.append(ch)
        i += 1

    if quote:
        return None
    body = "".join(out).rstrip()
    if body.endswith(","):
        body = body[:-1]
    return body + "".join(reversed(closers))
