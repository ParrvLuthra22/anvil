"""Text-mode tool calling for models without a usable native tool API.

Outgoing: ``adapt_messages_for_text_mode`` describes the tools in the system
prompt and rewrites OpenAI-style tool history (assistant ``tool_calls`` and
``tool`` role messages) into plain assistant/user turns, so the agent can keep
one canonical history format whichever mode is active.

Incoming: ``parse_text_tool_calls`` finds the model's calls in a reply. Models answer in whatever
dialect they were trained on, so all of these are accepted, surrounded by prose or not:

* our own fenced block: ```json {"tool": "read_file", "args": {...}}```
* Hermes / Qwen tags: ``<tool_call>{"name": "read_file", "arguments": {...}}</tool_call>``, closed or cut off
* Qwen3-Coder XML: ``<tool_call><function=read_file><parameter=path>a.py</parameter></function></tool_call>``
* a bare JSON object with ``tool``/``args`` or ``name``/``arguments`` (also ``parameters``)
* OpenAI-shaped JSON: ``{"function": {"name": ..., "arguments": "<json string>" or {...}}}`` or ``{"tool_calls": [...]}``
* several of the above in one reply (the first is taken; ``ParsedCalls`` reports them all)

Malformed JSON is repaired where that is safe (see ``loads_lenient``).
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_TAG_OPEN_RE = re.compile(r"<(tool_call|function_call|tool_use)\s*>", re.IGNORECASE)
# Anchors that start a bare call object: a name key first, or an arguments key first.
_ANCHOR_RE = re.compile(
    r"\{\s*[\"'](?:tool|name|function|tool_name|args|arguments|parameters)[\"']\s*:"
)
_XML_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function\s*>|(?=<function=)|\Z)", re.DOTALL | re.IGNORECASE)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)(?:</parameter\s*>|(?=<parameter=)|(?=</function)|\Z)", re.DOTALL | re.IGNORECASE
)
_VALID_ESCAPES = frozenset('"\\/bfnrtu')
_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}
_NAME_KEYS = ("tool", "name", "function", "tool_name")
_ARG_KEYS = ("args", "arguments", "parameters", "params", "input")
# ``name`` alone is a common key in unrelated JSON; it only marks a call when it comes with these argument keys.
_STRICT_ARG_KEYS = ("arguments", "parameters", "params", "input")

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


@dataclass
class ParsedCalls:
    """Every tool call found in a reply, in order, and the reply with all of that call syntax removed."""

    calls: list[dict]
    text: str


@dataclass
class _Found:
    start: int
    end: int
    calls: list[dict]


def parse_text_tool_calls(text: str, tools: list[dict] | None = None) -> ParsedCalls:
    """Find all the tool calls in ``text`` (see the module docstring for the accepted forms).

    ``tools`` (OpenAI-format schemas, optional) sharpens two things: a JSON object with just a ``name`` is only a
    call if that name is one of the tools, and a value written as text inside XML parameters is converted to the type
    the schema declares. Each call is ``{"tool": name, "args": {...}}``. ``text`` has every recognised call removed,
    prose around them kept.
    """
    found = _scan(text, _tool_index(tools))
    return ParsedCalls([call for item in found for call in item.calls], _without_spans(text, found))


def parse_text_tool_call(text: str, tools: list[dict] | None = None) -> tuple[dict | None, str]:
    """Extract the first tool call from ``text``.

    Returns ``({"tool": name, "args": {...}}, remaining_text)`` where ``remaining_text`` is ``text`` without that
    call (any further calls stay in it). If there is no call (no block, or malformed JSON that cannot be repaired)
    returns ``(None, text)`` with the text untouched.
    """
    found = _scan(text, _tool_index(tools))
    if not found:
        return None, text
    first = found[0]
    return first.calls[0], _without(text, first.start, first.end)


def _scan(text: str, index: Mapping[str, Mapping[str, str]]) -> list[_Found]:
    """All call spans in ``text`` in order of appearance: tags first, then fences, then bare objects."""
    found = _tag_calls(text, index)
    for fence in _FENCE_RE.finditer(text):
        if _overlaps(fence.start(), fence.end(), found):
            continue
        calls = _payload_calls(fence.group(1), index, "fence")[0] or [
            item for hit in _bare_calls(fence.group(1), index, []) for item in hit.calls
        ]
        if calls:
            found.append(_Found(fence.start(), fence.end(), calls))
    found += _bare_calls(text, index, found)
    found += _function_block_calls(text, index, found)
    if not found:
        found = _python_syntax_calls(text, index)
    return sorted(found, key=lambda item: item.start)


_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function\s*>", re.DOTALL | re.IGNORECASE)
_TOOL_TAG_AFTER_RE = re.compile(r"\s*</(?:tool_call|function_call|tool_use)\s*>", re.IGNORECASE)
_ORPHAN_FENCE_RE = re.compile(r"\s*(?:`{1,3}\w*\s*)?")
_OPENING_FENCE_RE = re.compile(r"```[\w+-]*[ \t]*\n?[ \t]*\Z")
_QUOTES = ("\"\"\"", "'''", "\"", "'")


def _function_block_calls(text: str, index: Mapping[str, Mapping[str, str]], taken: list[_Found]) -> list[_Found]:
    """Qwen3-Coder ``<function=NAME>...</function>`` blocks that lost their ``<tool_call>`` wrapper.

    Seen from a real provider whose own parser consumed the opening tag and left the rest in the reply. Only known
    tools, and only blocks that were closed: without the wrapper nothing else says the model meant the call, so half
    of an edit is never acted on. A closing tool-call tag right after the block is part of it (an opening one before
    it would have made the block a wrapped call, read by ``_tag_calls``).
    """
    found: list[_Found] = []
    for block in _FUNCTION_BLOCK_RE.finditer(text):
        name = block.group(1).strip()
        if _overlaps(block.start(), block.end(), taken) or (index and name not in index):
            continue
        call = _xml_call(name, block.group(2), index)
        if call is None:
            continue
        after = _TOOL_TAG_AFTER_RE.match(text, block.end())
        found.append(_Found(block.start(), after.end() if after else block.end(), [call]))
    return found


def _python_syntax_calls(text: str, index: Mapping[str, Mapping[str, str]]) -> list[_Found]:
    """``tool_name(key="value", ...)``: the Python-call form models fall back to, as the last thing in the reply.

    Only a known tool, starting a line (list and quote marks allowed before it), followed by nothing but an optional
    stray closing fence. Arguments must be literals: they are read with ``ast.literal_eval`` and never executed. A
    call that is cut off is not completed by guesswork.
    """
    if not index:
        return []
    names = "|".join(re.escape(name) for name in sorted(index, key=len, reverse=True))
    for opening in re.finditer(rf"^[ \t>*`\-]*({names})\(", text, re.MULTILINE):
        scanned = _call_arguments(text, opening.end())
        if scanned is None:
            continue
        close, inner = scanned
        tail = _ORPHAN_FENCE_RE.fullmatch(text, close)
        if tail is None:
            continue  # more reply follows: the call is not the last thing, so it is prose about a call
        call = _literal_call(opening.group(1), inner, index)
        if call is None:
            continue
        start = opening.start(1)
        fence = _OPENING_FENCE_RE.search(text, 0, start)
        return [_Found(fence.start() if fence else start, len(text), [call])]
    return []


def _call_arguments(text: str, begin: int) -> tuple[int, str] | None:
    """Where the parenthesis opened just before ``begin`` closes, and the text between, read string-aware.

    Returns ``(index after the closing parenthesis, arguments)``, or ``None`` when it never closes. A raw newline
    inside a single-line quoted string is turned into ``\\n``, which is what the model meant.
    """
    depth, position, out = 1, begin, []
    while position < len(text):
        quote = next((q for q in _QUOTES if text.startswith(q, position)), None)
        if quote is not None:
            out.append(quote)
            position += len(quote)
            while position < len(text) and not text.startswith(quote, position):
                char = text[position]
                if char == "\\" and position + 1 < len(text):
                    out.append(text[position:position + 2])
                    position += 2
                    continue
                out.append("\\n" if char == "\n" and len(quote) == 1 else "\\t" if char == "\t" and len(quote) == 1 else char)
                position += 1
            if position >= len(text):
                return None  # cut off inside a string
            out.append(quote)
            position += len(quote)
            continue
        char = text[position]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return position + 1, "".join(out)
        out.append(char)
        position += 1
    return None


def _literal_call(name: str, inner: str, index: Mapping[str, Mapping[str, str]]) -> dict | None:
    """``name(inner)`` as a call if every argument is a Python literal; positional ones take the schema's order."""
    try:
        tree = ast.parse(f"{name}({inner})", mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != name:
        return None
    parameters = list(index.get(name, {}))
    args: dict[str, Any] = {}
    try:
        for position, node in enumerate(call.args):
            if isinstance(node, ast.Starred) or position >= len(parameters):
                return None
            args[parameters[position]] = ast.literal_eval(node)
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in args:
                return None  # ``**kwargs``, or a name given twice: which value was meant is anyone's guess
            args[keyword.arg] = ast.literal_eval(keyword.value)
    except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
        return None
    return {"tool": name, "args": args}


def _tag_calls(text: str, index: Mapping[str, Mapping[str, str]]) -> list[_Found]:
    """Calls wrapped in ``<tool_call>`` (or ``<function_call>``, ``<tool_use>``) tags, closed or cut off."""
    found: list[_Found] = []
    resume = 0
    for opening in _TAG_OPEN_RE.finditer(text):
        if opening.start() < resume:
            continue
        after = opening.end()
        closing = re.compile(rf"</{opening.group(1)}\s*>", re.IGNORECASE).search(text, after)
        following = _TAG_OPEN_RE.search(text, after)
        if closing and (following is None or closing.start() < following.start()):
            content_end, span_end = closing.start(), closing.end()
        else:  # never closed: the reply ended, or the next call began
            content_end = span_end = following.start() if following else len(text)
        calls, used = _payload_calls(text[after:content_end], index, "tag")
        if not calls:
            continue
        if span_end == content_end:
            span_end = after + used
        found.append(_Found(opening.start(), span_end, calls))
        resume = span_end
    return found


def _bare_calls(text: str, index: Mapping[str, Mapping[str, str]], taken: list[_Found]) -> list[_Found]:
    """Call objects written straight into prose, outside the spans in ``taken``."""
    found: list[_Found] = []
    resume = 0
    for anchor in _ANCHOR_RE.finditer(text):
        start = anchor.start()
        if start < resume or _overlaps(start, start + 1, taken):
            continue
        end = _object_end(text, start)
        try:
            call = _as_call(loads_lenient(text[start:end]), index, "bare")
        except ValueError:
            continue
        if call is not None:
            found.append(_Found(start, end, [call]))
            resume = end
    return found


def _payload_calls(content: str, index: Mapping[str, Mapping[str, str]], context: str) -> tuple[list[dict], int]:
    """The calls in the content of a tag or a fence, and how many characters of it they take up."""
    xml = list(_XML_FUNCTION_RE.finditer(content))
    if xml:
        calls = [_xml_call(m.group(1), m.group(2), index) for m in xml]
        return [c for c in calls if c is not None], xml[-1].end()
    opening = re.search(r"[\[{]", content)
    if opening is None:
        return [], 0
    end = _object_end(content, opening.start())
    try:
        value = loads_lenient(content[opening.start():end])
    except ValueError:
        return [], 0
    return _calls_in(value, index, context), end


def _calls_in(value: Any, index: Mapping[str, Mapping[str, str]], context: str) -> list[dict]:
    """A parsed JSON value as calls: an object, a list of them, or a ``{"tool_calls": [...]}`` wrapper."""
    if isinstance(value, list):
        return [call for item in value for call in _calls_in(item, index, context)]
    if not isinstance(value, dict):
        return []
    wrapped = value.get("tool_calls")
    if isinstance(wrapped, list):
        return _calls_in(wrapped, index, context)
    call = _as_call(value, index, context)
    return [call] if call is not None else []


def _as_call(obj: Any, index: Mapping[str, Mapping[str, str]], context: str) -> dict | None:
    """Validate a parsed object as a call: ``{"tool"|"name"|"function": name, "args"|"arguments"|...: {...}}``.

    Inside ``<tool_call>`` tags any object with a name is a call. In fences and bare text ``tool`` is always a
    call marker, but ``name`` / ``function`` only count with arguments (``arguments``, ``parameters``, ...) or when
    the name is a known tool: ``{"name": "pkg", "version": "1.0"}`` is package metadata, not a call.
    """
    if not isinstance(obj, dict):
        return None
    source = obj
    name_key = next((key for key in _NAME_KEYS if key in obj), None)
    if isinstance(obj.get("function"), dict):  # OpenAI shape: {"type": "function", "function": {"name", "arguments"}}
        source, name_key = obj["function"], "name"
    name = source.get(name_key) if name_key else None
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    arg_key = next((key for key in _ARG_KEYS if key in source), None)
    if context != "tag" and name_key not in ("tool", "tool_name"):
        if name not in index and arg_key not in _STRICT_ARG_KEYS:
            return None
    args = source.get(arg_key) if arg_key else None
    if args is None:
        args = {}
    elif isinstance(args, str):
        try:
            args = loads_lenient(args) if args.strip() else {}
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"tool": name, "args": args}


def _xml_call(name: str, body: str, index: Mapping[str, Mapping[str, str]]) -> dict | None:
    """One ``<function=NAME><parameter=KEY>value</parameter>...</function>`` block (Qwen3-Coder) as a call."""
    types = index.get(name.strip(), {})
    args = {
        match.group(1): _coerce(match.group(2), types.get(match.group(1)))
        for match in _XML_PARAMETER_RE.finditer(body)
    }
    return {"tool": name.strip(), "args": args} if name.strip() else None


def _coerce(raw: str, declared: str | None) -> Any:
    """An XML parameter's text as the value it stands for: by the schema's type if known, else by how it looks."""
    value = re.sub(r"\A\r?\n|\r?\n\Z", "", raw)  # the template's own newlines; \Z, since $ also matches before a final "\n"
    if declared == "string":
        return value
    looks_json = value.strip()[:1] in ("{", "[") or value.strip() in ("true", "false", "null") or re.fullmatch(
        r"-?\d+(\.\d+)?", value.strip()
    )
    if declared is not None or looks_json:
        try:
            return loads_lenient(value)
        except ValueError:
            pass
    return value


def _tool_index(tools: list[dict] | None) -> dict[str, dict[str, str]]:
    """``{tool name: {parameter: declared JSON type}}`` from OpenAI-format tool schemas."""
    index: dict[str, dict[str, str]] = {}
    for tool in tools or []:
        spec = tool.get("function", tool)
        name = spec.get("name")
        if isinstance(name, str) and name:
            properties = (spec.get("parameters") or {}).get("properties") or {}
            index[name] = {
                key: str(prop.get("type")) for key, prop in properties.items() if isinstance(prop, dict) and prop.get("type")
            }
    return index


def _overlaps(start: int, end: int, found: list[_Found]) -> bool:
    return any(start < item.end and item.start < end for item in found)


def _without_spans(text: str, found: list[_Found]) -> str:
    """``text`` with the spans in ``found`` cut out; what is left around them is joined by blank lines."""
    pieces, position = [], 0
    for item in found:
        pieces.append(text[position:item.start])
        position = item.end
    pieces.append(text[position:])
    return "\n\n".join(piece.strip() for piece in pieces if piece.strip())


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
