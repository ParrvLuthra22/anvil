"""edit_file tool: exact-string replace inside a sandbox file."""

from __future__ import annotations

import difflib

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_CLOSE_MATCH_LINES = 5   # how many close lines to show on mismatch


_MAX_SHOWN_LINES = 30  # lines of a tolerant edit's result shown back to the model


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _indent_unit(lines: list[str]) -> str:
    """One level of indentation as ``lines`` use it: a tab, or the number of spaces between their indentation levels."""
    indents = [_indent(line) for line in lines if line.strip()]
    if any("\t" in indent for indent in indents):
        return "\t"
    widths = sorted({len(indent) for indent in indents if indent})
    steps = [b - a for a, b in zip([0] + widths, widths)]
    return " " * (min(steps) if steps else 4)


def _in_units(extra: str, model_unit: str, file_unit: str) -> str:
    """``extra`` (indentation beyond the first line, in the model's unit) written in the file's unit."""
    if not extra or model_unit == file_unit:
        return extra
    if set(extra) == {"\t"}:
        levels, rest = len(extra), 0
    elif set(extra) == {" "} and model_unit != "\t":
        levels, rest = divmod(len(extra), len(model_unit))
    else:
        return extra
    return file_unit * levels + " " * rest


def _line_ending(line: str) -> str:
    return line[len(line.rstrip("\r\n")):]


def _lines_matching_ignoring_indentation(content: str, old: str) -> list[tuple[int, int]]:
    """Where in ``content`` the lines of ``old`` occur when leading whitespace is ignored, as ``(first, end)`` line indexes."""
    wanted = [line.lstrip() for line in old.splitlines()]  # (splitlines has already taken the line endings off)
    if not wanted:
        return []
    lines = content.splitlines(keepends=True)
    return [
        (i, i + len(wanted))
        for i in range(len(lines) - len(wanted) + 1)
        if [line.lstrip().rstrip("\r\n") for line in lines[i : i + len(wanted)]] == wanted
    ]


def _replace_lines_reindented(content: str, span: tuple[int, int], old: str, new: str) -> tuple[str, tuple[int, int, str]]:
    """Replace the lines ``span`` of ``content`` with ``new``, indented as the file indents them.

    Each line of ``new`` keeps its indentation relative to the first non-blank line of ``old`` (deeper stays deeper, shallower
    stays shallower) on top of the file's own indentation there, so a model that is two levels off is put right and a tab-indented
    file gets tabs. The file's line ending is used throughout, and the line break after the block is kept even when ``new``
    has none: the block replaces whole lines, and the next line must stay on its own line.
    """
    first, end = span
    lines = content.splitlines(keepends=True)
    matched = lines[first:end]
    old_lines = old.splitlines()
    reference = next((i for i, line in enumerate(old_lines) if line.strip()), 0)
    old_indent, file_indent = _indent(old_lines[reference]), _indent(matched[reference])
    eol = _line_ending(matched[0]) or _line_ending(matched[-1]) or "\n"
    model_unit, file_unit = _indent_unit(old_lines), _indent_unit(matched)

    block: list[str] = []
    for line in new.splitlines():
        if not line.strip():
            block.append("")
            continue
        indent = _indent(line)
        if indent.startswith(old_indent):
            extra = _in_units(indent[len(old_indent):], model_unit, file_unit)
            block.append(file_indent + extra + line[len(indent):])
        else:  # shallower than the reference line: as much shallower in the file
            keep = max(0, len(file_indent) - (len(old_indent) - len(indent)))
            block.append(file_indent[:keep] + line[len(indent):])
    text = eol.join(block)
    if block:
        text += eol if new.endswith(("\n", "\r")) else _line_ending(matched[-1])
    lines[first:end] = [text] if text else []
    updated = "".join(lines)

    now = updated.splitlines()[first : first + len(block)]
    shown = [f"{first + 1 + n:6d}  {line}" for n, line in enumerate(now[:_MAX_SHOWN_LINES])]
    if len(now) > _MAX_SHOWN_LINES:
        shown.append(f"        ... {len(now) - _MAX_SHOWN_LINES} more lines")
    return updated, (first + 1, first + max(len(block), 1), "\n".join(shown))


class EditFileTool:
    """Replace an exact string in a file with new content.

    Fails if the old string is not found OR if it appears more than once,
    preventing ambiguous edits.  On a not-found mismatch, returns the
    5 closest lines using difflib to help the model correct itself.
    """

    name = "edit_file"
    description = (
        "Replace an exact string in a file with new content. "
        "The match must be unique — fails if the string appears 0 or >1 times. "
        "On a not-found mismatch, up to 5 closest existing lines are returned "
        "to help correct the query."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repository root.",
            },
            "old": {
                "type": "string",
                "description": "The exact string to find and replace (must be unique in the file).",
            },
            "new": {
                "type": "string",
                "description": "The replacement string.",
            },
        },
        "required": ["path", "old", "new"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Apply an exact str_replace edit inside *sandbox*."""
        path: str = args.get("path", "")
        old: str = args.get("old", "")
        new: str = args.get("new", "")

        if not path:
            return ToolResult(ok=False, output="'path' argument is required.")
        if not old:
            return ToolResult(ok=False, output="'old' argument must be a non-empty string.")

        try:
            content = sandbox.read_file(path)
        except PermissionError as exc:
            return ToolResult(ok=False, output=f"Access denied: {exc}")
        except FileNotFoundError:
            return ToolResult(ok=False, output=f"File not found: {path!r}")
        except OSError as exc:
            return ToolResult(ok=False, output=f"Could not read file: {exc}")

        # Warn if old and new are identical when stripped (whitespace-only diff)
        whitespace_only = (old.strip() == new.strip() and old != new)

        count = content.count(old)
        updated = ""
        tolerant: tuple[int, int, str] | None = None  # (first line, last line, those lines as they now read) of a tolerant match

        if count == 1:
            updated = content.replace(old, new, 1)
        elif count == 0:
            # Fallback: the same lines with different indentation (tabs vs spaces, one level too deep or too shallow).
            spans = _lines_matching_ignoring_indentation(content, old)
            if len(spans) != 1:
                close = self._closest_actual_block(old, content)
                hint = f"\nClosest existing block (with original indentation):\n{close}" if close else ""
                msg = (
                    f"String not found in {path!r}."
                    if not spans
                    else f"String found {len(spans)} times (ignoring indentation) in {path!r} — edit is ambiguous."
                )
                return ToolResult(ok=False, output=f"{msg}{hint}", meta={"match_count": 0})
            updated, tolerant = _replace_lines_reindented(content, spans[0], old, new)
        else:
            return ToolResult(
                ok=False,
                output=(
                    f"String found {count} times in {path!r} — edit is ambiguous. "
                    "Add more surrounding context to make it unique."
                ),
                meta={"match_count": count},
            )

        try:
            sandbox.write_file(path, updated)
        except (PermissionError, OSError) as exc:
            return ToolResult(ok=False, output=f"Could not write file: {exc}")

        # Syntax guard
        err = self._check_syntax(path, updated, sandbox)
        if err:
            # Revert
            sandbox.write_file(path, content)
            return ToolResult(
                ok=False,
                output=f"Edit reverted due to syntax error:\n{err}",
            )

        msg = f"Successfully replaced 1 occurrence in {path!r}."
        if tolerant is not None:
            first, last, shown = tolerant
            msg += (
                f" Whitespace-tolerant match: your `old` differed from the file in indentation, so it was matched line by line "
                f"(lines {first}-{last}) and `new` was re-indented to fit. Those lines now read:\n{shown}"
            )
        if whitespace_only:
            msg += (
                " ⚠ Warning: old and new differ only in whitespace — "
                "verify this is the intended change."
            )
        return ToolResult(
            ok=True,
            output=msg,
            meta={
                "path": path,
                "match_count": 1,
                "whitespace_only": whitespace_only,
                "tolerant_match": tolerant is not None,
                **({"lines": [tolerant[0], tolerant[1]]} if tolerant is not None else {}),
            },
        )

    @staticmethod
    def _check_syntax(path: str, content: str, sandbox: Sandbox) -> str | None:
        """Run a fast syntax check. Return error message on failure, None otherwise."""
        if path.endswith(".py"):
            import ast
            try:
                ast.parse(content, filename=path)
            except SyntaxError as e:
                return f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}"
            return None
        elif path.endswith((".js", ".ts")):
            res = sandbox.exec(f"node --check '{path}'")
            if res.exit_code == 127 or not res.stderr.strip() and res.exit_code == 0:
                pass
            elif res.exit_code != 0:
                return res.stderr.strip() or res.stdout.strip()
        elif path.endswith(".go"):
            res = sandbox.exec(f"gofmt -e '{path}'")
            if res.exit_code == 127:
                pass
            elif res.exit_code != 0:
                return res.stderr.strip() or res.stdout.strip()
        return None

    @staticmethod
    def _closest_actual_block(query: str, content: str) -> str:
        """Return a block of ~15 lines from *content* most similar to *query*."""
        query_lines = query.splitlines()
        content_lines = content.splitlines()
        
        if not content_lines:
            return ""
            
        sm = difflib.SequenceMatcher(None, query, content)
        _, j, _ = sm.find_longest_match(0, len(query), 0, len(content))
        
        char_pos = j
        before = content[:char_pos].count("\n")
        
        # ~15 lines around the best match
        start = max(0, before - 7)
        end = min(len(content_lines), before + 8)
        
        # Format with line numbers for clarity if needed, or just plain lines.
        # The prompt says "ACTUAL block from the file around the best fuzzy match (with real indentation, up to ~15 lines)".
        matches = content_lines[start:end]
        return "\n".join(matches)
