"""edit_file tool: exact-string replace inside a sandbox file."""

from __future__ import annotations

import difflib

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_CLOSE_MATCH_LINES = 5   # how many close lines to show on mismatch


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

        count = content.count(old)

        if count == 0:
            # Help the model by returning the closest matching lines
            close = self._closest_lines(old, content)
            hint = (
                f"\nClosest existing lines (for reference):\n{close}"
                if close else ""
            )
            return ToolResult(
                ok=False,
                output=f"String not found in {path!r}.{hint}",
                meta={"match_count": 0},
            )

        if count > 1:
            return ToolResult(
                ok=False,
                output=(
                    f"String found {count} times in {path!r} — edit is ambiguous. "
                    "Add more surrounding context to make it unique."
                ),
                meta={"match_count": count},
            )

        # Exactly one match — apply
        updated = content.replace(old, new, 1)
        try:
            sandbox.write_file(path, updated)
        except (PermissionError, OSError) as exc:
            return ToolResult(ok=False, output=f"Could not write file: {exc}")

        return ToolResult(
            ok=True,
            output=f"Successfully replaced 1 occurrence in {path!r}.",
            meta={"path": path, "match_count": 1},
        )

    @staticmethod
    def _closest_lines(query: str, content: str) -> str:
        """Return the *_CLOSE_MATCH_LINES* lines from *content* most similar to *query*."""
        # Use the first line of the query for matching heuristic
        query_first = query.splitlines()[0] if query.strip() else query
        content_lines = content.splitlines()
        matches = difflib.get_close_matches(
            query_first, content_lines, n=_CLOSE_MATCH_LINES, cutoff=0.3
        )
        if not matches:
            # Fall back to sequence matcher on the whole string
            sm = difflib.SequenceMatcher(None, query, content)
            _, j, _ = sm.find_longest_match(0, len(query), 0, len(content))
            # Return lines around the best match position
            char_pos = j
            before = content[:char_pos].count("\n")
            start = max(0, before - 2)
            end = min(len(content_lines), before + 3)
            matches = content_lines[start:end]
        return "\n".join(matches)
