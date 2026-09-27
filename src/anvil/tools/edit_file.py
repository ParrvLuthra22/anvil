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

        # Warn if old and new are identical when stripped (whitespace-only diff)
        whitespace_only = (old.strip() == new.strip() and old != new)

        count = content.count(old)
        updated = ""

        if count == 1:
            updated = content.replace(old, new, 1)
        elif count == 0:
            # Fallback 1: try a match ignoring leading whitespace per line
            def _strip_leading(s: str) -> str:
                return "\n".join(line.lstrip() for line in s.splitlines())

            old_stripped = _strip_leading(old)
            # Find all potential matches in content that map to old_stripped
            import re
            
            lines = content.splitlines(keepends=True)
            # We want to find a sequence of lines whose lstripped version matches old_stripped.
            # To be safe, we will just find instances of old_stripped in a stripped version of content,
            # but that destroys the original spacing. Let's do a line-by-line sliding window.
            old_lines_stripped = old.splitlines()
            old_lines_stripped_norm = [l.lstrip() for l in old_lines_stripped]
            
            matches = []
            for i in range(len(lines) - len(old_lines_stripped_norm) + 1):
                window = lines[i:i + len(old_lines_stripped_norm)]
                window_stripped = [l.lstrip().rstrip("\n\r") for l in window]
                # Compare without trailing newlines just for matching
                old_cmp = [l.rstrip("\n\r") for l in old_lines_stripped_norm]
                if window_stripped == old_cmp:
                    matches.append((i, i + len(old_lines_stripped_norm), window))

            if len(matches) == 1:
                # Exactly one match ignoring leading whitespace. Re-indent new_string.
                start_idx, end_idx, matched_lines = matches[0]
                
                # Determine original indentation of the first line of the block
                first_line = matched_lines[0]
                indent = first_line[:len(first_line) - len(first_line.lstrip())]
                
                # Re-indent new
                new_lines = new.splitlines(keepends=True)
                reindented_new = []
                for j, nl in enumerate(new_lines):
                    # Only add indent if the new line doesn't already have it
                    # But actually, the instruction says "re-indent new_string to the file's indentation".
                    # Let's see how much indent to add. If old was dedented, new is dedented.
                    # We just prepend `indent` if it's relative, but it's simpler to assume new has same relative indentation.
                    # Let's calculate the delta between `indent` and the first line of `old`
                    old_first = old_lines_stripped[0] if old_lines_stripped else ""
                    old_indent_str = old_first[:len(old_first) - len(old_first.lstrip())]
                    
                    if nl.startswith(old_indent_str):
                        reindented_new.append(indent + nl[len(old_indent_str):])
                    else:
                        reindented_new.append(indent + nl)
                
                new_block = "".join(reindented_new)
                
                # Replace the matched lines in content
                lines[start_idx:end_idx] = [new_block]
                updated = "".join(lines)
            else:
                # 0 or multiple fuzzy matches
                close = self._closest_actual_block(old, content)
                hint = f"\nClosest existing block (with original indentation):\n{close}" if close else ""
                
                msg = f"String not found in {path!r}." if len(matches) == 0 else f"String found {len(matches)} times (ignoring indentation) in {path!r} — edit is ambiguous."
                return ToolResult(
                    ok=False,
                    output=f"{msg}{hint}",
                    meta={"match_count": 0},
                )
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
        if whitespace_only:
            msg += (
                " ⚠ Warning: old and new differ only in whitespace — "
                "verify this is the intended change."
            )
        return ToolResult(
            ok=True,
            output=msg,
            meta={"path": path, "match_count": 1, "whitespace_only": whitespace_only},
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
