"""grep tool: search for a pattern using ripgrep (rg) or grep as fallback."""

from __future__ import annotations

import shutil
import subprocess

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_MAX_MATCHES = 50
_SEARCH_TIMEOUT = 30  # seconds


class GrepTool:
    """Search for a regex pattern in the repository using rg (or grep)."""

    name = "grep"
    description = (
        "Search for a regex pattern across files in the repository. "
        "Uses ripgrep (rg) if available, otherwise falls back to grep. "
        f"Returns at most {_MAX_MATCHES} matches in 'file:line:content' format."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory or file path relative to repo root to search within. "
                    "Defaults to the repo root ('.')."
                ),
                "default": ".",
            },
            "file_pattern": {
                "type": "string",
                "description": (
                    "Glob pattern to restrict which files are searched, "
                    "e.g. '*.py' or '*.{ts,js}'. Optional."
                ),
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search is case-sensitive. Defaults to true.",
                "default": True,
            },
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Run a grep search inside *sandbox*."""
        pattern: str = args.get("pattern", "")
        path: str = args.get("path", ".")
        file_pattern: str | None = args.get("file_pattern")
        case_sensitive: bool = args.get("case_sensitive", True)

        if not pattern:
            return ToolResult(ok=False, output="'pattern' argument is required.")

        search_path = sandbox.root / path
        if not search_path.exists():
            return ToolResult(ok=False, output=f"Search path does not exist: {path!r}")

        use_rg = shutil.which("rg") is not None

        if use_rg:
            cmd = ["rg", "--line-number", "--no-heading", "--color=never",
                   f"--max-count={_MAX_MATCHES}"]
            if not case_sensitive:
                cmd.append("--ignore-case")
            if file_pattern:
                cmd += ["--glob", file_pattern]
            cmd += [pattern, str(search_path)]
        else:
            cmd = ["grep", "-rn", "--include" if file_pattern else "-r"]
            if file_pattern:
                cmd += [f"--include={file_pattern}"]
            if not case_sensitive:
                cmd.append("-i")
            cmd += [pattern, str(search_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_SEARCH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output="Search timed out.")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output=f"Search error: {exc}")

        # rg exits 1 when no matches (not an error)
        if result.returncode not in (0, 1):
            return ToolResult(ok=False, output=f"Search failed: {result.stderr.strip()}")

        raw = result.stdout.strip()
        if not raw:
            return ToolResult(ok=True, output="No matches found.", meta={"match_count": 0})

        # Normalise paths to be relative to sandbox root
        root_str = str(sandbox.root) + "/"
        lines = [line.replace(root_str, "") for line in raw.splitlines()]

        # Cap at max matches
        truncated = len(lines) > _MAX_MATCHES
        lines = lines[:_MAX_MATCHES]
        output = "\n".join(lines)
        if truncated:
            output += f"\n... [results capped at {_MAX_MATCHES} matches]"

        return ToolResult(
            ok=True,
            output=output,
            meta={"match_count": len(lines), "tool": "rg" if use_rg else "grep"},
        )
