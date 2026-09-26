"""list_dir tool: list the contents of a directory inside the sandbox."""

from __future__ import annotations

import os
from pathlib import Path

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult


class ListDirTool:
    """List files and directories at a given path inside the sandbox."""

    name = "list_dir"
    description = (
        "List the contents of a directory in the repository. "
        "Returns file names, sizes, and whether each entry is a file or directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory path relative to the repository root. "
                    "Use '.' for the root itself."
                ),
            }
        },
        "required": ["path"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """List *args['path']* inside *sandbox*.

        Returns a formatted directory listing or an error ToolResult on failure.
        """
        path: str = args.get("path", ".")
        try:
            abs_path = (sandbox.root / path).resolve()
            # Path-safety: must stay inside sandbox root
            abs_path.relative_to(sandbox.root.resolve())
        except (ValueError, TypeError):
            return ToolResult(ok=False, output=f"Path {path!r} is outside the sandbox root.")

        if not abs_path.exists():
            return ToolResult(ok=False, output=f"Path does not exist: {path!r}")

        if abs_path.is_file():
            stat = abs_path.stat()
            return ToolResult(ok=True, output=f"[file] {path}  ({stat.st_size} bytes)")

        try:
            entries = sorted(abs_path.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError as exc:
            return ToolResult(ok=False, output=f"Permission denied: {exc}")

        if not entries:
            return ToolResult(ok=True, output=f"{path}/  (empty directory)")

        lines: list[str] = [f"{path}/"]
        for entry in entries:
            try:
                stat = entry.stat()
                if entry.is_dir():
                    lines.append(f"  d  {entry.name}/")
                else:
                    lines.append(f"  f  {entry.name}  ({stat.st_size} bytes)")
            except OSError:
                lines.append(f"  ?  {entry.name}  (stat failed)")

        return ToolResult(ok=True, output="\n".join(lines))
