"""read_file tool: read a file from the sandbox with optional line range."""

from __future__ import annotations

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_MAX_OUTPUT_CHARS = 16_000   # cap before handing to the model


class ReadFileTool:
    """Read the contents of a file, with optional 1-indexed line range."""

    name = "read_file"
    description = (
        "Read a file from the repository. "
        "Optionally specify a start/end line (1-indexed, inclusive) to read a slice. "
        "Output lines are numbered for easy reference."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repository root.",
            },
            "start": {
                "type": "integer",
                "description": "First line to return (1-indexed, inclusive). Optional.",
            },
            "end": {
                "type": "integer",
                "description": "Last line to return (1-indexed, inclusive). Optional.",
            },
        },
        "required": ["path"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Read *args['path']* from *sandbox*, returning numbered lines."""
        path: str = args.get("path", "")
        start: int | None = args.get("start")
        end: int | None = args.get("end")

        if not path:
            return ToolResult(ok=False, output="'path' argument is required.")

        if start is None and end is None:
            try:
                size = (sandbox.root / path).stat().st_size
                if size > 64 * 1024:
                    return ToolResult(
                        ok=False,
                        output=(
                            f"File {path!r} is too large ({size} bytes) to read entirely. "
                            "Please specify 'start' and 'end' line ranges."
                        ),
                    )
            except OSError:
                pass

        try:
            content = sandbox.read_file(path, start=start, end=end)
        except PermissionError as exc:
            return ToolResult(ok=False, output=f"Access denied: {exc}")
        except FileNotFoundError:
            return ToolResult(ok=False, output=f"File not found: {path!r}")
        except OSError as exc:
            return ToolResult(ok=False, output=f"Could not read file: {exc}")

        lines = content.splitlines(keepends=True)
        offset = (start - 1) if start is not None else 0
        numbered = "".join(
            f"{offset + i + 1:6d}  {line}" for i, line in enumerate(lines)
        )

        if len(numbered) > _MAX_OUTPUT_CHARS:
            half = _MAX_OUTPUT_CHARS // 2
            omitted = len(numbered) - 2 * half
            numbered = (
                numbered[:half]
                + f"\n... [{omitted} chars omitted — use start/end to narrow the range] ...\n"
                + numbered[-half:]
            )

        return ToolResult(
            ok=True,
            output=numbered or "(empty file)",
            meta={"path": path, "lines": len(lines), "start": start, "end": end},
        )
