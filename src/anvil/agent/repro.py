"""The one tool the agent layer contributes itself: writing a repro script under ``.anvil/``.

The tool contract has no way to create a file (``edit_file`` only replaces text in
an existing one), so the REPRODUCE phase needs this. It can only write inside
``.anvil/``, which ``outputs.filter_diff`` keeps out of the final patch.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from anvil.agent.outputs import SCRATCH_DIR
from anvil.sandbox.base import Sandbox
from anvil.tools.base import ToolResult


class WriteReproTool:
    """Create or overwrite a repro script inside the sandbox's ``.anvil/`` directory."""

    name = "write_repro"
    description = (
        "Create or overwrite a repro script under .anvil/ (excluded from the final patch). "
        "`path` is relative to .anvil/, e.g. 'repro.py'. Run the script afterwards with run_cmd."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Script path relative to .anvil/, e.g. 'repro.py'."},
            "content": {"type": "string", "description": "Full contents of the script."},
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Write the script; failures come back as ``ok=False`` results, never exceptions."""
        raw, content = args.get("path"), args.get("content")
        if not isinstance(raw, str) or not isinstance(content, str):
            return ToolResult(ok=False, output="Both 'path' and 'content' are required strings.")
        path = scratch_path(raw)
        if path is None:
            return ToolResult(ok=False, output=f"Invalid path {raw!r}: it must be a relative file path inside {SCRATCH_DIR}/.")
        try:
            sandbox.write_file(path, content)
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, output=f"Could not write {path}: {exc}")
        return ToolResult(ok=True, output=f"Wrote {len(content)} characters to {path}.", meta={"path": path})


def scratch_path(raw: str) -> str | None:
    """Normalise ``raw`` to a repo-relative path inside ``.anvil/``; ``None`` if it would escape."""
    path = PurePosixPath(raw.strip().replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = path.parts if path.parts[:1] == (SCRATCH_DIR,) else (SCRATCH_DIR, *path.parts)
    return "/".join(parts) if len(parts) > 1 else None
