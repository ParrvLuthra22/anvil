"""git_diff tool: return the current diff versus the sandbox baseline."""

from __future__ import annotations

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_MAX_DIFF_CHARS = 20_000


class GitDiffTool:
    """Return a unified diff of all changes made in the sandbox since baseline."""

    name = "git_diff"
    description = (
        "Return a unified diff of all changes made to the repository "
        "since the sandbox was created (the baseline commit). "
        "Includes modifications to existing files and newly created files."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Delegate to *sandbox.diff()* and cap output."""
        try:
            diff = sandbox.diff()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output=f"git diff failed: {exc}")

        if not diff or not diff.strip():
            return ToolResult(ok=True, output="(no changes)", meta={"changed": False})

        if len(diff) > _MAX_DIFF_CHARS:
            half = _MAX_DIFF_CHARS // 2
            omitted = len(diff) - 2 * half
            diff = (
                diff[:half]
                + f"\n... [{omitted} chars omitted] ...\n"
                + diff[-half:]
            )

        return ToolResult(ok=True, output=diff, meta={"changed": True})
