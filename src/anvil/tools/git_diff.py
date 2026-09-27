"""git_diff tool: return the current diff versus the sandbox baseline."""

from __future__ import annotations

import shlex

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_MAX_DIFF_CHARS = 20_000
_EXCLUDED_DIFF_PARTS = {".anvil", ".anvil_venv", "__pycache__"}


def _excluded_diff_path(path: str) -> bool:
    """Whether a repository path is a generated or harness-owned artifact."""
    parts = path.replace("\\", "/").split("/")
    return any(
        part in _EXCLUDED_DIFF_PARTS
        or part.endswith(".egg-info")
        or part.endswith((".pyc", ".pyo"))
        or part.startswith(".probe_")
        for part in parts
    )


def _filter_excluded_diff(diff: str) -> str:
    """Remove git-diff sections for artifacts excluded by the sandbox."""
    output: list[str] = []
    section: list[str] = []

    def keep_section(lines: list[str]) -> None:
        if not lines:
            return
        try:
            header = shlex.split(lines[0].removeprefix("diff --git "))
        except ValueError:
            output.extend(lines)
            return
        paths = [path[2:] for path in header if path.startswith(("a/", "b/"))]
        if not any(_excluded_diff_path(path) for path in paths):
            output.extend(lines)

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            keep_section(section)
            section = [line]
        elif section:
            section.append(line)
        else:
            output.append(line)
    keep_section(section)
    return "".join(output)


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
        """Delegate to *sandbox.diff()*, filter generated files, and cap output."""
        try:
            diff = _filter_excluded_diff(sandbox.diff())
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
