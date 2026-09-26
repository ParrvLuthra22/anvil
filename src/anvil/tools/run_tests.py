"""run_tests tool: run the repo's test suite using the detected test command."""

from __future__ import annotations

import re

from anvil.repo.profile import RepoProfile
from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_DEFAULT_TIMEOUT = 300   # 5 minutes for a full test suite
_FAILURE_LINES = 30      # max lines to include in the failure summary

# Patterns that flag a failure line in common test output formats
_FAILURE_RE = re.compile(
    r"(FAILED|ERROR|FAIL|AssertionError|Traceback|panic:|FAIL\t|--- FAIL|^not ok)",
    re.MULTILINE,
)


def _extract_failure_summary(output: str) -> str:
    """Return a compact summary of failure lines from test output."""
    lines = output.splitlines()
    failure_lines = [ln for ln in lines if _FAILURE_RE.search(ln)]
    if not failure_lines:
        return ""
    summary = failure_lines[:_FAILURE_LINES]
    if len(failure_lines) > _FAILURE_LINES:
        summary.append(f"... [{len(failure_lines) - _FAILURE_LINES} more failure lines omitted]")
    return "\n".join(summary)


class RunTestsTool:
    """Run the repository's test suite using the profile's detected test command."""

    name = "run_tests"
    description = (
        "Run the repository test suite. Uses the test command detected during profiling "
        "(e.g. 'pytest', 'go test ./...', 'cargo test'). "
        "An optional target narrows the scope (e.g. a test file or function name). "
        "Returns a failure summary on non-zero exit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "Optional test target to pass to the test runner "
                    "(e.g. 'tests/test_foo.py::test_bar' for pytest, "
                    "'./pkg/foo/...' for Go). "
                    "Leave empty to run all tests."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": f"Seconds before killing the test run. Defaults to {_DEFAULT_TIMEOUT}.",
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": [],
    }

    def __init__(self, profile: RepoProfile | None = None) -> None:
        """Optionally bind a *profile* at construction time for the test command."""
        self._profile = profile

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Run tests inside *sandbox*, optionally narrowed to *args['target']*."""
        target: str = args.get("target", "")
        timeout: int = int(args.get("timeout", _DEFAULT_TIMEOUT))

        # Resolve test command: prefer profile, allow fallback
        test_cmd: str | None = self._profile.test_cmd if self._profile else None
        if not test_cmd:
            # Try to auto-detect from sandbox root
            root = sandbox.root
            if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
                test_cmd = "pytest"
            elif (root / "go.mod").exists():
                test_cmd = "go test ./..."
            elif (root / "Cargo.toml").exists():
                test_cmd = "cargo test"
            elif (root / "package.json").exists():
                test_cmd = "npm test"
            else:
                return ToolResult(
                    ok=False,
                    output=(
                        "No test command is configured and none could be auto-detected. "
                        "Set RepoProfile.test_cmd or use run_cmd directly."
                    ),
                )

        cmd = f"{test_cmd} {target}".strip() if target else test_cmd

        result = sandbox.exec(cmd, timeout=timeout)

        # Combine output for analysis
        combined = "\n".join(filter(None, [result.stdout, result.stderr]))
        ok = result.exit_code == 0 and not result.timed_out

        output_parts: list[str] = []
        if result.timed_out:
            output_parts.append(f"[TIMED OUT after {timeout}s]")
        output_parts.append(f"[exit code: {result.exit_code}]")
        if result.stdout:
            output_parts.append(f"[stdout]\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        if not ok:
            summary = _extract_failure_summary(combined)
            if summary:
                output_parts.append(f"\n[failure summary]\n{summary}")

        return ToolResult(
            ok=ok,
            output="\n".join(output_parts),
            meta={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "test_cmd": cmd,
            },
        )
