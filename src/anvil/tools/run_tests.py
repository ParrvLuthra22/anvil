"""run_tests tool: run the repo's test suite using the detected test command."""

from __future__ import annotations

import re

from anvil.repo.profile import RepoProfile
from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_DEFAULT_TIMEOUT = 300   # 5 minutes for a full test suite
_MAX_OUTPUT_CHARS = 8000  # hard cap on what we return to the model

# ---------------------------------------------------------------------------
# Failure extraction — per-framework patterns
# ---------------------------------------------------------------------------

# Lines that identify the name of a failing test
_FAIL_NAME_RE = re.compile(
    r"(?:"
    r"FAILED\s+(?P<pytest>[\w/.\-]+(?:::[\w.\-]+)+)"          # pytest: FAILED path::name
    r"|--- FAIL:\s*(?P<go>[\w./]+)"                            # Go: --- FAIL: TestFoo
    r"|test (?P<rust>[\w:]+)\s+\.\.\. FAILED"                  # Rust: test foo::bar ... FAILED
    r"|^\s+\d+\)\s+(?P<junit>[\w .]+)$"                       # JUnit: 1) Some test name
    r"|not ok \d+ - (?P<tap>.*)"                               # TAP: not ok 1 - name
    r")",
    re.MULTILINE,
)

# Lines that carry the first failure message / assertion
_ASSERT_RE = re.compile(
    r"(?:"
    r"AssertionError[:\s]+(?P<msg>.+)"
    r"|assert .+$"
    r"|E\s{4,}(?P<pytest_e>.+)"                               # pytest inline E lines
    r"|FAIL\t[\w./]+\s+\[(?P<go_fail>[^\]]+)\]"
    r"|panicked at '(?P<rust_panic>[^']+)'"
    r")",
    re.MULTILINE,
)

_MAX_FAILURES = 10     # max individual failures to list
_MAX_ASSERTS  = 5      # max assertion messages to show
_FAILURE_LINE_CAP = 40 # max lines in fallback raw scan


def _extract_failure_summary(output: str) -> str:
    """Return a compact, structured failure summary from test runner output.

    Extracts:
    - Up to 10 failing test names (from FAILED / --- FAIL: / not ok lines)
    - Up to 5 assertion messages (AssertionError, pytest E-lines, panic messages)

    Falls back to a raw scan of lines containing FAILED/ERROR if patterns
    don't match (e.g. unusual test frameworks).
    """
    failing_tests: list[str] = []
    for m in _FAIL_NAME_RE.finditer(output):
        name = next((v for v in m.groupdict().values() if v), None)
        if name and name not in failing_tests:
            failing_tests.append(name)
        if len(failing_tests) >= _MAX_FAILURES:
            break

    assertions: list[str] = []
    for m in _ASSERT_RE.finditer(output):
        msg = next((v for v in m.groupdict().values() if v), None)
        if msg and msg.strip() not in assertions:
            assertions.append(msg.strip())
        if len(assertions) >= _MAX_ASSERTS:
            break

    parts: list[str] = []
    if failing_tests:
        parts.append("Failing tests:\n" + "\n".join(f"  • {t}" for t in failing_tests))
    if assertions:
        parts.append("First assertion(s):\n" + "\n".join(f"  » {a}" for a in assertions))

    if parts:
        return "\n".join(parts)

    # --- Fallback: raw line scan ---
    _FALLBACK_RE = re.compile(
        r"(FAILED|ERROR|FAIL|AssertionError|Traceback|panic:|--- FAIL|not ok)",
        re.MULTILINE,
    )
    lines = output.splitlines()
    raw = [ln for ln in lines if _FALLBACK_RE.search(ln)][:_FAILURE_LINE_CAP]
    if raw:
        extra = len([ln for ln in lines if _FALLBACK_RE.search(ln)]) - len(raw)
        if extra > 0:
            raw.append(f"... [{extra} more failure lines omitted]")
        return "\n".join(raw)

    return ""


def _cap(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    half = cap // 2
    omitted = len(text) - 2 * half
    return text[:half] + f"\n... [{omitted} chars omitted] ...\n" + text[-half:]


class RunTestsTool:
    """Run the repository's test suite using the profile's detected test command."""

    name = "run_tests"
    description = (
        "Run the repository test suite. Uses the test command detected during profiling "
        "(e.g. 'pytest', 'go test ./...', 'cargo test'). "
        "An optional target narrows the scope (e.g. a test file or function name). "
        "Returns failing test names and first assertion messages on non-zero exit."
    )
    parameters = {
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of specific test targets to run "
                    "(e.g. ['tests/test_foo.py::test_bar', 'tests/test_baz.py'] for pytest, "
                    "['./pkg/foo/...'] for Go). "
                    "Leave empty to run all tests."
                ),
            },
            "fast_fail": {
                "type": "boolean",
                "description": "If true, stops after the first test failure.",
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": f"Seconds before killing the test run. Defaults to {_DEFAULT_TIMEOUT}.",
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": [],
    }

    def __init__(self, profile: RepoProfile | None = None, char_cap: int = _MAX_OUTPUT_CHARS) -> None:
        """Optionally bind a *profile* at construction time for the test command.

        Args:
            profile:  The :class:`~anvil.repo.profile.RepoProfile` for the repo.
            char_cap: Maximum characters to return to the model. Defaults to 8 000.
        """
        self._profile = profile
        self._char_cap = char_cap

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Run tests inside *sandbox*, optionally narrowed to *args['targets']*."""
        targets: list[str] = args.get("targets", [])
        # Support fallback if old 'target' str is passed
        if not targets and args.get("target"):
            targets = [args["target"]]
        
        fast_fail: bool = args.get("fast_fail", False)
        timeout: int = int(args.get("timeout", _DEFAULT_TIMEOUT))

        # Resolve test command: prefer profile, allow fallback
        test_cmd: str | None = self._profile.test_cmd if self._profile else None
        if not test_cmd:
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

        # Apply fast-fail if requested
        flags = []
        if fast_fail:
            # We loosely map the flag to the detected test command
            if "pytest" in test_cmd or "django" in test_cmd.lower():
                flags.append("-x")
            elif "go test" in test_cmd:
                flags.append("-failfast")

        parts = [test_cmd] + flags + targets
        cmd = " ".join(parts).strip()

        result = sandbox.exec(cmd, timeout=timeout)

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

        raw = "\n".join(output_parts)
        return ToolResult(
            ok=ok,
            output=_cap(raw, self._char_cap),
            meta={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "test_cmd": cmd,
            },
        )
