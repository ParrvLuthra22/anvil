"""Offline doubles for the agent tests: sandbox, tools, registry, recording LLM and script helpers.

The FakeSandbox follows the Sandbox contract literally: ``checkpoint`` snapshots
without touching the tree and a ref stays valid after ``rollback``.

``project_exec`` simulates a tiny repo whose ``add()`` subtracts. Running its
repro script or its tests reacts to the files actually present in the sandbox, so
an edit that fixes the bug makes them pass and any other edit leaves them failing.
"""

from __future__ import annotations

import copy
import difflib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from anvil.agent.pipeline import Ingested, Workspace
from anvil.llm.client import LLMResponse
from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile
from anvil.sandbox.base import ExecResult
from anvil.tools.base import ToolResult
from tests.mock_llm import MockLLM

# ---- sandbox --------------------------------------------------------------------------------


class FakeSandbox:
    """In-memory ``Sandbox``; ``events`` records checkpoints, rollbacks and close for assertions."""

    def __init__(
        self,
        files: dict[str, str],
        on_exec: Callable[[str, dict[str, str]], ExecResult] | None = None,
    ) -> None:
        self.root = Path("/fake/repo")
        self.files = dict(files)
        self._baseline = dict(files)
        self._on_exec = on_exec
        self._snapshots: dict[str, dict[str, str]] = {}
        self.exec_log: list[str] = []
        self.events: list[tuple] = []
        self.closed = False

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        self.exec_log.append(cmd)
        if self._on_exec is None:
            return ExecResult(0, "", "", False, 0.0)
        return self._on_exec(cmd, self.files)

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        text = self.files[path]
        if start is None and end is None:
            return text
        return "".join(text.splitlines(keepends=True)[(start or 1) - 1 : end])

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content

    def diff(self) -> str:
        sections = []
        for path in sorted(set(self._baseline) | set(self.files)):
            old, new = self._baseline.get(path), self.files.get(path)
            if old == new:
                continue
            body = difflib.unified_diff(
                (old or "").splitlines(keepends=True),
                (new or "").splitlines(keepends=True),
                fromfile=f"a/{path}" if old is not None else "/dev/null",
                tofile=f"b/{path}" if new is not None else "/dev/null",
            )
            header = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n" if old is None else "")
            sections.append(header + "".join(body))
        return "".join(sections)

    def checkpoint(self, label: str) -> str:
        ref = f"ckpt-{len(self._snapshots) + 1}"
        self._snapshots[ref] = dict(self.files)
        self.events.append(("checkpoint", label, ref))
        return ref

    def rollback(self, ref: str) -> None:
        self.files = dict(self._snapshots[ref])
        self.events.append(("rollback", ref))

    def close(self) -> None:
        self.closed = True
        self.events.append(("close",))


# ---- tools ----------------------------------------------------------------------------------


@dataclass
class FakeTool:
    """A ``Tool`` whose behaviour is a plain function of (args, sandbox)."""

    name: str
    fn: Callable[[dict, FakeSandbox], ToolResult]
    description: str = "fake tool"
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})

    def run(self, args: dict, sandbox: FakeSandbox) -> ToolResult:
        return self.fn(args, sandbox)


def _list_dir(args: dict, sb: FakeSandbox) -> ToolResult:
    return ToolResult(True, "\n".join(sorted(sb.files)))


def _grep(args: dict, sb: FakeSandbox) -> ToolResult:
    pattern = str(args.get("pattern", ""))
    hits = [
        f"{path}:{n}:{line}"
        for path, text in sorted(sb.files.items())
        for n, line in enumerate(text.splitlines(), 1)
        if pattern and pattern in line
    ]
    return ToolResult(True, "\n".join(hits) or "(no matches)")


def _read_file(args: dict, sb: FakeSandbox) -> ToolResult:
    try:
        return ToolResult(True, sb.read_file(str(args.get("path", "")), args.get("start"), args.get("end")))
    except FileNotFoundError:
        return ToolResult(False, f"File not found: {args.get('path')!r}")


def _edit_file(args: dict, sb: FakeSandbox) -> ToolResult:
    path, old, new = str(args.get("path", "")), str(args.get("old", "")), str(args.get("new", ""))
    if path not in sb.files:
        return ToolResult(False, f"File not found: {path!r}")
    if not old or sb.files[path].count(old) != 1:
        return ToolResult(False, f"String not found exactly once in {path!r}.")
    sb.write_file(path, sb.files[path].replace(old, new, 1))
    return ToolResult(True, f"Replaced 1 occurrence in {path!r}.", {"path": path})


def _run_cmd(args: dict, sb: FakeSandbox) -> ToolResult:
    result = sb.exec(str(args.get("cmd", "")))
    return ToolResult(result.exit_code == 0, result.stdout + result.stderr, {"exit_code": result.exit_code})


def _run_tests(args: dict, sb: FakeSandbox) -> ToolResult:
    result = sb.exec(f"pytest {args.get('target', '')}".strip())
    return ToolResult(result.exit_code == 0, result.stdout + result.stderr, {"exit_code": result.exit_code})


def _git_diff(args: dict, sb: FakeSandbox) -> ToolResult:
    return ToolResult(True, sb.diff() or "(no changes)")


def default_tools() -> list[FakeTool]:
    """The seven contract tools, implemented over the FakeSandbox."""
    return [
        FakeTool("list_dir", _list_dir),
        FakeTool("grep", _grep),
        FakeTool("read_file", _read_file),
        FakeTool("edit_file", _edit_file),
        FakeTool("run_cmd", _run_cmd),
        FakeTool("run_tests", _run_tests),
        FakeTool("git_diff", _git_diff),
    ]


def crashing(name: str, exc: Exception | None = None) -> FakeTool:
    """A tool that raises instead of returning a result (which real tools must never do)."""

    def boom(args: dict, sb: FakeSandbox) -> ToolResult:
        raise exc or RuntimeError("tool exploded")

    return FakeTool(name, boom)


class FakeRegistry:
    """Minimal ``ToolRegistry``: register / get / schemas, like the real one."""

    def __init__(self, tools: list[Any] | None = None) -> None:
        self._tools: dict[str, Any] = {}
        for tool in default_tools() if tools is None else tools:
            self.register(tool)

    def register(self, tool: Any) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Any:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values()
        ]


# ---- LLM scripting --------------------------------------------------------------------------


class RecordingLLM:
    """A ``MockLLM`` that also remembers every (messages, tools) it was called with."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._inner = MockLLM(script)
        self._total = len(script)
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append((copy.deepcopy(messages), tools))
        return self._inner.chat(messages, tools)

    @property
    def remaining(self) -> int:
        """Scripted responses not yet consumed."""
        return self._total - len(self.calls)


_call_ids = itertools.count(1)


def call(tool: str, **args: Any) -> dict:
    """One scripted tool call in the normalised shape the LLM client returns."""
    return {"id": f"call_{next(_call_ids)}", "tool": tool, "args": args}


def reply(*calls: dict, text: str = "", tokens: int = 100) -> LLMResponse:
    """A scripted model reply making ``calls``, reporting ``tokens`` total usage."""
    prompt = tokens * 4 // 5
    usage = {"prompt_tokens": prompt, "completion_tokens": tokens - prompt, "total_tokens": tokens}
    return LLMResponse(text=text, tool_calls=list(calls), usage=usage)


def done(summary: str = "ok", **extra: Any) -> LLMResponse:
    """A reply that ends the phase with ``phase_done``."""
    return reply(call("phase_done", summary=summary, **extra))


def give_up(reason: str = "cannot") -> LLMResponse:
    """A reply that ends the phase with ``give_up``."""
    return reply(call("give_up", reason=reason))


# ---- the simulated project ------------------------------------------------------------------

BUGGY = "def add(a, b):\n    return a - b\n"
REPRO_SCRIPT = "from calc import add\nassert add(2, 3) == 5, f'add(2, 3) == {add(2, 3)}'\n"
REPRO_CMD = "python .anvil/repro.py"
FIX = {"path": "calc.py", "old": "return a - b", "new": "return a + b"}


def project_files() -> dict[str, str]:
    return {
        "calc.py": BUGGY,
        "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    }


def project_exec(cmd: str, files: dict[str, str]) -> ExecResult:
    """Run the repro script or the tests against the current files of the simulated project."""
    fixed = "return a + b" in files["calc.py"]
    if cmd.startswith(REPRO_CMD):
        script = files.get(".anvil/repro.py")
        if script is None:
            return ExecResult(2, "", "python: can't open file '.anvil/repro.py'\n", False, 0.01)
        if "add(2, 3)" not in script:
            return ExecResult(0, "nothing checked\n", "", False, 0.01)
        if fixed:
            return ExecResult(0, "OK\n", "", False, 0.01)
        return ExecResult(1, "", "AssertionError: add(2, 3) == -1\n", False, 0.01)
    if cmd.startswith("pytest"):
        if fixed:
            return ExecResult(0, "1 passed\n", "", False, 0.1)
        return ExecResult(1, "1 failed\n", "FAILED tests/test_calc.py::test_add\n", False, 0.1)
    return ExecResult(127, "", f"{cmd}: command not found\n", False, 0.0)


def project_sandbox() -> FakeSandbox:
    return FakeSandbox(project_files(), on_exec=project_exec)


class StashStyleSandbox(FakeSandbox):
    """A sandbox that behaves like the git-stash based one: it is deliberately *not* contract-clean.

    ``checkpoint`` empties the working tree back to the baseline (untracked files such as the
    repro vanish until a rollback), and ``rollback`` consumes the ref (a second rollback to it
    just resets to the baseline). The orchestrator must keep the repro alive through both.
    """

    def checkpoint(self, label: str) -> str:
        ref = super().checkpoint(label)
        self.files = dict(self._baseline)
        return ref

    def rollback(self, ref: str) -> None:
        snapshot = self._snapshots.pop(ref, None)
        self.files = dict(snapshot if snapshot is not None else self._baseline)
        self.events.append(("rollback", ref))


# ---- pipeline -------------------------------------------------------------------------------

ISSUE_URL = "https://github.com/acme/calc/issues/7"


def project_issue() -> IssueRef:
    return IssueRef(
        "acme", "calc", 7, ISSUE_URL,
        title="add() returns the wrong sum", body="add(2, 3) returns -1 instead of 5.",
    )


class FakePipeline:
    """A ``Pipeline`` that hands out a prepared sandbox and registry; either step can be made to fail."""

    def __init__(
        self,
        sandbox: FakeSandbox | None = None,
        registry: FakeRegistry | None = None,
        *,
        fail_ingest: Exception | None = None,
        fail_profile: Exception | None = None,
    ) -> None:
        self.sandbox = sandbox or project_sandbox()
        self.registry = registry or FakeRegistry()
        self.fail_ingest, self.fail_profile = fail_ingest, fail_profile

    def ingest(self, issue_url: str) -> Ingested:
        if self.fail_ingest:
            raise self.fail_ingest
        return Ingested(project_issue(), self.sandbox.root)

    def profile(self, ingested: Ingested) -> Workspace:
        if self.fail_profile:
            raise self.fail_profile
        profile = RepoProfile(["python"], "python", "pip install -e .", "pytest", "pytest")
        return Workspace(ingested.issue, profile, "calc.py\ntests/test_calc.py", self.sandbox, self.registry)
