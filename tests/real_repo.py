"""A real upstream git repository, a real clone of it, the real worktree sandbox and the real tools.

Only two things are stand-ins: the GitHub API (``fetch_issue``) and the model. ``https://github.com/acme/calc.git``
is redirected to a local repository with git's own ``url.<base>.insteadOf``, so ``clone_repo`` runs unmodified.
Tests that use this run real ``git``, ``python3`` and ``pytest`` subprocesses (about a second each).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from anvil.agent.orchestrator import run_harness
from anvil.events import AgentEvent, EventBus
from tests.fakes import RecordingLLM, call, done, give_up, reply

ISSUE_URL = "https://github.com/acme/calc/issues/1"
REPO_URL = "https://github.com/acme/calc"
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "anvil-test",
    "GIT_AUTHOR_EMAIL": "anvil-test@example.invalid",
    "GIT_COMMITTER_NAME": "anvil-test",
    "GIT_COMMITTER_EMAIL": "anvil-test@example.invalid",
}

BUGGY = "def add(a, b):\n    return a - b\n"
TOY_FILES = {
    "calc.py": BUGGY,
    "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    "pyproject.toml": '[project]\nname = "calc"\nversion = "0.1"\n\n[tool.pytest.ini_options]\npythonpath = ["."]\n',
}
REPRO = "import sys\nsys.path.insert(0, '.')\nfrom calc import add\nassert add(2, 3) == 5, f'add(2, 3) == {add(2, 3)}'\n"
REPRO_CMD = "python3 .anvil/repro.py"
FIX = {"path": "calc.py", "old": "return a - b", "new": "return a + b"}


@dataclass
class RealRun:
    events: list[AgentEvent]
    llm: RecordingLLM
    out: Path

    @property
    def patch(self) -> str:
        return (self.out / "patch.diff").read_text()

    @property
    def report(self) -> str:
        return (self.out / "report.md").read_text()

    @property
    def done(self) -> AgentEvent:
        (event,) = [e for e in self.events if e.type == "done"]
        return event

    def errors(self) -> list[str]:
        return [e.data["kind"] for e in self.events if e.type == "error"]

    def prompt_texts(self) -> list[str]:
        return [m.get("content") or "" for messages, _ in self.llm.calls for m in messages]


def upstream(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Create the toy upstream repository under ``tmp_path`` and return its path."""
    root = tmp_path / "upstream" / "calc"
    for name, text in (files or TOY_FILES).items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    env = {**os.environ, **_GIT_IDENTITY}
    for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=root, check=True, capture_output=True, env=env)
    return root


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, str] | None = None, *, fetch=True) -> Path:
    """Work in ``tmp_path`` (so a relative ``output_dir`` really is relative) against a local stand-in for GitHub."""
    root = upstream(tmp_path, files)
    monkeypatch.chdir(tmp_path)
    for key, value in _GIT_IDENTITY.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", f"url.file://{root}.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/acme/calc.git")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    monkeypatch.setenv("PATH", os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    if fetch:
        monkeypatch.setattr(
            "anvil.agent.pipeline.fetch_issue",
            lambda ref: replace(ref, title="add() returns the wrong sum", body="add(2, 3) returns -1 instead of 5."),
        )
    return root


def run_real(script, *, issue_url: str = ISSUE_URL, repo_url=None, issue_text=None, llm=None, **config) -> RealRun:
    """Run the harness for the toy issue with the default ``RepoPipeline`` and a *relative* ``output_dir``."""
    settings = {
        "model": "m", "base_url": "http://provider.invalid/v1", "tool_mode": "native",
        "sandbox": "worktree", "output_dir": "out", "install_dependencies": False, **config,
    }
    bus = EventBus()
    queue = bus.subscribe()
    llm = llm or RecordingLLM(script)
    extra = {k: v for k, v in (("repo_url", repo_url), ("issue_text", issue_text)) if v is not None}
    run_harness(issue_url, settings, bus, llm=llm, **extra)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return RealRun(events, llm, Path(settings["output_dir"]))


# ---- scripted phases (each list is the model's replies for one phase) -----------------------


def understand():
    return [done("add() subtracts instead of adding")]


def localize():
    return [reply(call("grep", pattern="def add")), reply(call("read_file", path="calc.py")), done("calc.py:2 uses a - b")]


def reproduce():
    return [
        reply(call("write_repro", path="repro.py", content=REPRO)),
        reply(call("run_cmd", cmd=REPRO_CMD)),
        done("the repro fails", repro_cmd=REPRO_CMD),
    ]


def good_patch(**edit):
    return [reply(call("edit_file", **{**FIX, **edit})), reply(call("run_cmd", cmd=REPRO_CMD)), done("changed - to +")]


def wrong_patch(new="return a * b"):
    return [reply(call("edit_file", path="calc.py", old="return a - b", new=new)), done(f"used {new!r}")]


def verify():
    return [reply(call("run_tests")), done("repro and tests pass")]


def review_ok():
    return [reply(call("git_diff")), done("minimal and correct")]


def review_changes():
    return [reply(call("git_diff")), give_up("Add a docstring to add().")]


def rework():
    docstring = {"path": "calc.py", "old": "def add(a, b):", "new": 'def add(a, b):\n    """Add two numbers."""'}
    return [reply(call("edit_file", **docstring)), reply(call("run_cmd", cmd=REPRO_CMD)), done("added a docstring")]


def finalize():
    return [done("Fixed the operator in add().")]
