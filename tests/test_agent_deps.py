"""Dependencies and the environment: installing them, routing commands to them, keeping them out of the patch,
and telling the model what it can rely on.

Audit findings covered: `ensure_deps` was never called, nothing put the venv on PATH, `.anvil_venv` was not excluded
from the patch, and the model had to guess whether `python` exists.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import anvil.agent.pipeline as pipeline_module
from anvil.agent.outputs import changed_files, filter_diff
from anvil.agent.pipeline import Ingested, RepoPipeline, _exclude_harness_dirs, _venv_dir
from anvil.agent.prepared_sandbox import PreparedSandbox
from anvil.agent.prompts import environment_note
from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile
from tests import real_repo as rr
from tests.fakes import FakePipeline, RecordingLLM
from tests.test_orchestrator import execute, happy

PY = RepoProfile(["python"], "python", "pip install -e .", "pytest", "pytest")


# ---- the patch never contains the harness's directories -------------------------------------


def _section(path: str, body: str = "+x = 1\n") -> str:
    return f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n{body}"


def test_the_dependency_venv_and_the_scratch_dir_are_excluded_from_the_patch():
    diff = (
        _section(".anvil_venv/lib/python3.11/site-packages/requests/api.py")
        + _section(".anvil_venv/bin/pytest")
        + _section(".anvil/repro.py")
        + "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    patch = filter_diff(diff)
    assert [c.path for c in changed_files(patch)] == ["calc.py"]
    assert ".anvil" not in patch


def test_only_the_harness_directories_at_the_top_level_are_excluded():
    diff = _section("src/.anvil_venv_notes.md") + _section("docs/anvil_venv/readme.md") + _section(".anvil_venv2/x.py")
    assert [c.path for c in changed_files(filter_diff(diff))] == [
        "src/.anvil_venv_notes.md", "docs/anvil_venv/readme.md", ".anvil_venv2/x.py",
    ]


# ---- git is told to ignore them (speed: one git subprocess per untracked file otherwise) ----


def _git(root: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env).stdout


@pytest.fixture
def clone(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.py").write_text("x = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "i")
    return root


def test_the_harness_dirs_are_added_to_the_clones_git_exclude(clone):
    _exclude_harness_dirs(clone)
    (clone / ".anvil_venv" / "lib").mkdir(parents=True)
    (clone / ".anvil_venv" / "lib" / "big.py").write_text("y = 2\n")
    (clone / ".anvil").mkdir()
    (clone / ".anvil" / "repro.py").write_text("z = 3\n")
    (clone / "real_new_file.py").write_text("w = 4\n")
    assert _git(clone, "status", "--porcelain").split() == ["??", "real_new_file.py"]
    assert _git(clone, "ls-files", "--others", "--exclude-standard").split() == ["real_new_file.py"]


def test_excluding_is_idempotent_and_keeps_what_was_there(clone):
    exclude = clone / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    exclude.write_text("# mine\n*.log")  # no trailing newline
    _exclude_harness_dirs(clone)
    _exclude_harness_dirs(clone)
    assert exclude.read_text().splitlines() == ["# mine", "*.log", ".anvil/", ".anvil_venv/"]


def test_a_directory_that_is_not_a_git_checkout_is_left_alone(tmp_path):
    _exclude_harness_dirs(tmp_path)
    assert not (tmp_path / ".git").exists()


def test_an_unwritable_exclude_file_is_not_fatal(clone):
    exclude = clone / ".git" / "info" / "exclude"
    exclude.unlink(missing_ok=True)
    exclude.mkdir(parents=True)  # a directory where the file should be: reading and writing both fail
    _exclude_harness_dirs(clone)
    assert exclude.is_dir()


def test_a_venv_full_of_files_does_not_slow_the_sandbox_diff_down(tmp_path, monkeypatch):
    """Without the exclude, WorktreeSandbox.diff() runs one `git diff --no-index` per untracked file."""
    rr.prepare(tmp_path, monkeypatch)
    pipeline = RepoPipeline({"sandbox": "worktree", "install_dependencies": False}, Path("out"))
    workspace = pipeline.profile(pipeline.ingest(rr.ISSUE_URL))
    try:
        root = workspace.sandbox.root
        for i in range(400):
            path = root / ".anvil_venv" / "lib" / f"m{i}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"v = {i}\n")
        started = time.monotonic()
        diff = workspace.sandbox.diff()
        assert diff == "" and time.monotonic() - started < 2.0
        ref = workspace.sandbox.checkpoint("x")
        assert ref and not ref.startswith("error")
        assert (root / ".anvil_venv" / "lib" / "m399.py").exists(), "a checkpoint must not sweep the venv away"
    finally:
        workspace.sandbox.close()


# ---- installing dependencies ----------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.notes: list[tuple[str, str]] = []

    def __call__(self, level, message):
        self.notes.append((level, message))

    def levels(self):
        return [level for level, _ in self.notes]


def make_pipeline(tmp_path, installer, config=None, notify=None):
    return RepoPipeline(
        config if config is not None else {}, tmp_path,
        sandbox_factory=lambda cfg, root, profile: FakeSandboxWithRoot(root),
        registry_factory=lambda profile: object(),
        deps_installer=installer, notify=notify,
    )


class FakeSandboxWithRoot:
    def __init__(self, root):
        self.root = root

    def exec(self, cmd, timeout=120):  # pragma: no cover - never used here
        raise AssertionError("not expected")


def profile_with(monkeypatch, tmp_path, installer, config=None):
    monkeypatch.setattr(pipeline_module, "profile_repo", lambda root: PY)
    monkeypatch.setattr(pipeline_module, "repo_map", lambda root: "map")
    notes = Recorder()
    pipeline = make_pipeline(tmp_path, installer, config, notes)
    root = tmp_path / "clone"
    root.mkdir(exist_ok=True)
    workspace = pipeline.profile(Ingested(IssueRef("a", "b", 1, "u"), root))
    return workspace, notes


def test_dependencies_are_installed_with_the_unwrapped_sandbox_and_the_venv_goes_on_path(tmp_path, monkeypatch):
    seen = {}

    def installer(sandbox, profile):
        seen["sandbox"], seen["profile"] = sandbox, profile
        return SimpleNamespace(ok=True, report="Python deps installed into .anvil_venv/.", venv_python=".anvil_venv/bin/python")

    workspace, notes = profile_with(monkeypatch, tmp_path, installer)
    assert isinstance(seen["sandbox"], FakeSandboxWithRoot) and seen["profile"] is PY
    assert isinstance(workspace.sandbox, PreparedSandbox) and workspace.sandbox.inner is seen["sandbox"]
    assert ".anvil_venv/bin" in workspace.sandbox.prepare("pytest")
    assert workspace.deps == "installed into .anvil_venv, which is first on PATH."
    assert notes.levels() == ["info", "info"] and "Installing dependencies (pip install -e .)" in notes.notes[0][1]
    assert "Python deps installed" in notes.notes[1][1]


def test_a_failed_install_is_a_warning_and_the_run_goes_on_without_the_venv(tmp_path, monkeypatch):
    def installer(sandbox, profile):
        return SimpleNamespace(ok=False, report="Install failed (exit 1).\nstderr: No matching distribution", venv_python=".anvil_venv/bin/python")

    workspace, notes = profile_with(monkeypatch, tmp_path, installer)
    assert notes.levels() == ["info", "warning"] and "No matching distribution" in notes.notes[1][1]
    assert ".anvil_venv" not in workspace.sandbox.prepare("pytest"), "a half-installed venv must not shadow the system tools"
    assert workspace.deps.startswith("NOT installed")


SKIP_REASON = "No install command detected — skipping."


def test_a_skipped_install_is_reported_as_skipped_with_its_reason_never_as_installed(tmp_path, monkeypatch):
    """Regression: ok=True, skipped=True (nothing to install) was announced as 'installed' to the user and the model."""
    workspace, notes = profile_with(
        monkeypatch, tmp_path, lambda s, p: SimpleNamespace(ok=True, report=SKIP_REASON, venv_python=None, skipped=True)
    )
    assert notes.levels() == ["info", "info"]
    assert notes.notes[1][1] == f"Dependencies skipped: {SKIP_REASON}"
    assert workspace.deps == f"skipped ({SKIP_REASON}); nothing was installed."
    assert "installed" not in notes.notes[1][1] and not workspace.deps.startswith("installed")
    assert ".anvil_venv" not in workspace.sandbox.prepare("pytest")


def test_a_missing_toolchain_is_skipped_not_a_failed_install(tmp_path, monkeypatch):
    reason = "npm not found on PATH — skipping JS dependency install."
    workspace, notes = profile_with(
        monkeypatch, tmp_path, lambda s, p: SimpleNamespace(ok=False, report=reason, venv_python=None, skipped=True)
    )
    assert notes.levels() == ["info", "warning"], "it is a warning: the tests probably cannot run"
    assert notes.notes[1][1] == f"Dependencies skipped: {reason}" and "failed" not in notes.notes[1][1]
    assert workspace.deps == f"NOT installed (skipped: {reason}): imports of third-party packages may fail; the tests may not run."


def test_a_multi_line_skip_reason_stays_on_one_line_for_the_model(tmp_path, monkeypatch):
    workspace, _ = profile_with(
        monkeypatch, tmp_path, lambda s, p: SimpleNamespace(ok=True, report="line one\n  line two\n", venv_python=None, skipped=True)
    )
    assert "\n" not in workspace.deps and "line one line two" in workspace.deps


def test_a_result_without_a_skipped_attribute_is_an_ordinary_install(tmp_path, monkeypatch):
    """Older installers return no `skipped`: they must keep working exactly as before."""
    workspace, notes = profile_with(
        monkeypatch, tmp_path, lambda s, p: SimpleNamespace(ok=True, report="Dependencies installed. (npm install)", venv_python=None)
    )
    assert workspace.deps == "installed." and notes.notes[1][1].startswith("Dependencies installed")


def test_an_installer_that_crashes_is_a_warning_not_a_failed_run(tmp_path, monkeypatch):
    def installer(sandbox, profile):
        raise RuntimeError("pip exploded")

    workspace, notes = profile_with(monkeypatch, tmp_path, installer)
    assert notes.levels() == ["info", "warning"] and "RuntimeError: pip exploded" in notes.notes[1][1]
    assert "installer crashed" in workspace.deps


def test_a_missing_installer_module_is_a_warning_not_a_failed_run(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "anvil.repo.deps", None)  # makes `from anvil.repo.deps import ...` raise ImportError
    workspace, notes = profile_with(monkeypatch, tmp_path, None)
    assert notes.levels() == ["warning"] and "not available" in notes.notes[0][1]
    assert "no installer" in workspace.deps


def test_the_config_can_switch_installation_off(tmp_path, monkeypatch):
    workspace, notes = profile_with(
        monkeypatch, tmp_path, lambda s, p: pytest.fail("must not install"), {"install_dependencies": False}
    )
    assert notes.notes == [] and "install_dependencies is off" in workspace.deps


def test_a_non_python_install_has_no_venv_to_put_on_path(tmp_path, monkeypatch):
    workspace, _ = profile_with(
        monkeypatch, tmp_path, lambda s, p: SimpleNamespace(ok=True, report="Dependencies installed. (npm install)", venv_python=None)
    )
    assert ".anvil_venv" not in workspace.sandbox.prepare("npm test") and workspace.deps == "installed."


@pytest.mark.parametrize(
    "venv_python, expected",
    [(".anvil_venv/bin/python", ".anvil_venv"), ("venv/bin/python3", "venv"), (None, None), ("", None)],
)
def test_the_venv_directory_is_derived_from_the_interpreter_path(venv_python, expected, tmp_path):
    assert _venv_dir(venv_python, FakeSandboxWithRoot(tmp_path)) == expected


def test_an_absolute_interpreter_path_is_made_relative_to_the_repository(tmp_path):
    sandbox = FakeSandboxWithRoot(tmp_path)
    assert _venv_dir(str(tmp_path / ".anvil_venv" / "bin" / "python"), sandbox) == ".anvil_venv"
    assert _venv_dir("/somewhere/else/bin/python", sandbox) is None


# ---- through the whole harness, real repository, real tools ---------------------------------


def shim_installer(sandbox, profile):
    """Stands in for pip: creates a venv whose `pytest` is recognisable, exactly where ensure_deps puts it."""
    bin_dir = sandbox.root / ".anvil_venv" / "bin"
    bin_dir.mkdir(parents=True)
    tool = bin_dir / "pytest"
    tool.write_text("#!/bin/sh\necho 'shim-pytest-from-the-dependency-venv: 1 passed'\nexit 0\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
    return SimpleNamespace(ok=True, report="Python deps installed into .anvil_venv/.", venv_python=".anvil_venv/bin/python")


def failing_installer(sandbox, profile):
    return SimpleNamespace(ok=False, report="Install failed (exit 1).", venv_python=None)


FULL_RUN = (
    rr.understand() + rr.localize() + rr.reproduce() + rr.good_patch() + rr.verify() + rr.review_ok() + rr.finalize()
)


def test_run_tests_uses_the_pytest_from_the_dependency_venv(tmp_path, monkeypatch):
    """Regression (audit): ensure_deps was never called, and nothing would have routed `pytest` to its venv anyway."""
    rr.prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_module, "_default_installer", lambda: shim_installer)
    run = rr.run_real(FULL_RUN, install_dependencies=True)

    assert any("shim-pytest-from-the-dependency-venv" in text for text in run.prompt_texts()), "run_tests must reach the venv"
    assert run.errors() == []
    assert ".anvil_venv" not in run.patch and ".anvil" not in run.patch and "+    return a + b" in run.patch
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)
    progress = [e.data["text"] for e in run.events if e.type == "message" and e.data["role"] == "system"]
    assert any(text.startswith("Installing dependencies") for text in progress)
    assert any("Python deps installed" in text for text in progress)
    system = run.llm.calls[0][0][0]["content"]
    assert "Dependencies: installed into .anvil_venv, which is first on PATH." in system


def test_a_skipped_install_is_described_to_the_model_as_skipped(tmp_path, monkeypatch):
    rr.prepare(tmp_path, monkeypatch)
    skipped = lambda sandbox, profile: SimpleNamespace(ok=True, report=SKIP_REASON, venv_python=None, skipped=True)  # noqa: E731
    monkeypatch.setattr(pipeline_module, "_default_installer", lambda: skipped)
    run = rr.run_real(FULL_RUN, install_dependencies=True)

    system = run.llm.calls[0][0][0]["content"]
    assert f"Dependencies: skipped ({SKIP_REASON}); nothing was installed." in system
    assert "installed into" not in system
    progress = [e.data["text"] for e in run.events if e.type == "message" and e.data["role"] == "system"]
    assert f"Dependencies skipped: {SKIP_REASON}" in progress and run.errors() == []


def test_a_failed_install_is_announced_as_a_deps_error_event_and_the_run_still_finishes(tmp_path, monkeypatch):
    rr.prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_module, "_default_installer", lambda: failing_installer)
    run = rr.run_real(FULL_RUN, install_dependencies=True)

    assert run.errors() == ["deps"]
    (event,) = [e for e in run.events if e.type == "error"]
    assert "Installing dependencies failed" in event.data["message"] and event.phase.value == "profile"
    assert "+    return a + b" in run.patch
    assert "Dependencies: NOT installed" in run.llm.calls[0][0][0]["content"]


def test_the_clone_of_a_real_run_ignores_the_harness_directories(tmp_path, monkeypatch):
    rr.prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline_module, "_default_installer", lambda: shim_installer)
    rr.run_real(FULL_RUN, install_dependencies=True)
    (clone,) = (tmp_path / "out" / "workspace").glob("acme__calc__1__*")
    exclude = (clone / ".git" / "info" / "exclude").read_text().splitlines()
    assert ".anvil/" in exclude and ".anvil_venv/" in exclude


# ---- telling the model about the machine ----------------------------------------------------


def test_the_note_names_the_interpreter_the_test_command_and_the_dependency_state():
    note = environment_note(PY, "installed into .anvil_venv, which is first on PATH.")
    assert "Primary language: python." in note and "`python3`" in note and "bare `python` may not exist" in note
    assert "sys.path.insert(0, '.')" in note
    assert "Test command: `pytest`" in note
    assert "Dependencies: installed into .anvil_venv, which is first on PATH." in note


@pytest.mark.parametrize(
    "language, test_cmd, toolchain",
    [("javascript", "npm test", "node"), ("typescript", "npm test", "node"), ("go", "go test ./...", "go"),
     ("rust", "cargo test", "cargo"), ("java", "mvn -q test", "java")],
)
def test_the_note_covers_the_other_languages(language, test_cmd, toolchain):
    note = environment_note(RepoProfile([language], language, None, test_cmd, None))
    assert f"Toolchain: `{toolchain}`" in note and f"Test command: `{test_cmd}`" in note
    assert "python3" not in note and "Dependencies:" not in note


def test_the_note_admits_it_when_no_test_command_was_found():
    note = environment_note(RepoProfile(["unknown"], "unknown", None, None, None))
    assert "No test command was detected" in note and "Toolchain" not in note


def test_every_phase_of_a_run_carries_the_note_in_its_system_prompt(tmp_path):
    run = execute(happy(), tmp_path)
    systems = [messages[0]["content"] for messages, _ in run.llm.calls]
    assert systems and all("Environment (facts about this machine" in text for text in systems)
    assert all("`python3`" in text and "Test command: `pytest`" in text for text in systems)
    assert not any("Dependencies:" in text for text in systems), "a pipeline that reports no dependency state adds no line"
