"""PreparedSandbox: how commands start (bytecode, stdin, PATH) and that everything else is left alone."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from anvil.agent.prepared_sandbox import PreparedSandbox
from anvil.sandbox.base import ExecResult


class LocalSandbox:
    """The smallest real Sandbox: commands run through a real shell in ``root``.

    stdin is a pipe that is held open and never written to, which is how a command sees the terminal under the TUI.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files_written: list[str] = []
        self.closed = False

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        start = time.monotonic()
        read_end, write_end = os.pipe()
        try:
            proc = subprocess.Popen(
                cmd, shell=True, cwd=self.root, stdin=read_end, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
                return ExecResult(proc.returncode, out, err, False, time.monotonic() - start)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                out, err = proc.communicate()
                return ExecResult(-1, out, err, True, time.monotonic() - start)
        finally:
            os.close(read_end)
            os.close(write_end)

    def read_file(self, path, start=None, end=None):
        return (self.root / path).read_text()

    def write_file(self, path, content):
        self.files_written.append(path)
        (self.root / path).write_text(content)

    def diff(self):
        return "the diff"

    def checkpoint(self, label):
        return f"ref-{label}"

    def rollback(self, ref):
        self.rolled_back = ref

    def close(self):
        self.closed = True


@pytest.fixture
def local(tmp_path):
    return LocalSandbox(tmp_path)


# ---- stale bytecode -------------------------------------------------------------------------


def _same_size_edit_then_import(sandbox, tmp_path: Path) -> str:
    """Write m.py, import it, rewrite it with a same-size edit and an identical mtime, import it again."""
    module = tmp_path / "m.py"
    for value in ("1", "2"):
        module.write_text(f"X = {value}\n")
        os.utime(module, (1_700_000_000, 1_700_000_000))  # the whole-second mtime a fast edit would share
        result = sandbox.exec(f"{sys.executable} -c 'import m; print(m.X)'")
    return result.stdout.strip()


def test_a_plain_sandbox_runs_stale_bytecode_after_a_same_size_edit(local, tmp_path):
    """Documents the hazard: this is what the harness did before, and why the repro kept failing for a right patch."""
    assert _same_size_edit_then_import(local, tmp_path) == "1"


def test_a_prepared_sandbox_always_runs_the_code_as_it_is_now(local, tmp_path):
    """Regression (audit): a same-size fix within the second of the previous run was invisible to the repro."""
    assert _same_size_edit_then_import(PreparedSandbox(local), tmp_path) == "2"
    assert not list(tmp_path.rglob("*.pyc")), "and it leaves no bytecode behind to end up in a diff"


# ---- stdin ----------------------------------------------------------------------------------


def test_a_command_that_reads_stdin_returns_instead_of_blocking_on_the_terminal(local):
    """Regression (audit): `cat` under a TTY-like stdin blocked until its timeout."""
    result = PreparedSandbox(local).exec("cat; echo done", timeout=5)
    assert result.timed_out is False and result.stdout.strip() == "done" and result.exit_code == 0
    assert local.exec("cat", timeout=1).timed_out, "without the wrapper the same command hangs (the hazard being tested)"


# ---- exit codes, output and syntax survive the wrapping -------------------------------------


def test_exit_codes_stdout_and_stderr_pass_through(local):
    prepared = PreparedSandbox(local)
    assert prepared.exec("echo out; echo err >&2; exit 7").__dict__ | {"duration": 0} == {
        "exit_code": 7, "stdout": "out\n", "stderr": "err\n", "timed_out": False, "duration": 0
    }


def test_a_trailing_comment_and_a_multi_line_command_still_work(local):
    prepared = PreparedSandbox(local)
    assert prepared.exec("echo one # a comment").stdout == "one\n"
    assert prepared.exec("echo a\necho b").stdout == "a\nb\n"
    assert prepared.exec("cd /tmp && pwd").stdout.strip() in ("/tmp", "/private/tmp")


def test_the_timeout_reaches_the_inner_sandbox(local):
    assert PreparedSandbox(local).exec("sleep 5", timeout=1).timed_out


def test_commands_run_in_the_repository_root(local, tmp_path):
    assert Path(PreparedSandbox(local).exec("pwd").stdout.strip()).resolve() == tmp_path.resolve()
    (tmp_path / "marker.txt").write_text("hi")
    assert PreparedSandbox(local).exec("cat marker.txt").stdout == "hi"


# ---- the venv on PATH -----------------------------------------------------------------------


def _fake_venv(root: Path, name: str = ".anvil_venv") -> None:
    bin_dir = root / name / "bin"
    bin_dir.mkdir(parents=True)
    tool = bin_dir / "venvtool"
    tool.write_text("#!/bin/sh\necho from-the-venv\n")
    tool.chmod(0o755)


def test_the_dependency_venv_comes_first_on_path(local, tmp_path):
    _fake_venv(tmp_path)
    assert "command not found" in PreparedSandbox(local).exec("venvtool").stderr
    prepared = PreparedSandbox(local, venv_dir=".anvil_venv")
    assert prepared.exec("venvtool").stdout == "from-the-venv\n"
    assert prepared.exec("echo $VIRTUAL_ENV").stdout.strip().endswith("/.anvil_venv")
    assert prepared.exec("command -v venvtool").stdout.strip().endswith("/.anvil_venv/bin/venvtool")


def test_the_venv_dir_is_quoted(local, tmp_path):
    _fake_venv(tmp_path, "my venv")
    assert PreparedSandbox(local, venv_dir="my venv").exec("venvtool").stdout == "from-the-venv\n"


def test_the_venv_is_looked_up_relative_to_the_repository_root_not_the_process_cwd(local, tmp_path, monkeypatch):
    _fake_venv(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert PreparedSandbox(local, venv_dir=".anvil_venv").exec("venvtool").stdout == "from-the-venv\n"


# ---- everything that is not a command is delegated ------------------------------------------


def test_files_diffs_checkpoints_and_close_are_delegated_untouched(local, tmp_path):
    prepared = PreparedSandbox(local)
    assert prepared.root == tmp_path
    prepared.write_file("a.txt", "x")
    assert prepared.read_file("a.txt") == "x" and local.files_written == ["a.txt"]
    assert prepared.diff() == "the diff" and prepared.checkpoint("s") == "ref-s"
    prepared.rollback("ref-s")
    assert local.rolled_back == "ref-s"
    prepared.close()
    assert local.closed
