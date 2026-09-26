"""Offline tests for repo/deps.py — all sandbox calls are faked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from anvil.repo.deps import DepsResult, ensure_deps, _INSTALL_TIMEOUT, _VENV_DIR
from anvil.repo.profile import RepoProfile
from anvil.sandbox.base import ExecResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exec_result(exit_code=0, stdout="", stderr="", timed_out=False):
    return ExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr,
                      timed_out=timed_out, duration=0.1)


def _python_profile(install_cmd="pip install -e '.[dev]'") -> RepoProfile:
    return RepoProfile(
        languages=["python"], primary_language="python",
        install_cmd=install_cmd, test_cmd="pytest", test_framework="pytest",
    )


def _go_profile() -> RepoProfile:
    return RepoProfile(
        languages=["go"], primary_language="go",
        install_cmd="go mod download", test_cmd="go test ./...", test_framework="go test",
    )


def _make_sandbox(exec_side_effects: list) -> MagicMock:
    """Return a mock Sandbox whose exec() returns the given results in sequence."""
    sb = MagicMock()
    sb.root = Path("/fake/root")
    sb.exec.side_effect = exec_side_effects
    return sb


# ---------------------------------------------------------------------------
# DepsResult dataclass
# ---------------------------------------------------------------------------

class TestDepsResult:
    def test_ok_true(self):
        r = DepsResult(ok=True, report="done")
        assert r.ok and r.venv_python is None

    def test_ok_false(self):
        r = DepsResult(ok=False, report="fail")
        assert not r.ok

    def test_venv_python_field(self):
        r = DepsResult(ok=True, report="ok", venv_python="/path/to/python")
        assert r.venv_python == "/path/to/python"


# ---------------------------------------------------------------------------
# ensure_deps — no install command
# ---------------------------------------------------------------------------

class TestEnsureDepsNoCmd:
    def test_no_install_cmd_returns_ok(self):
        profile = RepoProfile(
            languages=["python"], primary_language="python",
            install_cmd=None, test_cmd="pytest", test_framework="pytest",
        )
        sb = MagicMock()
        result = ensure_deps(sb, profile)
        assert result.ok
        assert "no install" in result.report.lower() or "skip" in result.report.lower()
        sb.exec.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_deps — Python repos
# ---------------------------------------------------------------------------

class TestEnsureDepsPython:
    def test_successful_python_install(self):
        """Happy path: venv creation + pip upgrade + pip install + framework install all succeed."""
        sb = _make_sandbox([
            _exec_result(0),   # python3 -m venv
            _exec_result(0),   # pip upgrade
            _exec_result(0),   # pip install -e '.[dev]'
            _exec_result(0),   # pytest install (new step 4)
        ])
        result = ensure_deps(sb, _python_profile())
        assert result.ok
        assert result.venv_python is not None
        assert _VENV_DIR in result.venv_python

    def test_venv_creation_failure(self):
        sb = _make_sandbox([
            _exec_result(1, stderr="python3 not found"),  # venv creation fails
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "venv" in result.report.lower()

    def test_pip_install_failure(self):
        sb = _make_sandbox([
            _exec_result(0),        # venv OK
            _exec_result(0),        # pip upgrade OK
            _exec_result(1, stderr="No module named 'setuptools'"),  # install fails
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "failed" in result.report.lower()
        # Still returns venv_python so caller can try anyway
        assert result.venv_python is not None

    def test_install_timeout(self):
        sb = _make_sandbox([
            _exec_result(0),        # venv OK
            _exec_result(0),        # pip upgrade OK
            _exec_result(-1, timed_out=True),  # install times out
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "timed out" in result.report.lower()

    def test_uses_venv_pip_not_bare_pip(self):
        """The install command must use the venv-scoped pip, not the system one."""
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
            _exec_result(0),
            _exec_result(0),   # framework install
        ])
        ensure_deps(sb, _python_profile("pip install -r requirements.txt"))
        # Third exec call should have the venv pip, not bare 'pip'
        install_call_cmd = sb.exec.call_args_list[2][0][0]
        assert _VENV_DIR in install_call_cmd
        assert "pip install" in install_call_cmd

    def test_venv_timeout_handled(self):
        sb = _make_sandbox([
            _exec_result(-1, timed_out=True),  # venv creation times out
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok


# ---------------------------------------------------------------------------
# ensure_deps — non-Python repos
# ---------------------------------------------------------------------------

class TestEnsureDepsGeneric:
    def test_go_success(self):
        sb = _make_sandbox([_exec_result(0, stdout="downloading modules")])
        result = ensure_deps(sb, _go_profile())
        assert result.ok
        assert result.venv_python is None

    def test_go_failure(self):
        sb = _make_sandbox([_exec_result(1, stderr="network error")])
        result = ensure_deps(sb, _go_profile())
        assert not result.ok
        assert "failed" in result.report.lower()

    def test_generic_timeout(self):
        sb = _make_sandbox([_exec_result(-1, timed_out=True)])
        result = ensure_deps(sb, _go_profile())
        assert not result.ok
        assert "timed out" in result.report.lower()

    def test_timeout_value_passed(self):
        """Ensure exec is called with the 5-minute cap."""
        sb = MagicMock()
        sb.root = Path("/fake")
        sb.exec.return_value = _exec_result(0)
        ensure_deps(sb, _go_profile())
        # The timeout kwarg to exec must be <= _INSTALL_TIMEOUT
        _, kwargs = sb.exec.call_args
        timeout_used = kwargs.get("timeout", sb.exec.call_args[0][1] if len(sb.exec.call_args[0]) > 1 else _INSTALL_TIMEOUT)
        assert timeout_used <= _INSTALL_TIMEOUT
