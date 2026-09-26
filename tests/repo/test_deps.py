"""Offline tests for repo/deps.py — all sandbox calls are faked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


def _js_profile() -> RepoProfile:
    return RepoProfile(
        languages=["javascript"], primary_language="javascript",
        install_cmd="npm install", test_cmd="npm test", test_framework="mocha",
    )


def _rust_profile() -> RepoProfile:
    return RepoProfile(
        languages=["rust"], primary_language="rust",
        install_cmd="cargo build", test_cmd="cargo test", test_framework="cargo test",
    )


def _java_profile() -> RepoProfile:
    return RepoProfile(
        languages=["java"], primary_language="java",
        install_cmd="mvn -q package -DskipTests",
        test_cmd="mvn -q test", test_framework="junit",
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

    def test_skipped_flag(self):
        r = DepsResult(ok=True, report="skip", skipped=True)
        assert r.skipped


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
        assert result.skipped
        sb.exec.assert_not_called()

    def test_ensure_deps_never_raises(self):
        """Bug 4: ensure_deps must never raise, even on internal errors."""
        sb = MagicMock()
        sb.exec.side_effect = RuntimeError("boom")
        sb.root = Path("/fake")
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "internal error" in result.report.lower() or "boom" in result.report.lower()


# ---------------------------------------------------------------------------
# ensure_deps — Python repos
# ---------------------------------------------------------------------------

class TestEnsureDepsPython:
    def test_successful_python_install(self):
        """Happy path: venv creation + pip upgrade + pip install + framework install."""
        sb = _make_sandbox([
            _exec_result(0),   # python3 -m venv
            _exec_result(0),   # pip upgrade
            _exec_result(0),   # pip install -e '.[dev]'
            _exec_result(0),   # pytest install (step 4)
        ])
        result = ensure_deps(sb, _python_profile())
        assert result.ok
        assert result.venv_python is not None
        assert _VENV_DIR in result.venv_python

    def test_venv_creation_failure(self):
        sb = _make_sandbox([
            _exec_result(1, stderr="python3 not found"),
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "venv" in result.report.lower()

    def test_pip_install_failure(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
            _exec_result(1, stderr="No module named 'setuptools'"),
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "failed" in result.report.lower()
        assert result.venv_python is not None

    def test_install_timeout(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
            _exec_result(-1, timed_out=True),
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok
        assert "timed out" in result.report.lower()

    def test_uses_venv_pip_not_bare_pip(self):
        """The install command must use the venv-scoped pip."""
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
            _exec_result(0),
            _exec_result(0),
        ])
        ensure_deps(sb, _python_profile("pip install -r requirements.txt"))
        install_call_cmd = sb.exec.call_args_list[2][0][0]
        assert _VENV_DIR in install_call_cmd
        assert "pip install" in install_call_cmd

    def test_venv_timeout_handled(self):
        sb = _make_sandbox([
            _exec_result(-1, timed_out=True),
        ])
        result = ensure_deps(sb, _python_profile())
        assert not result.ok


# ---------------------------------------------------------------------------
# ensure_deps — JS repos (new in this PR)
# ---------------------------------------------------------------------------

class TestEnsureDepsJS:
    def test_js_success_with_npm(self):
        """Bug 4: JS path probes npm then runs install."""
        sb = _make_sandbox([
            _exec_result(0),   # command -v npm
            _exec_result(0, stdout="added 100 packages"),   # npm install
        ])
        # No lock file on fake root → npm install directly
        result = ensure_deps(sb, _js_profile())
        assert result.ok
        assert result.venv_python is None

    def test_js_npm_not_found_skips_cleanly(self):
        """Bug 4: missing npm → skipped=True, not an error crash."""
        sb = _make_sandbox([
            _exec_result(1),  # command -v npm fails
        ])
        result = ensure_deps(sb, _js_profile())
        assert not result.ok
        assert result.skipped
        assert "npm" in result.report.lower()

    def test_js_install_failure(self):
        sb = _make_sandbox([
            _exec_result(0),   # npm found
            _exec_result(1, stderr="npm ERR!"),
        ])
        result = ensure_deps(sb, _js_profile())
        assert not result.ok
        assert "failed" in result.report.lower()


# ---------------------------------------------------------------------------
# ensure_deps — Go repos
# ---------------------------------------------------------------------------

class TestEnsureDepsGo:
    def test_go_success(self):
        """Bug 4: Go path probes 'go', then runs go mod download."""
        sb = _make_sandbox([
            _exec_result(0),   # command -v go
            _exec_result(0, stdout="downloading modules"),
        ])
        result = ensure_deps(sb, _go_profile())
        assert result.ok
        assert result.venv_python is None

    def test_go_toolchain_missing_skips(self):
        """Bug 4: missing go binary → skip cleanly."""
        sb = _make_sandbox([
            _exec_result(1),   # command -v go fails
        ])
        result = ensure_deps(sb, _go_profile())
        assert not result.ok
        assert result.skipped
        assert "go" in result.report.lower()

    def test_go_download_failure(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(1, stderr="network error"),
        ])
        result = ensure_deps(sb, _go_profile())
        assert not result.ok
        assert "failed" in result.report.lower()

    def test_go_timeout(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(-1, timed_out=True),
        ])
        result = ensure_deps(sb, _go_profile())
        assert not result.ok
        assert "timed out" in result.report.lower()

    def test_timeout_value_not_exceeded(self):
        """Install exec must use the 5-minute cap."""
        sb = MagicMock()
        sb.root = Path("/fake")
        sb.exec.return_value = _exec_result(0)
        ensure_deps(sb, _go_profile())
        for c in sb.exec.call_args_list:
            args = c.args
            kwargs = c.kwargs
            t = kwargs.get("timeout", args[1] if len(args) > 1 else _INSTALL_TIMEOUT)
            assert t <= _INSTALL_TIMEOUT



# ---------------------------------------------------------------------------
# ensure_deps — Rust repos
# ---------------------------------------------------------------------------

class TestEnsureDepsRust:
    def test_rust_success(self):
        sb = _make_sandbox([
            _exec_result(0),   # command -v cargo
            _exec_result(0),   # cargo fetch
        ])
        result = ensure_deps(sb, _rust_profile())
        assert result.ok

    def test_rust_cargo_missing_skips(self):
        sb = _make_sandbox([_exec_result(1)])
        result = ensure_deps(sb, _rust_profile())
        assert not result.ok
        assert result.skipped
        assert "cargo" in result.report.lower()


# ---------------------------------------------------------------------------
# ensure_deps — Java repos
# ---------------------------------------------------------------------------

class TestEnsureDepsJava:
    def test_java_success(self):
        sb = _make_sandbox([
            _exec_result(0),   # command -v mvn
            _exec_result(0),   # mvn dependency:resolve
        ])
        result = ensure_deps(sb, _java_profile())
        assert result.ok

    def test_java_mvn_missing_skips(self):
        sb = _make_sandbox([_exec_result(1)])
        result = ensure_deps(sb, _java_profile())
        assert not result.ok
        assert result.skipped
        assert "mvn" in result.report.lower()

    def test_java_uses_dependency_resolve(self):
        """Bug 4: Java should prefer dependency:resolve over full package build."""
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
        ])
        ensure_deps(sb, _java_profile())
        install_cmd = sb.exec.call_args_list[1][0][0]
        assert "dependency:resolve" in install_cmd
