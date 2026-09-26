"""Offline tests for repo/deps.py — all sandbox calls are faked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anvil.repo.deps import (
    DepsResult,
    ensure_deps,
    _INSTALL_TIMEOUT,
    _VENV_DIR,
    _PYTHON_CANDIDATES,
    _parse_requires_python,
    _pick_python_interpreter,
)
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
    """Return a mock Sandbox whose exec() returns the given results in sequence.

    sandbox.root is set to a tmp path that has NO pyproject.toml or setup.cfg
    so _parse_requires_python returns None (no version check needed in probes).
    """
    sb = MagicMock()
    sb.root = Path("/fake/root")  # no pyproject.toml → no version check
    sb.exec.side_effect = exec_side_effects
    return sb


def _py_probe_ok() -> list:
    """Mock 6 exec calls: command -v, version, rm, venv create, import check, rm."""
    return [
        _exec_result(0),                  # command -v
        _exec_result(0, stdout="3.13\n"), # version print
        _exec_result(0),                  # rm -rf .probe_venv
        _exec_result(0),                  # venv create
        _exec_result(0),                  # import check
        _exec_result(0),                  # rm -rf .probe_venv (cleanup)
    ]


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
        result = ensure_deps(sb, profile, as_of="2021")
        assert result.ok
        assert result.skipped
        sb.exec.assert_not_called()

    def test_ensure_deps_never_raises(self):
        """Bug 4: ensure_deps must never raise, even on internal errors."""
        sb = MagicMock()
        sb.exec.side_effect = RuntimeError("boom")
        sb.root = Path("/fake")
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert not result.ok
        assert "internal error" in result.report.lower() or "boom" in result.report.lower()


# ---------------------------------------------------------------------------
# _parse_requires_python (Task 3 unit tests)
# ---------------------------------------------------------------------------

class TestParseRequiresPython:
    def test_no_files_returns_none(self, tmp_path):
        sb = MagicMock()
        sb.root = tmp_path
        assert _parse_requires_python(sb) is None

    def test_pyproject_with_constraint(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nrequires-python = ">=3.11"\n'
        )
        sb = MagicMock()
        sb.root = tmp_path
        assert _parse_requires_python(sb) == ">=3.11"

    def test_pyproject_without_constraint_defaults_39(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
        sb = MagicMock()
        sb.root = tmp_path
        assert _parse_requires_python(sb) == ">=3.9"

    def test_setup_cfg_with_constraint(self, tmp_path):
        (tmp_path / "setup.cfg").write_text(
            "[options]\npython_requires = >=3.10\n"
        )
        sb = MagicMock()
        sb.root = tmp_path
        assert _parse_requires_python(sb) == ">=3.10"


# ---------------------------------------------------------------------------
# _pick_python_interpreter (Task 3 unit tests)
# ---------------------------------------------------------------------------

class TestPickPythonInterpreter:
    def test_first_candidate_found(self):
        """When python3.13 is available and can create a venv, it is chosen."""
        sb = _make_sandbox(_py_probe_ok())
        result = _pick_python_interpreter(sb)
        assert result == "python3.13"

    def test_fallback_to_python3(self):
        """Falls back through candidates until python3 works."""
        # candidates: 3.13, 3.12, 3.11, 3.10, 3.9, 3 all fail command -v except python3
        sb = _make_sandbox([
            _exec_result(1),  # python3.13 not found
            _exec_result(1),  # python3.12 not found
            _exec_result(1),  # python3.11 not found
            _exec_result(1),  # python3.10 not found
            _exec_result(1),  # python3.9 not found
            _exec_result(0),                  # python3 found
            _exec_result(0, stdout="3.9\n"),  # python3 version
            _exec_result(0),                  # rm -rf .probe_venv
            _exec_result(0),                  # python3 venv create ok
            _exec_result(0),                  # import pyexpat ok
            _exec_result(0),                  # rm -rf .probe_venv

        ])
        result = _pick_python_interpreter(sb)
        assert result == "python3"

    def test_none_if_all_fail(self):
        """Returns None if no candidate is usable."""
        sb = _make_sandbox([_exec_result(1)] * len(_PYTHON_CANDIDATES))  # all command -v fail
        result = _pick_python_interpreter(sb)
        assert result is None

    def test_skips_when_venv_not_supported(self):
        """Skips interpreter that can't create a venv."""
        sb = _make_sandbox([
            _exec_result(0),                  # python3.13 found
            _exec_result(0, stdout="3.13\n"), # python3.13 version
            _exec_result(0),                  # rm -rf .probe_venv
            _exec_result(1),                  # python3.13 venv create fails
            _exec_result(0),                  # python3.12 found
            _exec_result(0, stdout="3.12\n"), # python3.12 version
            _exec_result(0),                  # rm -rf .probe_venv
            _exec_result(0),                  # python3.12 venv create ok
            _exec_result(0),                  # python3.12 import ok
            _exec_result(0),                  # rm -rf .probe_venv

        ])
        result = _pick_python_interpreter(sb)
        assert result == "python3.12"

    def test_specifier_evaluation(self, tmp_path):
        """Tests for '>=3.8,<3.12', '!=3.12.*', '==3.10.*', '>=3.13', none satisfiable."""
        def run_with_spec(spec_str):
            (tmp_path / "pyproject.toml").write_text(f'[project]\nrequires-python = "{spec_str}"\n')
            sb = MagicMock()
            sb.root = tmp_path
            
            def mock_exec(cmd, **kwargs):
                if "command -v" in cmd:
                    return _exec_result(0)
                if "import sys" in cmd:
                    # extract 'python3.x' from cmd
                    for c in _PYTHON_CANDIDATES:
                        if cmd.startswith(c):
                            ver = c.replace("python", "")
                            if not ver or ver == "3":
                                ver = "3.8"  # fallback
                            return _exec_result(0, stdout=f"{ver}\n")
                    return _exec_result(0, stdout="3.9\n")
                if "venv" in cmd and "probe" in cmd:
                    return _exec_result(0)
                if "import pyexpat" in cmd:
                    return _exec_result(0)
                if "rm -rf" in cmd:
                    return _exec_result(0)
                return _exec_result(1)
            
            sb.exec.side_effect = mock_exec
            return _pick_python_interpreter(sb)

        assert run_with_spec(">=3.8,<3.12") == "python3.11"
        assert run_with_spec("!=3.12.*") == "python3.13"
        assert run_with_spec("==3.10.*") == "python3.10"
        assert run_with_spec(">=3.13") == "python3.13"
        assert run_with_spec(">=4.0") is None


# ---------------------------------------------------------------------------
# ensure_deps — Python repos
# ---------------------------------------------------------------------------

class TestEnsureDepsPython:
    def test_successful_python_install(self):
        """Happy path: probe ok + venv + pip upgrade + pip install + framework."""
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(0),   # interp -m venv .anvil_venv
                _exec_result(0),   # pip upgrade
                _exec_result(1),   # command -v uv (fail -> pip)
                _exec_result(0),   # pip install -e '.[dev]'
                _exec_result(0),   # pytest install
            ]
        )
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert result.ok
        assert result.venv_python is not None
        assert _VENV_DIR in result.venv_python

    def test_no_interpreter_found_skips(self):
        """Task 3: all candidates fail → skipped with clear reason."""
        sb = _make_sandbox([_exec_result(1)] * (len(_PYTHON_CANDIDATES)*6 + 1))
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert not result.ok
        assert result.skipped
        assert "python" in result.report.lower()

    def test_venv_creation_failure(self):
        """Venv creation failing returns ok=False."""
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(1, stderr="No space left"),
            ]
        )
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert not result.ok
        assert "venv" in result.report.lower()

    def test_pip_install_failure(self):
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(0),
                _exec_result(0),
                _exec_result(1), # test -x uv (fail -> pip)
                _exec_result(1, stderr="No module named 'setuptools'"),
            ]
        )
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert not result.ok
        assert "failed" in result.report.lower()
        assert result.venv_python is not None

    def test_install_timeout(self):
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(0),
                _exec_result(0),
                _exec_result(1), # test -x uv (fail -> pip)
                _exec_result(-1, timed_out=True),
            ]
        )
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert not result.ok
        assert "timed out" in result.report.lower()

    def test_uses_venv_pip_not_bare_pip(self):
        """The install command must use the venv-scoped pip."""
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(0),
                _exec_result(0),
                _exec_result(1), # test -x uv
                _exec_result(0),
                _exec_result(0),
            ]
        )
        ensure_deps(sb, _python_profile("pip install -r requirements.txt"), as_of="2021")
        install_call_cmd = sb.exec.call_args_list[-2][0][0] # it was 4, now 5? Let's just check all calls
        assert any(".anvil_venv/bin/pip install" in str(c) for c in sb.exec.call_args_list)

    def test_venv_timeout_is_handled(self):
        """venv creation times out → ok=False."""
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(-1, timed_out=True),
            ]
        )
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert not result.ok

    def test_report_mentions_interpreter(self):
        """Task 3: success report names the interpreter used."""
        sb = _make_sandbox(
            _py_probe_ok() + [
                _exec_result(0),
                _exec_result(0),
                _exec_result(1), # uv probe fail
                _exec_result(0),
                _exec_result(0),
            ]
        )
        result = ensure_deps(sb, _python_profile(), as_of="2021")
        assert result.ok
        assert "python" in result.report.lower()


# ---------------------------------------------------------------------------
# ensure_deps — JS repos
# ---------------------------------------------------------------------------

class TestEnsureDepsJS:
    def test_js_success_with_npm(self):
        sb = _make_sandbox([
            _exec_result(0),   # command -v npm
            _exec_result(0, stdout="added 100 packages"),
        ])
        result = ensure_deps(sb, _js_profile(), as_of="2021")
        assert result.ok
        assert result.venv_python is None

    def test_js_npm_not_found_skips_cleanly(self):
        sb = _make_sandbox([_exec_result(1)])
        result = ensure_deps(sb, _js_profile(), as_of="2021")
        assert not result.ok
        assert result.skipped
        assert "npm" in result.report.lower()

    def test_js_install_failure(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(1, stderr="npm ERR!"),
        ])
        result = ensure_deps(sb, _js_profile(), as_of="2021")
        assert not result.ok
        assert "failed" in result.report.lower()


# ---------------------------------------------------------------------------
# ensure_deps — Go repos
# ---------------------------------------------------------------------------

class TestEnsureDepsGo:
    def test_go_success(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0, stdout="downloading modules"),
        ])
        result = ensure_deps(sb, _go_profile(), as_of="2021")
        assert result.ok
        assert result.venv_python is None

    def test_go_toolchain_missing_skips(self):
        sb = _make_sandbox([_exec_result(1)])
        result = ensure_deps(sb, _go_profile(), as_of="2021")
        assert not result.ok
        assert result.skipped
        assert "go" in result.report.lower()

    def test_go_download_failure(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(1, stderr="network error"),
        ])
        result = ensure_deps(sb, _go_profile(), as_of="2021")
        assert not result.ok
        assert "failed" in result.report.lower()

    def test_go_timeout(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(-1, timed_out=True),
        ])
        result = ensure_deps(sb, _go_profile(), as_of="2021")
        assert not result.ok
        assert "timed out" in result.report.lower()

    def test_timeout_value_not_exceeded(self):
        """Install exec must use the 5-minute cap."""
        sb = MagicMock()
        sb.root = Path("/fake")
        sb.exec.return_value = _exec_result(0)
        ensure_deps(sb, _go_profile(), as_of="2021")
        for c in sb.exec.call_args_list:
            t = c.kwargs.get("timeout", c.args[1] if len(c.args) > 1 else _INSTALL_TIMEOUT)
            assert t <= _INSTALL_TIMEOUT


# ---------------------------------------------------------------------------
# ensure_deps — Rust repos
# ---------------------------------------------------------------------------

class TestEnsureDepsRust:
    def test_rust_success(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
        ])
        result = ensure_deps(sb, _rust_profile(), as_of="2021")
        assert result.ok

    def test_rust_cargo_missing_skips(self):
        sb = _make_sandbox([_exec_result(1)])
        result = ensure_deps(sb, _rust_profile(), as_of="2021")
        assert not result.ok
        assert result.skipped
        assert "cargo" in result.report.lower()


# ---------------------------------------------------------------------------
# ensure_deps — Java repos
# ---------------------------------------------------------------------------

class TestEnsureDepsJava:
    def test_java_success(self):
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
        ])
        result = ensure_deps(sb, _java_profile(), as_of="2021")
        assert result.ok

    def test_java_mvn_missing_skips(self):
        sb = _make_sandbox([_exec_result(1)])
        result = ensure_deps(sb, _java_profile(), as_of="2021")
        assert not result.ok
        assert result.skipped
        assert "mvn" in result.report.lower()

    def test_java_uses_dependency_resolve(self):
        """Java should prefer dependency:resolve over full package build."""
        sb = _make_sandbox([
            _exec_result(0),
            _exec_result(0),
        ])
        ensure_deps(sb, _java_profile(), as_of="2021")
        install_cmd = sb.exec.call_args_list[1][0][0]
        assert "dependency:resolve" in install_cmd
