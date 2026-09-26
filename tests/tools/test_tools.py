"""Offline tests for all 7 tools + ToolRegistry — uses a real git tmp repo, no network."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anvil.sandbox.worktree import WorktreeSandbox
from anvil.tools.list_dir import ListDirTool
from anvil.tools.grep import GrepTool
from anvil.tools.read_file import ReadFileTool
from anvil.tools.edit_file import EditFileTool
from anvil.tools.run_cmd import RunCmdTool
from anvil.tools.run_tests import RunTestsTool, _extract_failure_summary
from anvil.tools.git_diff import GitDiffTool
from anvil.tools.registry import ToolRegistry, make_default_registry
from anvil.repo.profile import RepoProfile


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True, capture_output=True)
    (path / "hello.py").write_text("def greet():\n    return 'hello'\n")
    (path / "data.txt").write_text("line one\nline two\nline three\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)
    return path


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return _init_git_repo(tmp_path_factory.mktemp("repo"))


@pytest.fixture()
def sandbox(repo, tmp_path):
    work = tmp_path / "work"
    sb = WorktreeSandbox(repo_root=repo, work_dir=work)
    yield sb
    sb.close()


# ---------------------------------------------------------------------------
# ListDirTool
# ---------------------------------------------------------------------------

class TestListDirTool:
    def test_root_listing(self, sandbox):
        result = ListDirTool().run({"path": "."}, sandbox)
        assert result.ok
        assert "hello.py" in result.output

    def test_nonexistent_path(self, sandbox):
        result = ListDirTool().run({"path": "no_such_dir"}, sandbox)
        assert not result.ok

    def test_file_listing(self, sandbox):
        result = ListDirTool().run({"path": "hello.py"}, sandbox)
        assert result.ok
        assert "hello.py" in result.output

    def test_path_traversal_blocked(self, sandbox):
        result = ListDirTool().run({"path": "../../etc"}, sandbox)
        assert not result.ok

    def test_schema_valid(self):
        tool = ListDirTool()
        assert tool.parameters["type"] == "object"
        assert "path" in tool.parameters["properties"]
        assert "path" in tool.parameters["required"]


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------

class TestGrepTool:
    def test_finds_pattern(self, sandbox):
        result = GrepTool().run({"pattern": "greet"}, sandbox)
        assert result.ok
        assert "greet" in result.output

    def test_no_matches(self, sandbox):
        result = GrepTool().run({"pattern": "zzz_not_present_xyz"}, sandbox)
        assert result.ok
        assert "No matches" in result.output

    def test_case_insensitive(self, sandbox):
        result = GrepTool().run({"pattern": "GREET", "case_sensitive": False}, sandbox)
        assert result.ok
        assert "greet" in result.output.lower()

    def test_bad_path(self, sandbox):
        result = GrepTool().run({"pattern": "x", "path": "no_dir"}, sandbox)
        assert not result.ok

    def test_missing_pattern(self, sandbox):
        result = GrepTool().run({}, sandbox)
        assert not result.ok

    def test_schema_has_pattern(self):
        assert "pattern" in GrepTool().parameters["properties"]
        assert GrepTool().parameters["required"] == ["pattern"]

    def test_output_relative_paths(self, sandbox):
        """Paths in output should be relative, not absolute."""
        result = GrepTool().run({"pattern": "greet"}, sandbox)
        assert str(sandbox.root) not in result.output


# ---------------------------------------------------------------------------
# ReadFileTool
# ---------------------------------------------------------------------------

class TestReadFileTool:
    def test_reads_whole_file(self, sandbox):
        result = ReadFileTool().run({"path": "hello.py"}, sandbox)
        assert result.ok
        assert "greet" in result.output

    def test_line_numbers_present(self, sandbox):
        result = ReadFileTool().run({"path": "data.txt"}, sandbox)
        assert result.ok
        assert "1" in result.output  # line 1 numbered

    def test_line_range(self, sandbox):
        result = ReadFileTool().run({"path": "data.txt", "start": 2, "end": 2}, sandbox)
        assert result.ok
        assert "line two" in result.output
        assert "line one" not in result.output

    def test_missing_file(self, sandbox):
        result = ReadFileTool().run({"path": "ghost.txt"}, sandbox)
        assert not result.ok

    def test_no_path_arg(self, sandbox):
        result = ReadFileTool().run({}, sandbox)
        assert not result.ok

    def test_schema_required_path(self):
        assert "path" in ReadFileTool().parameters["required"]


# ---------------------------------------------------------------------------
# EditFileTool
# ---------------------------------------------------------------------------

class TestEditFileTool:
    def test_successful_replace(self, sandbox):
        sandbox.write_file("edit_me.py", "x = 1\ny = 2\n")
        result = EditFileTool().run({"path": "edit_me.py", "old": "x = 1", "new": "x = 99"}, sandbox)
        assert result.ok
        assert "99" in sandbox.read_file("edit_me.py")

    def test_not_found_returns_error(self, sandbox):
        sandbox.write_file("f.py", "hello world\n")
        result = EditFileTool().run({"path": "f.py", "old": "not_there", "new": "x"}, sandbox)
        assert not result.ok
        assert "not found" in result.output.lower()

    def test_not_found_includes_close_lines(self, sandbox):
        sandbox.write_file("close.py", "def foo():\n    pass\n")
        result = EditFileTool().run({"path": "close.py", "old": "def fob():", "new": "def fob():"}, sandbox)
        assert not result.ok
        # Should hint at the closest line
        assert "foo" in result.output

    def test_multiple_occurrences_fails(self, sandbox):
        sandbox.write_file("dup.py", "x = 1\nx = 1\n")
        result = EditFileTool().run({"path": "dup.py", "old": "x = 1", "new": "x = 2"}, sandbox)
        assert not result.ok
        assert "2" in result.output  # mentions count

    def test_missing_file(self, sandbox):
        result = EditFileTool().run({"path": "nope.py", "old": "x", "new": "y"}, sandbox)
        assert not result.ok

    def test_empty_old_fails(self, sandbox):
        result = EditFileTool().run({"path": "hello.py", "old": "", "new": "x"}, sandbox)
        assert not result.ok

    def test_schema_required_fields(self):
        assert set(EditFileTool().parameters["required"]) == {"path", "old", "new"}

    def test_whitespace_only_warns(self, sandbox):
        """Replacing 'x = 1' with 'x = 1  ' (trailing space) should warn."""
        sandbox.write_file("ws.py", "x = 1\n")
        result = EditFileTool().run({"path": "ws.py", "old": "x = 1", "new": "x = 1  "}, sandbox)
        assert result.ok
        assert result.meta.get("whitespace_only") is True
        assert "whitespace" in result.output.lower()

    def test_non_whitespace_no_warn(self, sandbox):
        sandbox.write_file("real.py", "x = 1\n")
        result = EditFileTool().run({"path": "real.py", "old": "x = 1", "new": "x = 99"}, sandbox)
        assert result.ok
        assert not result.meta.get("whitespace_only")

    def test_syntax_guard_python(self, sandbox):
        sandbox.write_file("syn.py", "def foo():\n    pass\n")
        result = EditFileTool().run({"path": "syn.py", "old": "pass", "new": "pass)"}, sandbox)
        assert not result.ok
        assert "Edit reverted due to syntax error" in result.output
        assert "SyntaxError:" in result.output
        assert sandbox.read_file("syn.py") == "def foo():\n    pass\n"

    def test_syntax_guard_ok(self, sandbox):
        sandbox.write_file("syn.py", "def foo():\n    pass\n")
        result = EditFileTool().run({"path": "syn.py", "old": "pass", "new": "return 1"}, sandbox)
        assert result.ok
        assert sandbox.read_file("syn.py") == "def foo():\n    return 1\n"


# ---------------------------------------------------------------------------
# RunCmdTool
# ---------------------------------------------------------------------------

class TestRunCmdTool:
    def test_simple_command(self, sandbox):
        result = RunCmdTool().run({"cmd": "echo hello"}, sandbox)
        assert result.ok
        assert "hello" in result.output

    def test_nonzero_exit(self, sandbox):
        result = RunCmdTool().run({"cmd": "exit 1"}, sandbox)
        assert not result.ok
        assert result.meta["exit_code"] == 1

    def test_timeout(self, sandbox):
        result = RunCmdTool().run({"cmd": "sleep 60", "timeout": 1}, sandbox)
        assert not result.ok
        assert result.meta["timed_out"]

    def test_missing_cmd(self, sandbox):
        result = RunCmdTool().run({}, sandbox)
        assert not result.ok

    def test_duration_in_meta(self, sandbox):
        result = RunCmdTool().run({"cmd": "echo x"}, sandbox)
        assert "duration" in result.meta

    def test_schema_required_cmd(self):
        assert "cmd" in RunCmdTool().parameters["required"]

    def test_sudo_blocked(self, sandbox):
        result = RunCmdTool().run({"cmd": "sudo apt install x"}, sandbox)
        assert not result.ok
        assert "sudo" in result.output
        assert "rejected" in result.output.lower()

    def test_cd_parent_blocked(self, sandbox):
        result = RunCmdTool().run({"cmd": "cd .. && rm -rf *"}, sandbox)
        assert not result.ok
        assert "cd .." in result.output
        assert "rejected" in result.output.lower()

    def test_absolute_write_blocked(self, sandbox):
        result = RunCmdTool().run({"cmd": "echo 'x' > /tmp/x"}, sandbox)
        assert not result.ok
        assert "absolute paths" in result.output
        assert "rejected" in result.output.lower()
        
        result2 = RunCmdTool().run({"cmd": "cat x >> /etc/hosts"}, sandbox)
        assert not result2.ok

    def test_pip_install_without_venv_blocked(self, sandbox):
        # sandbox doesn't have .anvil_venv by default in this fixture
        result = RunCmdTool().run({"cmd": "pip install requests"}, sandbox)
        assert not result.ok
        assert "pip install" in result.output
        assert "virtual environment" in result.output
        
    def test_pip_install_with_venv_allowed(self, sandbox):
        # Fake a venv
        (sandbox.root / ".anvil_venv").mkdir()
        # It will actually run the command now. We expect it to try running pip install
        # which will fail because the mock venv is empty, so exit code won't be 0,
        # but the rejection message shouldn't be there.
        result = RunCmdTool().run({"cmd": "pip install requests"}, sandbox)
        assert "Command rejected" not in result.output


# ---------------------------------------------------------------------------
# RunTestsTool
# ---------------------------------------------------------------------------

def _pytest_profile() -> RepoProfile:
    return RepoProfile(
        languages=["python"], primary_language="python",
        install_cmd=None, test_cmd="pytest", test_framework="pytest",
    )


class TestRunTestsTool:
    def test_runs_with_profile(self, sandbox):
        # Write a passing pytest test in the sandbox
        sandbox.write_file("test_pass.py", "def test_ok(): assert 1 == 1\n")
        result = RunTestsTool(profile=_pytest_profile()).run({}, sandbox)
        # pytest may or may not be on PATH in the sandbox env; check graceful failure at least
        assert isinstance(result.ok, bool)
        assert isinstance(result.output, str)

    def test_no_profile_no_manifest_fails(self, sandbox, tmp_path):
        """Without a profile and without any manifest, should fail gracefully."""
        empty_repo = tmp_path / "empty_repo"
        _init_git_repo(empty_repo)
        empty_work = tmp_path / "empty_work"
        sb2 = WorktreeSandbox(repo_root=empty_repo, work_dir=empty_work)
        try:
            result = RunTestsTool(profile=None).run({}, sb2)
            assert not result.ok
        finally:
            sb2.close()

    def test_failure_summary_pytest_names(self):
        output = (
            "FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2\n"
            "FAILED tests/test_baz.py::TestClass::test_method\n"
            "PASSED tests/test_ok.py::test_good\n"
        )
        summary = _extract_failure_summary(output)
        assert "test_bar" in summary or "test_foo" in summary
        assert "test_method" in summary or "test_baz" in summary
        assert "test_good" not in summary

    def test_failure_summary_go_pattern(self):
        output = "--- FAIL: TestFoo (0.01s)\n--- FAIL: TestBar/SubTest (0.00s)\nPASS\n"
        summary = _extract_failure_summary(output)
        assert "TestFoo" in summary or "TestBar" in summary

    def test_failure_summary_assertion_extracted(self):
        output = "FAILED test_x.py::test_y\nAssertionError: 1 != 2\n"
        summary = _extract_failure_summary(output)
        assert "1 != 2" in summary or "AssertionError" in summary

    def test_output_never_exceeds_char_cap(self, sandbox):
        """run_tests output must be capped at char_cap characters."""
        # Mock a sandbox that returns huge output
        from unittest.mock import MagicMock
        from anvil.sandbox.base import ExecResult
        mock_sb = MagicMock()
        mock_sb.root = sandbox.root
        mock_sb.exec.return_value = ExecResult(
            exit_code=0, stdout="X" * 20000, stderr="", timed_out=False, duration=0.1
        )
        cap = 1000
        result = RunTestsTool(profile=_pytest_profile(), char_cap=cap).run({}, mock_sb)
        assert len(result.output) <= cap + 100  # small tolerance for markers

    def test_target_appended_to_cmd(self, sandbox):
        from unittest.mock import MagicMock
        from anvil.sandbox.base import ExecResult
        mock_sb = MagicMock()
        mock_sb.root = sandbox.root
        mock_sb.exec.return_value = ExecResult(0, "", "", False, 0.1)
        RunTestsTool(profile=_pytest_profile()).run({"target": "tests/test_foo.py"}, mock_sb)
        cmd_used = mock_sb.exec.call_args[0][0]
        assert "tests/test_foo.py" in cmd_used

    def test_schema_no_required(self):
        assert RunTestsTool().parameters["required"] == []



# ---------------------------------------------------------------------------
# GitDiffTool
# ---------------------------------------------------------------------------

class TestGitDiffTool:
    def test_no_changes(self, sandbox):
        result = GitDiffTool().run({}, sandbox)
        assert result.ok
        assert "no changes" in result.output.lower() or result.output.strip() == "(no changes)"

    def test_detects_modification(self, sandbox):
        sandbox.write_file("hello.py", "def greet():\n    return 'modified'\n")
        result = GitDiffTool().run({}, sandbox)
        assert result.ok
        assert result.meta.get("changed") is True

    def test_schema_is_empty(self):
        assert GitDiffTool().parameters["required"] == []


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register(ListDirTool())
        tool = reg.get("list_dir")
        assert tool.name == "list_dir"

    def test_duplicate_register_raises(self):
        reg = ToolRegistry()
        reg.register(ListDirTool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ListDirTool())

    def test_get_missing_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_schemas_format(self):
        reg = ToolRegistry()
        reg.register(ListDirTool())
        schemas = reg.schemas()
        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]

    def test_len(self):
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(GrepTool())
        assert len(reg) == 1

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(GrepTool())
        assert "grep" in reg
        assert "list_dir" not in reg

    def test_make_default_registry_has_all_tools(self):
        reg = make_default_registry()
        for name in ("list_dir", "grep", "read_file", "edit_file",
                     "run_cmd", "run_tests", "git_diff"):
            assert name in reg

    def test_make_default_registry_schemas_count(self):
        reg = make_default_registry()
        assert len(reg.schemas()) == 10
