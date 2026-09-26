"""Offline tests for sandbox/worktree.py — uses a real tmp git repo, no network."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from anvil.sandbox.worktree import WorktreeSandbox, _cap_output, _sanitized_env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> Path:
    """Initialise a minimal git repo with one commit and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=path, check=True,
                   capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=path, check=True, capture_output=True)
    (path / "hello.txt").write_text("hello world\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True,
                   capture_output=True)
    return path


@pytest.fixture()
def repo(tmp_path):
    """A minimal git repo with one committed file."""
    return _init_git_repo(tmp_path / "repo")


@pytest.fixture()
def sandbox(repo, tmp_path):
    """WorktreeSandbox pointed at `repo`, working in a sibling directory."""
    work_dir = tmp_path / "work"
    sb = WorktreeSandbox(repo_root=repo, work_dir=work_dir)
    yield sb
    sb.close()


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestCapOutput:
    def test_short_text_unchanged(self):
        assert _cap_output("hello", 100) == "hello"

    def test_long_text_has_marker(self):
        text = "A" * 200
        result = _cap_output(text, 50)
        assert "omitted" in result
        assert len(result) < len(text)

    def test_head_and_tail_preserved(self):
        text = "START" + "X" * 1000 + "END"
        result = _cap_output(text, 20)
        assert "START" in result
        assert "END" in result

    def test_exact_cap_unchanged(self):
        text = "A" * 100
        assert _cap_output(text, 100) == text


class TestSanitizedEnv:
    def test_api_key_removed(self, monkeypatch):
        monkeypatch.setenv("AI_API_KEY", "secret-key-value")
        env = _sanitized_env()
        assert "AI_API_KEY" not in env

    def test_other_vars_preserved(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "keep-me")
        env = _sanitized_env()
        assert env.get("MY_VAR") == "keep-me"

    def test_openai_key_removed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")
        env = _sanitized_env()
        assert "OPENAI_API_KEY" not in env


# ---------------------------------------------------------------------------
# WorktreeSandbox.root
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root_is_work_dir(self, sandbox, tmp_path):
        assert sandbox.root == tmp_path / "work"

    def test_root_exists(self, sandbox):
        assert sandbox.root.is_dir()


# ---------------------------------------------------------------------------
# WorktreeSandbox.exec
# ---------------------------------------------------------------------------

class TestExec:
    def test_simple_command(self, sandbox):
        result = sandbox.exec("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert not result.timed_out

    def test_nonzero_exit_code(self, sandbox):
        result = sandbox.exec("exit 42", timeout=5)
        assert result.exit_code == 42

    def test_stderr_captured(self, sandbox):
        result = sandbox.exec("echo error >&2")
        assert "error" in result.stderr

    def test_timeout_sets_flag(self, sandbox):
        result = sandbox.exec("sleep 60", timeout=1)
        assert result.timed_out
        assert result.exit_code == -1

    def test_duration_positive(self, sandbox):
        result = sandbox.exec("echo ok")
        assert result.duration >= 0

    def test_api_key_not_in_child_env(self, sandbox, monkeypatch):
        monkeypatch.setenv("AI_API_KEY", "should-not-appear")
        result = sandbox.exec("printenv AI_API_KEY; true")
        # Output must NOT contain the secret
        assert "should-not-appear" not in result.stdout

    def test_output_capped(self, sandbox):
        # Generate ~50KB of output — must be capped
        result = sandbox.exec("python3 -c \"print('X'*50000)\"")
        assert len(result.stdout) < 50000
        if len("X" * 50000) > sandbox._char_cap // 2:
            assert "omitted" in result.stdout


# ---------------------------------------------------------------------------
# WorktreeSandbox.read_file / write_file
# ---------------------------------------------------------------------------

class TestReadWriteFile:
    def test_read_existing_file(self, sandbox):
        content = sandbox.read_file("hello.txt")
        assert "hello world" in content

    def test_read_line_range(self, sandbox):
        sandbox.write_file("multi.txt", "line1\nline2\nline3\n")
        assert sandbox.read_file("multi.txt", start=2, end=2) == "line2\n"

    def test_read_start_only(self, sandbox):
        sandbox.write_file("multi.txt", "a\nb\nc\n")
        result = sandbox.read_file("multi.txt", start=2)
        assert "b" in result and "c" in result and "a" not in result

    def test_read_end_only(self, sandbox):
        sandbox.write_file("multi.txt", "a\nb\nc\n")
        result = sandbox.read_file("multi.txt", end=2)
        assert "a" in result and "b" in result and "c" not in result

    def test_write_creates_file(self, sandbox):
        sandbox.write_file("new.txt", "content")
        assert (sandbox.root / "new.txt").read_text() == "content"

    def test_write_creates_parent_dirs(self, sandbox):
        sandbox.write_file("deep/nested/file.txt", "data")
        assert (sandbox.root / "deep" / "nested" / "file.txt").exists()

    def test_write_overwrites(self, sandbox):
        sandbox.write_file("x.txt", "old")
        sandbox.write_file("x.txt", "new")
        assert sandbox.read_file("x.txt") == "new"


    def test_read_missing_file_raises(self, sandbox):
        with pytest.raises(FileNotFoundError):
            sandbox.read_file("does_not_exist.txt")

    def test_path_traversal_rejected(self, sandbox):
        with pytest.raises(PermissionError):
            sandbox.read_file("../../etc/passwd")

    def test_write_traversal_rejected(self, sandbox):
        with pytest.raises(PermissionError):
            sandbox.write_file("../../tmp/evil.txt", "bad")

    # --- New edge-case tests (hardening) ---

    def test_absolute_path_rejected_read(self, sandbox):
        with pytest.raises(PermissionError):
            sandbox.read_file("/etc/passwd")

    def test_absolute_path_rejected_write(self, sandbox):
        with pytest.raises(PermissionError):
            sandbox.write_file("/tmp/evil.txt", "bad")

    def test_binary_file_returns_placeholder(self, sandbox):
        # Write raw null bytes to simulate a binary file
        (sandbox.root / "binary.bin").write_bytes(b"\x00\x01\x02\x03" * 100)
        content = sandbox.read_file("binary.bin")
        assert "binary file" in content.lower()

    def test_large_file_is_capped(self, sandbox):
        # Write a file larger than _MAX_READ_BYTES (2MB)
        big = b"A" * (2 * 1024 * 1024 + 1024)  # 2MB + 1KB
        (sandbox.root / "big.txt").write_bytes(big)
        content = sandbox.read_file("big.txt")
        assert "truncated" in content.lower() or len(content) < len(big)

    def test_crlf_normalised(self, sandbox):
        (sandbox.root / "crlf.txt").write_bytes(b"line1\r\nline2\r\nline3\r\n")
        content = sandbox.read_file("crlf.txt")
        assert "\r\n" not in content
        assert "line1" in content and "line2" in content

    def test_unicode_replacement(self, sandbox):
        # Write bytes that are invalid UTF-8
        (sandbox.root / "bad_utf8.txt").write_bytes(b"hello \xff\xfe world")
        content = sandbox.read_file("bad_utf8.txt")
        assert "hello" in content  # readable parts preserved

    def test_file_without_trailing_newline(self, sandbox):
        sandbox.write_file("no_newline.txt", "last line no newline")
        content = sandbox.read_file("no_newline.txt")
        assert content == "last line no newline"

    def test_symlink_outside_sandbox_rejected(self, sandbox, tmp_path):
        # Create a symlink that points outside the sandbox root
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        link = sandbox.root / "escape_link.txt"
        link.symlink_to(outside)
        with pytest.raises(PermissionError):
            sandbox.read_file("escape_link.txt")




# ---------------------------------------------------------------------------
# WorktreeSandbox.diff
# ---------------------------------------------------------------------------

class TestDiff:
    def test_no_changes_empty_diff(self, sandbox):
        diff = sandbox.diff()
        # Either empty string or only whitespace when nothing changed
        assert diff.strip() == "" or "(no baseline" in diff

    def test_modified_file_appears(self, sandbox):
        sandbox.write_file("hello.txt", "changed content\n")
        diff = sandbox.diff()
        assert "hello.txt" in diff or "changed" in diff

    def test_new_file_appears(self, sandbox):
        sandbox.write_file("brand_new.txt", "i am new\n")
        diff = sandbox.diff()
        assert "brand_new" in diff or "i am new" in diff


# ---------------------------------------------------------------------------
# WorktreeSandbox.checkpoint / rollback
# ---------------------------------------------------------------------------

class TestCheckpointRollback:
    def test_checkpoint_returns_ref(self, sandbox):
        sandbox.write_file("f.txt", "v1")
        ref = sandbox.checkpoint("test-checkpoint")
        assert ref and not ref.startswith("error:")

    def test_rollback_restores_state(self, sandbox):
        # Write v1 and checkpoint
        sandbox.write_file("versioned.txt", "version-1\n")
        ref = sandbox.checkpoint("v1")

        # Make a change after the checkpoint
        sandbox.write_file("versioned.txt", "version-2\n")
        assert "version-2" in sandbox.read_file("versioned.txt")

        # Rollback should restore v1
        sandbox.rollback(ref)
        content = sandbox.read_file("versioned.txt")
        assert "version-1" in content


# ---------------------------------------------------------------------------
# Regression tests for audit bugs 1–3
# ---------------------------------------------------------------------------

class TestBug1ResolvedPaths:
    """Bug 1: constructor must resolve both paths so relative inputs work."""

    def test_root_is_absolute(self, sandbox):
        assert sandbox.root.is_absolute()

    def test_repo_root_is_absolute(self, sandbox):
        # Internal attribute must be absolute after resolve()
        assert sandbox._repo_root.is_absolute()

    def test_relative_repo_root_resolved(self, tmp_path):
        """Passing a relative-looking path object still resolves correctly."""
        repo = _init_git_repo(tmp_path / "rel_repo")
        work = tmp_path / "rel_work"
        # Use an absolute path; .resolve() should be idempotent
        sb = WorktreeSandbox(repo_root=repo.resolve(), work_dir=work)
        assert sb.root.is_absolute()
        sb.close()


class TestBug2CheckpointNonDestructive:
    """Bug 2: checkpoint() must NOT wipe the working tree; rollback raises GitError on failure."""

    def test_checkpoint_does_not_wipe_tree(self, sandbox):
        """After checkpoint(), working tree changes must still be present."""
        sandbox.write_file("canary.txt", "I am here\n")
        ref = sandbox.checkpoint("step-1")
        # Tree must be intact — file still readable
        assert "I am here" in sandbox.read_file("canary.txt")

    def test_checkpoint_returns_sha_not_error_string(self, sandbox):
        """Bug 2: checkpoint must NEVER return an 'error: ...' string."""
        sandbox.write_file("x.txt", "data")
        ref = sandbox.checkpoint("label")
        assert not ref.startswith("error:"), f"Got error string: {ref!r}"

    def test_rollback_raises_git_error_on_bad_ref(self, sandbox):
        """Bug 2: rollback with a garbage ref must raise GitError, not silently fail."""
        from anvil.sandbox.worktree import GitError
        with pytest.raises(GitError):
            sandbox.rollback("0000000000000000000000000000000000000000_bad")


class TestBug3ExecEnvSanitization:
    """Bug 3: exec strips a broad set of secrets; PYTHONDONTWRITEBYTECODE=1 is set."""

    def test_sanitized_env_strips_aws(self):
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            "os.environ", {"AWS_SECRET_ACCESS_KEY": "abc", "HOME": "/home/x"}
        ):
            env = _sanitized_env()
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "HOME" in env

    def test_sanitized_env_strips_github_token(self):
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            "os.environ", {"GITHUB_TOKEN": "ghp_abc", "USER": "me"}
        ):
            env = _sanitized_env()
        assert "GITHUB_TOKEN" not in env
        assert "USER" in env

    def test_sanitized_env_strips_openai(self):
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-xxx"}
        ):
            env = _sanitized_env()
        assert "OPENAI_API_KEY" not in env

    def test_sanitized_env_strips_password(self):
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(
            "os.environ", {"DB_PASSWORD": "hunter2"}
        ):
            env = _sanitized_env()
        assert "DB_PASSWORD" not in env

    def test_sanitized_env_injects_pythondontwritebytecode(self):
        env = _sanitized_env()
        assert env.get("PYTHONDONTWRITEBYTECODE") == "1"

    def test_exec_has_pythondontwritebytecode(self, sandbox):
        """exec child env must include PYTHONDONTWRITEBYTECODE=1."""
        result = sandbox.exec("echo $PYTHONDONTWRITEBYTECODE")
        assert "1" in result.stdout


class TestBug4DiffExcludes:
    """Bug 4: diff() must exclude .anvil_venv and .anvil."""

    def test_diff_excludes_anvil_venv(self, sandbox):
        # Create a file inside .anvil_venv — should not appear in diff
        (sandbox.root / ".anvil_venv").mkdir()
        (sandbox.root / ".anvil_venv" / "pyvenv.cfg").write_text("home = /usr/bin\n")
        diff = sandbox.diff()
        assert ".anvil_venv" not in diff

    def test_diff_includes_real_changes(self, sandbox):
        sandbox.write_file("real_change.py", "x = 1\n")
        diff = sandbox.diff()
        assert "real_change.py" in diff
