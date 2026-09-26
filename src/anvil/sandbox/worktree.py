"""WorktreeSandbox: run agent commands in an isolated git worktree (or plain copy)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Maximum raw bytes to read from a file before truncating to avoid OOM.
_MAX_READ_BYTES = 2 * 1024 * 1024   # 2 MB
# Bytes threshold above which we check for binary content.
_BINARY_SNIFF_BYTES = 8192

from anvil.sandbox.base import ExecResult, Sandbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHAR_CAP = 8000      # characters; each half gets half
_STASH_PREFIX = "anvil-checkpoint"


def _cap_output(text: str, cap: int) -> str:
    """Truncate *text* to *cap* chars using head+tail with an omission marker."""
    if len(text) <= cap:
        return text
    half = cap // 2
    omitted = len(text) - 2 * half
    return text[:half] + f"\n... [{omitted} chars omitted] ...\n" + text[-half:]


def _sanitized_env() -> dict[str, str]:
    """Return os.environ minus secrets that must not leak into child processes."""
    env = os.environ.copy()
    for key in ("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY", "GROQ_API_KEY"):
        env.pop(key, None)
    return env


def _run_git(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess (check=False)."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# WorktreeSandbox
# ---------------------------------------------------------------------------

class WorktreeSandbox:
    """Sandbox backed by a git worktree (or a plain directory copy as fallback).

    Implements the :class:`~anvil.sandbox.base.Sandbox` protocol.
    """

    def __init__(
        self,
        repo_root: Path,
        work_dir: Path,
        char_cap: int = _DEFAULT_CHAR_CAP,
    ) -> None:
        """Set up a worktree rooted at *work_dir* from the repo at *repo_root*.

        If the repo cannot be used as a worktree (e.g. shallow clone without a
        local branch), falls back to ``shutil.copytree``.

        Args:
            repo_root: The directory returned by ``clone_repo`` — the repo clone.
            work_dir:  Where the isolated copy will live.  Created if absent.
            char_cap:  Maximum characters for combined stdout+stderr per exec call.
        """
        self._repo_root = repo_root
        self._work_dir = work_dir
        self._char_cap = char_cap
        self._use_worktree = False

        work_dir.mkdir(parents=True, exist_ok=True)

        # Try git worktree add
        result = _run_git(
            ["worktree", "add", "--detach", str(work_dir)],
            cwd=repo_root,
            timeout=30,
        )
        if result.returncode == 0:
            self._use_worktree = True
        else:
            # Fallback: plain copy (works even with shallow clones)
            if work_dir.exists():
                shutil.rmtree(work_dir)
            shutil.copytree(repo_root, work_dir)

        # Record the baseline commit so diff() can compare against it
        rev = _run_git(["rev-parse", "HEAD"], cwd=self._work_dir)
        self._baseline_sha: str = rev.stdout.strip() if rev.returncode == 0 else ""

    # ------------------------------------------------------------------
    # Sandbox protocol
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """The working directory exposed to the agent."""
        return self._work_dir

    def _safe_path(self, path: str) -> Path:
        """Resolve *path* (including symlinks) relative to root and reject escapes.

        Uses ``os.path.realpath`` so that a symlink pointing outside the sandbox
        root is caught, not just ``..``-based traversal.

        Raises:
            PermissionError: If *path* resolves to a location outside the sandbox root.
        """
        # Reject absolute paths outright
        if os.path.isabs(path):
            raise PermissionError(
                f"Absolute path {path!r} is not allowed inside the sandbox."
            )
        candidate = self._work_dir / path
        # realpath follows all symlinks; resolve() doesn't on all Python versions
        real_candidate = Path(os.path.realpath(candidate))
        real_root = Path(os.path.realpath(self._work_dir))
        try:
            real_candidate.relative_to(real_root)
        except ValueError:
            raise PermissionError(
                f"Path {path!r} escapes the sandbox root (resolves to {real_candidate})."
            )
        return real_candidate

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        """Run *cmd* in the sandbox root with a hard timeout.

        The child process group is killed on timeout so no orphans remain.
        ``AI_API_KEY`` and other secret env vars are stripped from the child env.
        stdout and stderr are each capped at ``char_cap // 2`` characters.
        """
        start = time.monotonic()
        env = _sanitized_env()

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self._work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,   # own process group → os.killpg works
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                timed_out = False
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                # Kill the whole process group
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
                timed_out = True
                exit_code = -1

        except Exception as exc:  # noqa: BLE001
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"exec error: {exc}",
                timed_out=False,
                duration=time.monotonic() - start,
            )

        duration = time.monotonic() - start
        half_cap = self._char_cap // 2
        return ExecResult(
            exit_code=exit_code,
            stdout=_cap_output(stdout, half_cap),
            stderr=_cap_output(stderr, half_cap),
            timed_out=timed_out,
            duration=duration,
        )

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read a file, optionally restricted to a 1-indexed inclusive line range.

        Handles binary files, large files (capped at 2 MB), CRLF line endings
        and unicode (invalid bytes replaced with the U+FFFD replacement character).

        Args:
            path:  Path relative to sandbox root.
            start: First line to return (1-indexed, inclusive).
            end:   Last line to return (1-indexed, inclusive).

        Returns:
            The requested file content (or slice).

        Raises:
            FileNotFoundError: If the path does not exist.
            PermissionError:   If the path escapes the sandbox root.
            IsADirectoryError: If *path* points to a directory.
        """
        abs_path = self._safe_path(path)

        if not abs_path.exists():
            raise FileNotFoundError(f"No such file: {path!r}")
        if abs_path.is_dir():
            raise IsADirectoryError(f"{path!r} is a directory, not a file.")

        # Read raw bytes so we can detect binary content and cap size.
        raw_bytes = abs_path.read_bytes()
        truncated = False
        if len(raw_bytes) > _MAX_READ_BYTES:
            raw_bytes = raw_bytes[:_MAX_READ_BYTES]
            truncated = True

        # Binary detection: look for null bytes in the first sniff window.
        sniff = raw_bytes[:_BINARY_SNIFF_BYTES]
        if b"\x00" in sniff:
            size_kb = abs_path.stat().st_size // 1024
            return f"(binary file — {size_kb} KB, cannot display as text)"

        text = raw_bytes.decode("utf-8", errors="replace")
        # Normalise CRLF → LF so line counting is consistent cross-platform.
        text = text.replace("\r\n", "\n")

        if truncated:
            text += (
                f"\n... [file truncated — showing first {_MAX_READ_BYTES // 1024} KB; "
                "use start/end to read later sections] ..."
            )

        lines = text.splitlines(keepends=True)

        if start is None and end is None:
            return "".join(lines)

        # Convert from 1-indexed inclusive to 0-indexed Python slice
        lo = (start - 1) if start is not None else 0
        hi = end if end is not None else len(lines)
        lo = max(lo, 0)
        hi = min(hi, len(lines))
        return "".join(lines[lo:hi])

    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite *path* (relative to root) with *content*.

        Parent directories are created automatically.

        Raises:
            PermissionError: If the path escapes the sandbox root.
        """
        abs_path = self._safe_path(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

    def diff(self) -> str:
        """Return a unified diff of all changes versus the baseline commit.

        Includes untracked new files so the agent can see files it created.
        """
        if not self._baseline_sha:
            return "(no baseline — diff unavailable)"

        # Tracked changes
        tracked = _run_git(
            ["diff", self._baseline_sha, "--", "."],
            cwd=self._work_dir,
        )
        parts: list[str] = [tracked.stdout] if tracked.returncode == 0 else []

        # Untracked new files — add to index temporarily then diff
        untracked = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=self._work_dir,
        )
        for new_file in untracked.stdout.splitlines():
            nf_path = self._work_dir / new_file
            if nf_path.exists():
                r = _run_git(
                    ["diff", "--no-index", "--", "/dev/null", new_file],
                    cwd=self._work_dir,
                )
                # git diff --no-index exits 1 when there's a diff (normal)
                parts.append(r.stdout)

        return "".join(parts)

    def checkpoint(self, label: str) -> str:
        """Snapshot current state via ``git stash`` and return the stash ref.

        Returns:
            A stash ref string like ``stash@{0}``, or an error string prefixed
            with ``"error:"`` if stashing fails.
        """
        # Stage everything so stash captures untracked files too
        _run_git(["add", "-A"], cwd=self._work_dir)
        msg = f"{_STASH_PREFIX}:{label}"
        result = _run_git(
            ["stash", "push", "--include-untracked", "-m", msg],
            cwd=self._work_dir,
        )
        if result.returncode != 0:
            return f"error: git stash failed — {result.stderr.strip()}"

        # Find the ref of the stash we just pushed
        list_result = _run_git(["stash", "list", "--format=%gd %s"], cwd=self._work_dir)
        for line in list_result.stdout.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2 and msg in parts[1]:
                return parts[0]

        return "stash@{0}"  # best guess if parsing fails

    def rollback(self, ref: str) -> None:
        """Restore to the state captured by *ref* (a stash ref like ``stash@{0}``).

        The current working-tree state is discarded.
        """
        # Discard any current changes
        _run_git(["checkout", "--", "."], cwd=self._work_dir)
        _run_git(["clean", "-fd"], cwd=self._work_dir)

        result = _run_git(
            ["stash", "pop", ref],
            cwd=self._work_dir,
        )
        if result.returncode != 0:
            # Try apply as fallback (doesn't remove the stash entry)
            _run_git(["stash", "apply", ref], cwd=self._work_dir)

    def close(self) -> None:
        """Remove the worktree (or temp directory) and free resources."""
        if self._use_worktree:
            _run_git(
                ["worktree", "remove", "--force", str(self._work_dir)],
                cwd=self._repo_root,
            )
        else:
            shutil.rmtree(self._work_dir, ignore_errors=True)
