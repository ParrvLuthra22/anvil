"""WorktreeSandbox: run agent commands in an isolated git worktree (or plain copy)."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from anvil.sandbox.base import ExecResult, Sandbox

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHAR_CAP = 8000      # characters; each half gets half
_STASH_PREFIX = "anvil-checkpoint"

# Maximum raw bytes to read from a file before truncating to avoid OOM.
_MAX_READ_BYTES = 2 * 1024 * 1024   # 2 MB
# Bytes threshold above which we check for binary content.
_BINARY_SNIFF_BYTES = 8192

# Paths excluded from diff() — venv and anvil internals are never patch-relevant.
# These are passed as ``:(exclude)`` pathspecs to git-diff AND filtered from
# untracked file listings, so *.egg-info, __pycache__, *.pyc, and the venv
# never appear in what the model sees or in the generated patch.
_DIFF_EXCLUDE = (
    ".anvil_venv",
    ".anvil",
    "*.egg-info",
    "__pycache__",
    "*.pyc",
    "*.pyo",
)

# Regex that matches *any* env var name carrying a secret.
# Covers: anything ending with KEY, TOKEN, SECRET, PASSWORD, CREDENTIAL,
# plus AWS_*, GITHUB_*, GH_*, ANTHROPIC_*, OPENAI_*, AI_*, GROQ_*, GOOGLE_*.
_SECRET_KEY_RE = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)$"
    r"|(^AWS_|^GITHUB_|^GH_|^ANTHROPIC_|^OPENAI_|^AI_|^GROQ_|^GOOGLE_)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Typed exception for git failures
# ---------------------------------------------------------------------------

class GitError(RuntimeError):
    """Raised when a git operation that must succeed fails."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"git {' '.join(cmd[1:] if cmd[0]=='git' else cmd)} failed "
            f"(exit {returncode}): {stderr.strip()[:300]}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap_output(text: str, cap: int) -> str:
    """Truncate *text* to *cap* chars using head+tail with an omission marker."""
    if len(text) <= cap:
        return text
    half = cap // 2
    omitted = len(text) - 2 * half
    return text[:half] + f"\n... [{omitted} chars omitted] ...\n" + text[-half:]


def _sanitized_env() -> dict[str, str]:
    """Return os.environ minus *all* secret-looking variables.

    Strips any var whose name matches ``_SECRET_KEY_RE`` (keys, tokens,
    passwords, credentials, plus AWS/GitHub/AI prefixes).
    Also injects ``PYTHONDONTWRITEBYTECODE=1``.
    """
    env = {k: v for k, v in os.environ.items()
           if not _SECRET_KEY_RE.search(k)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _git_env() -> dict[str, str]:
    """Return the environment for git subprocesses.

    Inherits the sanitized environment and always sets a minimal git identity so
    that ``git stash create`` and any commit we make work on machines where
    ``user.name`` / ``user.email`` are not configured globally.
    """
    env = _sanitized_env()
    # Only set these if not already defined — respect the developer's identity.
    env.setdefault("GIT_AUTHOR_NAME", "anvil-agent")
    env.setdefault("GIT_AUTHOR_EMAIL", "anvil@localhost")
    env.setdefault("GIT_COMMITTER_NAME", "anvil-agent")
    env.setdefault("GIT_COMMITTER_EMAIL", "anvil@localhost")
    return env


def _is_excluded(path: str) -> bool:
    """Return True if *path* matches any entry in ``_DIFF_EXCLUDE``.

    Handles both plain prefix directories (e.g. ``.anvil_venv``) and glob
    patterns (e.g. ``*.pyc``, ``__pycache__``).
    """
    import fnmatch
    parts = path.replace("\\", "/").split("/")
    for exc in _DIFF_EXCLUDE:
        if "*" in exc:
            # Match any segment of the path against the glob
            if any(fnmatch.fnmatch(part, exc) for part in parts):
                return True
        else:
            # Plain prefix directory or exact match
            if path == exc or path.startswith(exc + "/") or exc in parts:
                return True
    return False


def _run_git(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and return the CompletedProcess (check=False)."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        env=_git_env(),
    )


def _require_git(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and raise :class:`GitError` if it fails."""
    result = _run_git(args, cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        raise GitError(["git"] + args, result.returncode, result.stderr)
    return result



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

        Both paths are resolved to absolute paths immediately so that any
        later ``git worktree add`` (run with ``cwd=repo_root``) operates on
        the correct directories regardless of the caller's working directory.

        Falls back to ``shutil.copytree`` if the repo cannot be used as a
        worktree (e.g. a shallow clone without a local branch).

        Args:
            repo_root: The directory returned by ``clone_repo`` — the repo clone.
            work_dir:  Where the isolated copy will live.  Created if absent.
            char_cap:  Maximum characters for combined stdout+stderr per exec call.
        """
        # Bug 1: resolve both paths so relative inputs never land in the wrong place.
        self._repo_root = repo_root.resolve()
        self._work_dir = work_dir.resolve()
        self._char_cap = char_cap
        self._use_worktree = False

        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Try git worktree add (cwd=repo_root so git finds the right .git)
        result = _run_git(
            ["worktree", "add", "--detach", str(self._work_dir)],
            cwd=self._repo_root,
            timeout=30,
        )
        if result.returncode == 0:
            self._use_worktree = True
        else:
            # Fallback: plain copy (works even with shallow clones)
            if self._work_dir.exists():
                shutil.rmtree(self._work_dir)
            shutil.copytree(self._repo_root, self._work_dir)

        # Record the baseline commit so diff() and rollback can compare against it
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
        if os.path.isabs(path):
            raise PermissionError(
                f"Absolute path {path!r} is not allowed inside the sandbox."
            )
        candidate = self._work_dir / path
        real_candidate = Path(os.path.realpath(candidate))
        real_root = Path(os.path.realpath(self._work_dir))
        try:
            real_candidate.relative_to(real_root)
        except ValueError:
            raise PermissionError(
                f"Path {path!r} escapes the sandbox root (resolves to {real_candidate})."
            )
        return real_candidate

    def _venv_bin(self) -> str | None:
        """Return the .anvil_venv/bin path if it exists, else None."""
        venv_bin = self._work_dir / ".anvil_venv" / "bin"
        return str(venv_bin) if venv_bin.is_dir() else None

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        """Run *cmd* in the sandbox root with a hard timeout.

        - ``stdin`` is ``/dev/null`` so interactive prompts never hang.
        - The child process group is killed on timeout so no orphans remain.
        - All secret env vars are stripped (see ``_sanitized_env``).
        - ``PYTHONDONTWRITEBYTECODE=1`` is injected.
        - If ``.anvil_venv/bin`` exists it is prepended to ``PATH`` so
          venv-installed tools (pytest, etc.) are found automatically.
        - stdout and stderr are each capped at ``char_cap // 2`` characters.
        """
        start = time.monotonic()
        env = _sanitized_env()

        # Bug 4: prepend venv bin to PATH when it exists
        venv_bin = self._venv_bin()
        if venv_bin:
            env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                cwd=self._work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,   # Bug 3: no interactive stdin
                text=True,
                start_new_session=True,     # own process group → os.killpg works
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

        raw_bytes = abs_path.read_bytes()
        truncated = False
        if len(raw_bytes) > _MAX_READ_BYTES:
            raw_bytes = raw_bytes[:_MAX_READ_BYTES]
            truncated = True

        sniff = raw_bytes[:_BINARY_SNIFF_BYTES]
        if b"\x00" in sniff:
            size_kb = abs_path.stat().st_size // 1024
            return f"(binary file — {size_kb} KB, cannot display as text)"

        text = raw_bytes.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n")

        if truncated:
            text += (
                f"\n... [file truncated — showing first {_MAX_READ_BYTES // 1024} KB; "
                "use start/end to read later sections] ..."
            )

        lines = text.splitlines(keepends=True)

        if start is None and end is None:
            return "".join(lines)

        lo = (start - 1) if start is not None else 0
        hi = end if end is not None else len(lines)
        lo = max(lo, 0)
        hi = min(hi, len(lines))
        return "".join(lines[lo:hi])

    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite *path* (relative to root) with *content*.

        Raises:
            PermissionError: If the path escapes the sandbox root.
        """
        abs_path = self._safe_path(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

    def diff(self) -> str:
        """Return a unified diff of all changes versus the baseline commit.

        Excludes ``.anvil_venv``, ``.anvil``, ``*.egg-info``, ``__pycache__``,
        ``*.pyc`` and ``*.pyo`` so those never appear in what the model sees
        or in the generated patch.
        """
        if not self._baseline_sha:
            return "(no baseline — diff unavailable)"

        # Build exclusion pathspecs for git diff.
        # Prefix-style dirs use a plain prefix; glob patterns use :!glob syntax.
        exclude_args: list[str] = []
        for exc in _DIFF_EXCLUDE:
            if "*" in exc:
                # Glob — must use the glob: magic signature
                exclude_args += [f":(exclude,glob)**/{exc}"]
            else:
                exclude_args += [f":(exclude){exc}"]

        tracked = _run_git(
            ["diff", self._baseline_sha, "--", "."] + exclude_args,
            cwd=self._work_dir,
        )
        parts: list[str] = [tracked.stdout] if tracked.returncode == 0 else []

        untracked = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=self._work_dir,
        )
        for new_file in untracked.stdout.splitlines():
            # Skip paths that match any exclusion pattern
            if _is_excluded(new_file):
                continue
            nf_path = self._work_dir / new_file
            if nf_path.exists():
                r = _run_git(
                    ["diff", "--no-index", "--", "/dev/null", new_file],
                    cwd=self._work_dir,
                )
                parts.append(r.stdout)

        return "".join(parts)

    def checkpoint(self, label: str) -> str:
        """Snapshot current state without touching the working tree.

        Uses ``git stash create`` which creates a stash object and returns its
        SHA **without** modifying the working tree or index.  The SHA is the
        stable ref to pass to :meth:`rollback`.

        Raises:
            GitError: If ``git add -A`` or ``git stash create`` fail.

        Returns:
            The stash SHA (a 40-character hex string), or ``"baseline"`` if
            there are no changes to snapshot (already at the baseline).
        """
        # Stage everything so stash create sees untracked files too
        _require_git(["add", "-A"], cwd=self._work_dir)

        result = _run_git(
            ["stash", "create", f"anvil-checkpoint:{label}"],
            cwd=self._work_dir,
        )
        if result.returncode != 0:
            raise GitError(
                ["git", "stash", "create"],
                result.returncode,
                result.stderr,
            )

        sha = result.stdout.strip()
        if not sha:
            # No changes vs HEAD → return baseline SHA as the ref
            return self._baseline_sha or "baseline"
        return sha

    def rollback(self, ref: str) -> None:
        """Restore the working tree to the state captured by *ref*.

        ``ref`` is either a stash SHA returned by :meth:`checkpoint` or
        ``"baseline"`` / the baseline commit SHA, in which case the tree is
        reset to the original clone state.

        Raises:
            GitError: If the reset or apply fails.
        """
        # Hard-reset to HEAD, then clean untracked files
        _require_git(["reset", "--hard", "HEAD"], cwd=self._work_dir)
        _require_git(["clean", "-fd"], cwd=self._work_dir)

        if ref in ("baseline", self._baseline_sha, ""):
            # Already at baseline after the hard reset — nothing more to do.
            return

        result = _run_git(["stash", "apply", ref], cwd=self._work_dir)
        if result.returncode != 0:
            raise GitError(
                ["git", "stash", "apply", ref],
                result.returncode,
                result.stderr,
            )

    def close(self) -> None:
        """Remove the worktree (or temp directory) and free resources."""
        if self._use_worktree:
            _run_git(
                ["worktree", "remove", "--force", str(self._work_dir)],
                cwd=self._repo_root,
            )
        else:
            shutil.rmtree(self._work_dir, ignore_errors=True)
