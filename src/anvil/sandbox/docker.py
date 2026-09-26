"""DockerSandbox — EXPERIMENTAL opt-in backend.

Runs agent commands inside an ephemeral Docker container bind-mounted on a
local worktree.  Docker is **not required**; the default sandbox is
:class:`~anvil.sandbox.worktree.WorktreeSandbox`.  Use DockerSandbox only
when explicit isolation is needed and Docker is confirmed available.

.. warning::
    This backend is **experimental**.  It adds container pull latency and
    requires Docker to be running.  The evaluator environment may not have
    Docker; prefer WorktreeSandbox unless you specifically need containerisation.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from anvil.sandbox.base import ExecResult, Sandbox
from anvil.sandbox.worktree import (
    GitError,
    _cap_output,
    _run_git,
    _require_git,
    _sanitized_env,
    _DIFF_EXCLUDE,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-language Docker image defaults
# ---------------------------------------------------------------------------

_LANGUAGE_IMAGES: dict[str, str] = {
    "python": "python:3.11-slim",
    "javascript": "node:20-slim",
    "typescript": "node:20-slim",
    "go": "golang:1.22",
    "rust": "rust:1.78",
    "java": "eclipse-temurin:21-jre-alpine",
}
_FALLBACK_IMAGE = "ubuntu:22.04"

_DEFAULT_CHAR_CAP = 8000
_DOCKER_TIMEOUT = 10   # seconds to wait for `docker info`


def _pick_image(primary_language: str, image_overrides: dict[str, str] | None = None) -> str:
    """Return the Docker image to use for *primary_language*."""
    overrides = image_overrides or {}
    return overrides.get(primary_language) or _LANGUAGE_IMAGES.get(primary_language, _FALLBACK_IMAGE)


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=_DOCKER_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------

class DockerSandbox:
    """EXPERIMENTAL: Sandbox that runs commands inside a Docker container.

    The repository working directory is bind-mounted into the container at
    ``/workspace``.  Resource limits (memory, CPUs) are applied to prevent
    runaway processes.

    .. warning::
        Experimental — opt-in only.  Do not use unless Docker is confirmed
        available and containerisation is explicitly required.

    Implements the :class:`~anvil.sandbox.base.Sandbox` protocol.
    """

    def __init__(
        self,
        repo_root: Path,
        work_dir: Path,
        primary_language: str = "python",
        image: str | None = None,
        image_overrides: dict[str, str] | None = None,
        char_cap: int = _DEFAULT_CHAR_CAP,
        memory_limit: str = "512m",
        cpu_limit: str = "1",
    ) -> None:
        """Set up a DockerSandbox.

        Both *repo_root* and *work_dir* are resolved to absolute paths on
        construction (Bug 1 fix) so that ``git worktree add`` — run with
        ``cwd=repo_root`` — operates on the correct directories.

        Args:
            repo_root:         The cloned repository directory.
            work_dir:          Directory to use as the container workspace.
            primary_language:  Used to pick the default Docker image.
            image:             Override the image name entirely.
            image_overrides:   Per-language image overrides (from config.yaml).
            char_cap:          Max characters per exec output.
            memory_limit:      Docker ``--memory`` value.
            cpu_limit:         Docker ``--cpus`` value.
        """
        log.info(
            "[EXPERIMENTAL] DockerSandbox: using Docker backend. "
            "This is opt-in only — prefer WorktreeSandbox for reliability."
        )
        # Bug 1: resolve both paths
        self._repo_root = repo_root.resolve()
        self._work_dir = work_dir.resolve()
        self._char_cap = char_cap
        self._memory = memory_limit
        self._cpus = cpu_limit
        self._image = image or _pick_image(primary_language, image_overrides)

        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Try git worktree; fallback to plain copy
        self._use_worktree = False
        result = _run_git(
            ["worktree", "add", "--detach", str(self._work_dir)],
            cwd=self._repo_root,
        )
        if result.returncode == 0:
            self._use_worktree = True
        else:
            if self._work_dir.exists():
                shutil.rmtree(self._work_dir)
            shutil.copytree(self._repo_root, self._work_dir)

        rev = _run_git(["rev-parse", "HEAD"], cwd=self._work_dir)
        self._baseline_sha: str = rev.stdout.strip() if rev.returncode == 0 else ""

    @property
    def root(self) -> Path:
        """The local path that is bind-mounted into the container."""
        return self._work_dir

    def _safe_path(self, path: str) -> Path:
        """Resolve *path* relative to root, reject absolute paths and escapes."""
        if os.path.isabs(path):
            raise PermissionError(f"Absolute path {path!r} is not allowed inside the sandbox.")
        candidate = self._work_dir / path
        real_candidate = Path(os.path.realpath(candidate))
        real_root = Path(os.path.realpath(self._work_dir))
        try:
            real_candidate.relative_to(real_root)
        except ValueError:
            raise PermissionError(f"Path {path!r} escapes the sandbox root.")
        return real_candidate

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        """Run *cmd* inside a Docker container with resource limits and a timeout.

        - ``stdin`` is ``/dev/null`` so prompts never hang.
        - Secret env vars are stripped from the *host* env passed to the
          Docker CLI (the container itself has a clean env).
        """
        start = time.monotonic()
        docker_cmd = [
            "docker", "run", "--rm",
            "--volume", f"{self._work_dir}:/workspace",
            "--workdir", "/workspace",
            "--memory", self._memory,
            "--cpus", self._cpus,
            "--network", "none",
            self._image,
            "sh", "-c", cmd,
        ]

        env = _sanitized_env()

        try:
            proc = subprocess.Popen(
                docker_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,   # Bug 3: no interactive stdin
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                timed_out = False
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
                timed_out = True
                exit_code = -1

        except Exception as exc:  # noqa: BLE001
            return ExecResult(
                exit_code=-1, stdout="", stderr=f"docker exec error: {exc}",
                timed_out=False, duration=time.monotonic() - start,
            )

        half = self._char_cap // 2
        return ExecResult(
            exit_code=exit_code,
            stdout=_cap_output(stdout, half),
            stderr=_cap_output(stderr, half),
            timed_out=timed_out,
            duration=time.monotonic() - start,
        )

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        """Read a file from the local worktree (no Docker needed for reads)."""
        abs_path = self._safe_path(path)
        if not abs_path.exists():
            raise FileNotFoundError(f"No such file: {path!r}")
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        if start is None and end is None:
            return "".join(lines)
        lo = (start - 1) if start is not None else 0
        hi = end if end is not None else len(lines)
        return "".join(lines[max(lo, 0):min(hi, len(lines))])

    def write_file(self, path: str, content: str) -> None:
        """Write a file to the local worktree (Docker bind-mount makes it visible)."""
        abs_path = self._safe_path(path)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

    def diff(self) -> str:
        """Return unified diff vs baseline commit, excluding .anvil dirs."""
        if not self._baseline_sha:
            return "(no baseline — diff unavailable)"

        exclude_args: list[str] = []
        for exc in _DIFF_EXCLUDE:
            exclude_args += [":(exclude)" + exc]

        tracked = _run_git(
            ["diff", self._baseline_sha, "--", "."] + exclude_args,
            cwd=self._work_dir,
        )
        parts = [tracked.stdout] if tracked.returncode == 0 else []
        untracked = _run_git(
            ["ls-files", "--others", "--exclude-standard"],
            cwd=self._work_dir,
        )
        for nf in untracked.stdout.splitlines():
            if any(nf.startswith(exc) for exc in _DIFF_EXCLUDE):
                continue
            p = self._work_dir / nf
            if p.exists():
                r = _run_git(["diff", "--no-index", "--", "/dev/null", nf], cwd=self._work_dir)
                parts.append(r.stdout)
        return "".join(parts)

    def checkpoint(self, label: str) -> str:
        """Snapshot current state without touching the working tree.

        Uses ``git stash create`` — the working tree is NOT modified.

        Raises:
            GitError: If staging or stash creation fails.

        Returns:
            A stash SHA or the baseline SHA when there are no changes.
        """
        _require_git(["add", "-A"], cwd=self._work_dir)
        result = _run_git(
            ["stash", "create", f"anvil-checkpoint:{label}"],
            cwd=self._work_dir,
        )
        if result.returncode != 0:
            raise GitError(["git", "stash", "create"], result.returncode, result.stderr)
        sha = result.stdout.strip()
        return sha if sha else (self._baseline_sha or "baseline")

    def rollback(self, ref: str) -> None:
        """Restore the working tree to the state captured by *ref*.

        Raises:
            GitError: If the reset or apply fails.
        """
        _require_git(["reset", "--hard", "HEAD"], cwd=self._work_dir)
        _require_git(["clean", "-fd"], cwd=self._work_dir)
        if ref in ("baseline", self._baseline_sha, ""):
            return
        result = _run_git(["stash", "apply", ref], cwd=self._work_dir)
        if result.returncode != 0:
            raise GitError(["git", "stash", "apply", ref], result.returncode, result.stderr)

    def close(self) -> None:
        """Remove the worktree or temp directory."""
        if self._use_worktree:
            _run_git(
                ["worktree", "remove", "--force", str(self._work_dir)],
                cwd=self._repo_root,
            )
        else:
            shutil.rmtree(self._work_dir, ignore_errors=True)
