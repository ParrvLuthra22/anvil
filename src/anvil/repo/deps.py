"""Dependency installation helper for the sandbox.

``ensure_deps`` runs the repo's install command inside the sandbox with a
hard timeout. For Python repos it creates an isolated venv *inside* the
sandbox work directory so the harness's own environment is never modified.
On any failure it returns a human-readable report instead of raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from anvil.repo.profile import RepoProfile
from anvil.sandbox.base import Sandbox

log = logging.getLogger(__name__)

_INSTALL_TIMEOUT = 300   # 5 minutes maximum for any install command
_VENV_DIR = ".anvil_venv"  # relative to sandbox root; separate from the harness venv


@dataclass
class DepsResult:
    """Outcome of a dependency installation attempt."""

    ok: bool
    """True if installation succeeded (exit 0, no timeout)."""

    report: str
    """Human-readable summary of what happened."""

    venv_python: str | None = None
    """Absolute path to the Python interpreter in the created venv, if applicable."""


def ensure_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Install dependencies required to build and test the repo.

    For Python repos, creates a fresh venv inside the sandbox root at
    ``.anvil_venv/`` so the harness's own ``sys.path`` is never altered.
    The venv path is returned in ``DepsResult.venv_python`` so callers
    can prefix commands (e.g. ``.anvil_venv/bin/pytest``).

    For all other languages, runs ``profile.install_cmd`` directly
    (the sandbox env already has the language toolchain on PATH).

    If ``profile.install_cmd`` is ``None``, returns ``ok=True`` immediately
    with a note that no install is needed.

    On timeout or non-zero exit the run continues: the agent can still
    attempt to read and patch code even if install fails.

    Args:
        sandbox: An open :class:`~anvil.sandbox.base.Sandbox` instance.
        profile: The :class:`~anvil.repo.profile.RepoProfile` for the repo.

    Returns:
        A :class:`DepsResult` — never raises.
    """
    if not profile.install_cmd:
        return DepsResult(ok=True, report="No install command detected — skipping.")

    is_python = profile.primary_language == "python"

    if is_python:
        return _ensure_python_deps(sandbox, profile)
    else:
        return _ensure_generic_deps(sandbox, profile)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_python_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Create an isolated venv and install Python dependencies into it."""
    venv_path = _VENV_DIR
    venv_python = f"{venv_path}/bin/python"
    venv_pip = f"{venv_path}/bin/pip"

    # Step 1: Create the venv
    log.info("Creating isolated Python venv at %s/%s", sandbox.root, venv_path)
    create_result = sandbox.exec(
        f"python3 -m venv {venv_path}",
        timeout=60,
    )
    if create_result.exit_code != 0 or create_result.timed_out:
        return DepsResult(
            ok=False,
            report=(
                f"Failed to create Python venv.\n"
                f"stdout: {create_result.stdout[:500]}\n"
                f"stderr: {create_result.stderr[:500]}"
            ),
        )

    # Step 2: Upgrade pip silently
    sandbox.exec(f"{venv_pip} install --quiet --upgrade pip", timeout=60)

    # Step 3: Run the install command using the venv's pip
    # Replace bare 'pip install' with venv-scoped pip so the right env is used.
    install_cmd = profile.install_cmd
    install_cmd = install_cmd.replace("pip install", f"{venv_pip} install", 1)

    log.info("Running Python install: %s", install_cmd)
    result = sandbox.exec(install_cmd, timeout=_INSTALL_TIMEOUT)

    if result.timed_out:
        return DepsResult(
            ok=False,
            report=(
                f"Install timed out after {_INSTALL_TIMEOUT}s. "
                f"Command: {install_cmd}\n"
                f"Partial stdout: {result.stdout[:400]}"
            ),
            venv_python=venv_python,  # still return path; might be partially installed
        )

    if result.exit_code != 0:
        return DepsResult(
            ok=False,
            report=(
                f"Install failed (exit {result.exit_code}).\n"
                f"Command: {install_cmd}\n"
                f"stdout: {result.stdout[:400]}\n"
                f"stderr: {result.stderr[:400]}"
            ),
            venv_python=venv_python,
        )

    # Step 4: Also install the test framework itself — projects sometimes don't
    # list it as a direct dev dependency, so we guarantee it's in the venv.
    test_framework_pkg = _test_framework_package(profile)
    if test_framework_pkg:
        sandbox.exec(
            f"{venv_pip} install --quiet {test_framework_pkg}",
            timeout=60,
        )

    return DepsResult(
        ok=True,
        report=f"Python deps installed into {venv_path}/.",
        venv_python=venv_python,
    )


def _test_framework_package(profile: RepoProfile) -> str | None:
    """Return the pip package name for the repo's test framework, if known."""
    mapping = {
        "pytest":   "pytest",
        "nose":     "nose2",
        "nose2":    "nose2",
        "unittest": None,   # stdlib, no install needed
    }
    if not profile.test_framework:
        return "pytest"  # default
    return mapping.get(profile.test_framework.lower(), profile.test_framework)


def _ensure_generic_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Run the install command as-is for non-Python repos."""
    cmd = profile.install_cmd
    assert cmd is not None  # guarded by caller

    log.info("Running install: %s", cmd)
    result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)

    if result.timed_out:
        return DepsResult(
            ok=False,
            report=(
                f"Install timed out after {_INSTALL_TIMEOUT}s.\n"
                f"Command: {cmd}\n"
                f"Partial stdout: {result.stdout[:400]}"
            ),
        )

    if result.exit_code != 0:
        return DepsResult(
            ok=False,
            report=(
                f"Install failed (exit {result.exit_code}).\n"
                f"Command: {cmd}\n"
                f"stdout: {result.stdout[:400]}\n"
                f"stderr: {result.stderr[:400]}"
            ),
        )

    return DepsResult(
        ok=True,
        report=f"Dependencies installed. ({cmd})",
    )

