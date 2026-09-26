"""Dependency installation helper for the sandbox.

``ensure_deps`` runs the repo's install command inside the sandbox with a
hard 5-minute cap.  For Python repos it creates an isolated venv *inside*
the sandbox work directory so the harness's own environment is never
modified.  For JS/Go/Rust/Java it uses the toolchain on PATH, skipping
cleanly with a clear message if the toolchain is absent.

On any failure it returns a :class:`DepsResult` — it never raises.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from anvil.repo.profile import RepoProfile
from anvil.sandbox.base import Sandbox

log = logging.getLogger(__name__)

_INSTALL_TIMEOUT = 300   # 5 minutes maximum for any install command
_VENV_DIR = ".anvil_venv"  # relative to sandbox root; separate from the harness venv

# Interpreter candidates probed in order (newest first).
_PYTHON_CANDIDATES = ["python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python3"]


@dataclass
class DepsResult:
    """Outcome of a dependency installation attempt."""

    ok: bool
    """True if installation succeeded (exit 0, no timeout)."""

    report: str
    """Human-readable summary of what happened."""

    venv_python: str | None = None
    """Relative path to the Python interpreter in the created venv, if applicable."""

    skipped: bool = False
    """True if installation was skipped (no toolchain / no install cmd)."""


def ensure_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Install dependencies required to build and test the repo.

    Never raises — all errors are captured in :attr:`DepsResult.report`.

    Args:
        sandbox: An open :class:`~anvil.sandbox.base.Sandbox` instance.
        profile: The :class:`~anvil.repo.profile.RepoProfile` for the repo.

    Returns:
        A :class:`DepsResult`.
    """
    try:
        return _ensure_deps_inner(sandbox, profile)
    except Exception as exc:  # noqa: BLE001
        return DepsResult(ok=False, report=f"ensure_deps internal error: {exc}")


def _ensure_deps_inner(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    if not profile.install_cmd:
        return DepsResult(ok=True, report="No install command detected — skipping.", skipped=True)

    lang = profile.primary_language

    if lang == "python":
        return _ensure_python_deps(sandbox, profile)
    if lang in ("javascript", "typescript"):
        return _ensure_js_deps(sandbox, profile)
    if lang == "go":
        return _ensure_go_deps(sandbox, profile)
    if lang == "rust":
        return _ensure_rust_deps(sandbox, profile)
    if lang == "java":
        return _ensure_java_deps(sandbox, profile)

    return _ensure_generic_deps(sandbox, profile)


# ---------------------------------------------------------------------------
# Python interpreter selection (Task 3)
# ---------------------------------------------------------------------------

def _parse_requires_python(sandbox: Sandbox) -> str | None:
    """Parse requires-python from pyproject.toml or setup.cfg in the sandbox root.

    Returns the raw specifier string, or None if no constraint found.
    Defaults to ">=3.9" when a Python project exists but no constraint is stated.
    """
    # pyproject.toml: requires-python = ">=3.10"
    pyproject = sandbox.root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'requires-python\s*=\s*["\']([^"\'\']+)["\']', text)
        if m:
            return m.group(1)
        # pyproject exists → it's a Python project; default minimum
        return ">=3.9"

    # setup.cfg: python_requires = >=3.10
    setup_cfg = sandbox.root / "setup.cfg"
    if setup_cfg.exists():
        text = setup_cfg.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'python_requires\s*=\s*([^\n]+)', text)
        if m:
            return m.group(1).strip()
        return ">=3.9"

    return None  # no Python metadata found


def _pick_python_interpreter(sandbox: Sandbox) -> str | None:
    """Return the first interpreter that satisfies requires-python and can create a venv.

    Probes ``python3.13`` down to ``python3`` in order.
    For each candidate:
    1. Check it is on PATH (``command -v``).
    2. Verify its version satisfies ``requires-python`` using packaging.specifiers.
    3. Verify it can create a venv (``-m venv --help``).

    Returns:
        The interpreter name (e.g. ``"python3.11"``), or ``None`` if none qualifies.
    """
    spec_str = _parse_requires_python(sandbox)
    specifier = None
    if spec_str:
        try:
            from packaging.specifiers import SpecifierSet
            specifier = SpecifierSet(spec_str)
        except Exception as e:
            log.warning("Could not parse requires-python %r: %s", spec_str, e)
            specifier = None

    for interp in _PYTHON_CANDIDATES:
        # 1. Is it on PATH?
        probe = sandbox.exec(f"command -v {interp}", timeout=5)
        if probe.exit_code != 0:
            continue

        # 2. Check version if constraint exists
        ver_result = sandbox.exec(
            f'{interp} -c "import sys; print(str(sys.version_info.major) + \'.\' + str(sys.version_info.minor))"',
            timeout=5,
        )
        if ver_result.exit_code != 0:
            continue
            
        actual_version = ver_result.stdout.strip()
        
        if specifier is not None:
            if not specifier.contains(actual_version):
                log.debug(
                    "Skipping %s (version %s does not satisfy %s)",
                    interp, actual_version, spec_str
                )
                continue

        # 3. Can it create a venv?
        venv_probe = sandbox.exec(f"{interp} -m venv --help", timeout=10)
        if venv_probe.exit_code != 0:
            log.debug("Skipping %s — cannot create venv", interp)
            continue

        log.info("Selected Python interpreter: %s", interp)
        return interp

    return None


def _ensure_python_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Create an isolated venv using the best available interpreter and install deps."""
    venv_path = _VENV_DIR
    venv_python = f"{venv_path}/bin/python"
    venv_pip = f"{venv_path}/bin/pip"

    # Task 3: pick the right interpreter
    interp = _pick_python_interpreter(sandbox)
    if interp is None:
        spec_str = _parse_requires_python(sandbox)
        reason = f"requires Python {spec_str}" if spec_str else "requires Python"
        return DepsResult(
            ok=False,
            report=(
                f"No suitable Python interpreter found ({reason}). "
                f"Tried: {', '.join(_PYTHON_CANDIDATES)}."
            ),
            skipped=True,
        )

    # Step 1: Create the venv
    log.info("Creating isolated Python venv at %s/%s using %s", sandbox.root, venv_path, interp)
    create_result = sandbox.exec(f"{interp} -m venv {venv_path}", timeout=60)
    if create_result.exit_code != 0 or create_result.timed_out:
        return DepsResult(
            ok=False,
            report=(
                f"Failed to create Python venv with {interp!r}.\n"
                f"stdout: {create_result.stdout[:500]}\n"
                f"stderr: {create_result.stderr[:500]}"
            ),
        )

    # Step 2: Upgrade pip silently
    sandbox.exec(f"{venv_pip} install --quiet --upgrade pip", timeout=60)

    # Step 3: Run the install command using the venv's pip
    install_cmd = profile.install_cmd
    assert install_cmd is not None
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
            venv_python=venv_python,
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

    # Step 4: Guarantee the test framework is in the venv
    test_framework_pkg = _test_framework_package(profile)
    if test_framework_pkg:
        sandbox.exec(
            f"{venv_pip} install --quiet {test_framework_pkg}",
            timeout=60,
        )

    return DepsResult(
        ok=True,
        report=f"Python deps installed into {venv_path}/ using {interp}.",
        venv_python=venv_python,
    )


def _ensure_js_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Install JS/TS deps with npm ci (preferred) or npm install."""
    # Check if npm is available
    probe = sandbox.exec("command -v npm", timeout=5)
    if probe.exit_code != 0:
        return DepsResult(
            ok=False,
            report="npm not found on PATH — skipping JS dependency install.",
            skipped=True,
        )

    # Try `npm ci` first (reproducible from lock file), fall back to `npm install`
    lock_exists = (sandbox.root / "package-lock.json").exists() or \
                  (sandbox.root / "yarn.lock").exists() or \
                  (sandbox.root / "pnpm-lock.yaml").exists()

    cmd = "npm ci --prefer-offline 2>&1 || npm install" if lock_exists else "npm install"
    log.info("Running JS install: %s", cmd)
    result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)
    return _check_result(result, cmd)


def _ensure_go_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Download Go module dependencies."""
    probe = sandbox.exec("command -v go", timeout=5)
    if probe.exit_code != 0:
        return DepsResult(
            ok=False,
            report="go toolchain not found on PATH — skipping Go dependency download.",
            skipped=True,
        )
    cmd = "go mod download"
    log.info("Running Go install: %s", cmd)
    result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)
    return _check_result(result, cmd)


def _ensure_rust_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Pre-fetch Rust crate dependencies."""
    probe = sandbox.exec("command -v cargo", timeout=5)
    if probe.exit_code != 0:
        return DepsResult(
            ok=False,
            report="cargo not found on PATH — skipping Rust dependency fetch.",
            skipped=True,
        )
    cmd = "cargo fetch"
    log.info("Running Rust fetch: %s", cmd)
    result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)
    return _check_result(result, cmd)


def _ensure_java_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Resolve Java dependencies via Maven or Gradle."""
    cmd = profile.install_cmd
    assert cmd is not None

    # Detect toolchain from install_cmd
    tool = "mvn" if "mvn" in cmd else "gradle"
    probe = sandbox.exec(f"command -v {tool}", timeout=5)
    if probe.exit_code != 0:
        return DepsResult(
            ok=False,
            report=f"{tool} not found on PATH — skipping Java dependency resolve.",
            skipped=True,
        )

    # Override the install_cmd to use dependency:resolve for speed if possible
    if "mvn" in cmd and "dependency:resolve" not in cmd:
        cmd = "mvn -q -DskipTests dependency:resolve"
    log.info("Running Java install: %s", cmd)
    result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)
    return _check_result(result, cmd)


def _ensure_generic_deps(sandbox: Sandbox, profile: RepoProfile) -> DepsResult:
    """Run the install command as-is for unrecognised languages."""
    cmd = profile.install_cmd
    assert cmd is not None
    log.info("Running install: %s", cmd)
    result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)
    return _check_result(result, cmd)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _check_result(result, cmd: str) -> DepsResult:
    """Convert an ExecResult into a DepsResult."""
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
    return DepsResult(ok=True, report=f"Dependencies installed. ({cmd})")


def _test_framework_package(profile: RepoProfile) -> str | None:
    """Return the pip package name for the repo's test framework, if known."""
    mapping = {
        "pytest":   "pytest",
        "nose":     "nose2",
        "nose2":    "nose2",
        "unittest": None,
    }
    if not profile.test_framework:
        return "pytest"
    return mapping.get(profile.test_framework.lower(), profile.test_framework)
