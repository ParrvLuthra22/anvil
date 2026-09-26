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
import shlex
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
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


def get_commit_date(sandbox: Sandbox, ref: str = "HEAD") -> str:
    """Return the committer timestamp for *ref*, or an empty string on failure."""
    result = sandbox.exec(f"git show -s --format=%cI {shlex.quote(ref)}", timeout=10)
    if result.exit_code != 0 or result.timed_out:
        log.warning("Could not get commit date for %r: %s", ref, result.stderr)
        return ""
    return result.stdout.strip()


def ensure_deps(
    sandbox: Sandbox,
    profile: RepoProfile,
    *,
    as_of: str | date | datetime | None = None,
) -> DepsResult:
    """Install dependencies required to build and test the repo.

    Never raises — all errors are captured in :attr:`DepsResult.report`.

    Args:
        sandbox: An open :class:`~anvil.sandbox.base.Sandbox` instance.
        profile: The :class:`~anvil.repo.profile.RepoProfile` for the repo.
        as_of: Optional PyPI upload-time cutoff. A date includes the full UTC
            calendar day; a timestamp is an exact cutoff. If omitted for a
            Python repo with an install command, the checked-out commit date is used.

    Returns:
        A :class:`DepsResult`.
    """
    try:
        cutoff = _normalize_as_of(as_of) if as_of is not None else None
    except ValueError as exc:
        return DepsResult(ok=False, report=str(exc))
    try:
        return _ensure_deps_inner(sandbox, profile, cutoff)
    except Exception as exc:  # noqa: BLE001
        return DepsResult(ok=False, report=f"ensure_deps internal error: {exc}")


def _ensure_deps_inner(sandbox: Sandbox, profile: RepoProfile, as_of: str | None = None) -> DepsResult:
    if not profile.install_cmd:
        if profile.primary_language == "python" and as_of is not None:
            # Scoring still needs a fresh interpreter and date-pinned test runner
            # even when the repository has no package install command.
            return _ensure_python_deps(sandbox, profile, as_of=as_of)
        return DepsResult(ok=True, report="No install command detected — skipping.", skipped=True)

    lang = profile.primary_language

    if lang == "python":
        return _ensure_python_deps(sandbox, profile, as_of=as_of)
    if as_of is not None:
        return DepsResult(
            ok=False,
            report=f"Date-pinned dependency installation is only supported for Python, not {lang}.",
        )
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


def _normalize_as_of(value: str | date | datetime) -> str:
    """Normalize an inclusive date or exact timestamp to an RFC 3339 UTC cutoff."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.max)
    elif isinstance(value, str):
        raw = value.strip()
        try:
            if len(raw) == 10:
                parsed = datetime.combine(date.fromisoformat(raw), time.max)
            else:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"as_of must be an ISO-8601 date or timestamp: {value!r}") from None
    else:
        raise ValueError("as_of must be an ISO-8601 date or timestamp")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    # Preserve timestamp precision; truncating fractional seconds could make
    # dependencies uploaded just before the requested cutoff unavailable.
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_python_deps(
    sandbox: Sandbox,
    profile: RepoProfile,
    *,
    as_of: str | None = None,
) -> DepsResult:
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

    # Date-aware resolution uses uv's per-artifact upload cutoff. Bootstrap uv
    # inside the target venv if it is unavailable on PATH; never use global pip.
    uv = "uv"
    if as_of is not None:
        uv_probe = sandbox.exec("command -v uv", timeout=5)
        if uv_probe.exit_code != 0:
            bootstrap = sandbox.exec(f"{venv_python} -m pip install --quiet uv", timeout=120)
            if bootstrap.exit_code != 0 or bootstrap.timed_out:
                return DepsResult(
                    ok=False,
                    report=("Date-pinned install requires uv; could not install uv into the isolated "
                            f"environment. stderr: {bootstrap.stderr[:400]}"),
                    venv_python=venv_python,
                )
            uv = f"{venv_path}/bin/uv"
        else:
            uv = uv_probe.stdout.strip() or "uv"
        uv_help = sandbox.exec(f"{shlex.quote(uv)} pip install --help", timeout=10)
        if uv_help.exit_code != 0 or "--exclude-newer" not in uv_help.stdout:
            return DepsResult(
                ok=False,
                report="Date-pinned install requires a uv version supporting --exclude-newer.",
                venv_python=venv_python,
            )
    else:
        # Upgrade only the isolated environment's pip.
        sandbox.exec(f"{venv_pip} install --quiet --upgrade pip", timeout=60)

    # Step 3: Run the install command using the venv's pip
    install_cmd = profile.install_cmd
    if install_cmd is None:
        install_cmd = ""
    elif as_of is None:
        install_cmd = install_cmd.replace("pip install", f"{venv_pip} install")
    elif as_of is not None:
        if "pip install" not in install_cmd:
            return DepsResult(
                ok=False,
                report="Cannot apply the date cutoff: the Python install command has no 'pip install' step.",
                venv_python=venv_python,
            )
        uv_prefix = (
            f"{shlex.quote(uv)} pip install --python {shlex.quote(venv_python)} "
            f"--exclude-newer {shlex.quote(as_of)}"
        )
        install_cmd = install_cmd.replace("pip install", uv_prefix)

    if install_cmd:
        log.info("Running Python install: %s", install_cmd)
        result = sandbox.exec(install_cmd, timeout=_INSTALL_TIMEOUT)
    else:
        result = None

    if result is not None and result.timed_out:
        return DepsResult(
            ok=False,
            report=(
                f"Install timed out after {_INSTALL_TIMEOUT}s. "
                f"Command: {install_cmd}\n"
                f"Partial stdout: {result.stdout[:400]}"
            ),
            venv_python=venv_python,
        )

    if result is not None and result.exit_code != 0:
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
        if as_of is None:
            sandbox.exec(f"{venv_pip} install --quiet {test_framework_pkg}", timeout=60)
        else:
            framework_cmd = (
                f"{shlex.quote(uv)} pip install --python {shlex.quote(venv_python)} "
                f"--exclude-newer {shlex.quote(as_of)} {shlex.quote(test_framework_pkg)}"
            )
            framework_result = sandbox.exec(framework_cmd, timeout=60)
            if framework_result.exit_code != 0 or framework_result.timed_out:
                return DepsResult(
                    ok=False,
                    report=f"Date-pinned test framework install failed: {framework_result.stderr[:400]}",
                    venv_python=venv_python,
                )

    return DepsResult(
        ok=True,
        report=(
            f"Python deps installed into {venv_path}/ using {interp}."
            if as_of is None
            else f"Python deps installed into {venv_path}/ using {interp}, excluding uploads after {as_of}."
        ),
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
