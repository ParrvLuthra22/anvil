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
        if as_of is None and profile.primary_language == "python":
            commit_date = get_commit_date(sandbox, "HEAD")
            if commit_date:
                as_of = commit_date
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


def _pick_python_interpreter(sandbox: Sandbox, as_of: str | None = None) -> str | None:
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
    if as_of:
        # Python 3.13 removed cgi, imp, etc. (released Oct 2024)
        if as_of < "2024-10-01":
            spec_str = f"{spec_str},<3.13" if spec_str else "<3.13"
        # Python 3.12 removed distutils (released Oct 2023)
        if as_of < "2023-10-01":
            spec_str = f"{spec_str},<3.12" if spec_str else "<3.12"
            
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

        # 3. Can it create a venv and does it have required modules?
        temp_result = sandbox.exec(
            f'{interp} -c "import tempfile; print(tempfile.mkdtemp(prefix=\'anvil-probe-\'))"',
            timeout=10,
        )
        if temp_result.exit_code != 0 or temp_result.timed_out:
            log.debug("Skipping %s — cannot create temporary probe directory: %s", interp, temp_result.stderr[:100])
            continue
        probe_dir = temp_result.stdout.strip()
        if not probe_dir:
            log.debug("Skipping %s — temporary probe directory path was empty", interp)
            continue

        try:
            quoted_probe_dir = shlex.quote(probe_dir)
            venv_create = sandbox.exec(f"{interp} -m venv {quoted_probe_dir}", timeout=30)
            if venv_create.exit_code != 0 or venv_create.timed_out:
                log.debug("Skipping %s — cannot create venv: %s", interp, venv_create.stderr[:100])
                continue

            verify = sandbox.exec(
                f"{quoted_probe_dir}/bin/python -c 'import ensurepip, pyexpat, ssl'",
                timeout=10,
            )
        finally:
            try:
                sandbox.exec(f"rm -rf -- {shlex.quote(probe_dir)}", timeout=10)
            except Exception:  # noqa: BLE001 - cleanup must not hide probe failures
                log.warning("Could not remove temporary interpreter probe %s", probe_dir)
        
        if verify.exit_code != 0:
            log.debug("Skipping %s — broken interpreter (missing ensurepip, pyexpat, or ssl): %s", interp, verify.stderr[:100])
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
    interp = _pick_python_interpreter(sandbox, as_of=as_of)
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

    # Step 2: Upgrade pip and try to install uv
    sandbox.exec(f"{venv_pip} install --quiet --upgrade pip uv", timeout=60)

    # Step 3: Run the install command using the venv's pip or uv
    install_cmd = profile.install_cmd
    if install_cmd is None:
        install_cmd = ""
        
    strategies = []
    if install_cmd and "pip install" in install_cmd:
        uv_path = f"{venv_path}/bin/uv"
        uv_probe = sandbox.exec(f"test -x {uv_path}", timeout=5)
        
        if uv_probe.exit_code == 0:
            uv_cmd = f"VIRTUAL_ENV={venv_path} {uv_path} pip install"
            if as_of:
                uv_cmd += f" --exclude-newer {as_of}"
            strategies.append(("uv_pinned", install_cmd.replace("pip install", uv_cmd, 1)))
        
        strategies.append(("pip_plain", install_cmd.replace("pip install", f"{venv_pip} install", 1)))
    elif install_cmd:
        strategies.append(("generic", install_cmd))
        
    result = None
    used_strategy = None
    if strategies:
        for strategy_name, cmd in strategies:
            log.info("Running Python install strategy %s: %s", strategy_name, cmd)
            result = sandbox.exec(cmd, timeout=_INSTALL_TIMEOUT)
            
            if result.exit_code == 0 and not result.timed_out:
                used_strategy = strategy_name
                break
            else:
                log.debug("Install strategy %s failed (exit %s).", strategy_name, result.exit_code)

        if result is None or result.timed_out or result.exit_code != 0:
            status = "Install timed out" if (result and result.timed_out) else "Install failed"
            return DepsResult(
                ok=False,
                report=(
                    f"{status} after trying all strategies.\n"
                    f"Command: {strategies[-1][1] if strategies else 'None'}\n"
                    f"stdout: {result.stdout[:400] if result else ''}\n"
                    f"stderr: {result.stderr[:400] if result else ''}"
                ),
                venv_python=venv_python,
            )

    # Step 4: Guarantee the test framework is in the venv
    test_framework_pkg = _test_framework_package(profile)
    if test_framework_pkg:
        # Always use the best tool available for the framework
        cmd_framework = f"{venv_pip} install --quiet {test_framework_pkg}"
        if as_of:
            uv_path = f"{venv_path}/bin/uv"
            if sandbox.exec(f"test -x {uv_path}", timeout=5).exit_code == 0:
                cmd_framework = f"VIRTUAL_ENV={venv_path} {uv_path} pip install --quiet --exclude-newer {as_of} {test_framework_pkg}"
            
        framework_result = sandbox.exec(cmd_framework, timeout=60)
        if framework_result.exit_code != 0 or framework_result.timed_out:
            return DepsResult(
                ok=False,
                report=f"Test framework install failed: {framework_result.stderr[:400]}",
                venv_python=venv_python,
            )

    return DepsResult(
        ok=True,
        report=(
            f"Python deps installed into {venv_path}/ using {interp} (strategy: {used_strategy})."
            if as_of is None
            else f"Python deps installed into {venv_path}/ using {interp}, excluding uploads after {as_of} (strategy: {used_strategy})."
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
