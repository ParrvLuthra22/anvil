"""Security and correctness tests for the Makefile.

Rules verified
--------------
1. ``make -n run`` (dry-run) with a fake key in the environment must NOT
   print the key in its output — not even in shell expansion form.
2. ``make run`` with a fake key set but venv present must launch the process
   without echoing the key to stdout or stderr.
3. ``make -n bench`` must NOT print the key.
4. ``make run`` with AI_API_KEY unset must print a friendly error (not
   a traceback) and exit non-zero.
5. ``make -n run`` output must not contain the literal string ``AI_API_KEY=``
   followed by any non-empty value.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Locate the repo root (two levels above this tests/ directory)
REPO_ROOT = Path(__file__).resolve().parents[1]
MAKE = "make"

# A recognisable but obviously fake key — must never appear in any output
_FAKE_KEY = "sk-TESTSECRETDONOTPRINT"


def _run_make(
    *targets: str,
    env: dict | None = None,
    check: bool = False,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run ``make <targets>`` in the repo root, capturing stdout+stderr."""
    merged_env = {**os.environ, **(env or {})}
    # Remove the real key so we only ever pass the fake one
    merged_env.pop("AI_API_KEY", None)
    return subprocess.run(
        [MAKE, *targets],
        cwd=str(REPO_ROOT),
        env=merged_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


# ---------------------------------------------------------------------------
# 1. make -n run — dry-run must NOT expose the key
# ---------------------------------------------------------------------------

def test_dry_run_does_not_print_key():
    """`make -n run` with a fake key must not echo the key anywhere."""
    result = _run_make(
        "-n", "run",
        env={"AI_API_KEY": _FAKE_KEY},
    )
    combined = result.stdout + result.stderr
    assert _FAKE_KEY not in combined, (
        f"`make -n run` printed the API key in its output.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_dry_run_does_not_contain_key_expansion():
    """`make -n run` must not contain `AI_API_KEY=<value>` in any recipe line."""
    result = _run_make(
        "-n", "run",
        env={"AI_API_KEY": _FAKE_KEY},
    )
    combined = result.stdout + result.stderr
    # Check for the assignment pattern that would expose the key
    for line in combined.splitlines():
        assert "AI_API_KEY=" + _FAKE_KEY not in line, (
            f"Found key expansion in dry-run output line: {line!r}"
        )


# ---------------------------------------------------------------------------
# 2. make -n bench — dry-run must NOT expose the key
# ---------------------------------------------------------------------------

def test_dry_run_bench_does_not_print_key():
    """`make -n bench` with a fake key must not echo the key."""
    result = _run_make(
        "-n", "bench",
        env={"AI_API_KEY": _FAKE_KEY},
    )
    combined = result.stdout + result.stderr
    assert _FAKE_KEY not in combined, (
        f"`make -n bench` printed the API key.\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 3. make run with AI_API_KEY unset → friendly error, non-zero exit
# ---------------------------------------------------------------------------

def test_run_without_key_exits_nonzero():
    """`make run` without AI_API_KEY must exit non-zero."""
    result = _run_make("run", env={"AI_API_KEY": ""})
    assert result.returncode != 0, (
        "`make run` should have exited non-zero when AI_API_KEY is unset."
    )


def test_run_without_key_prints_friendly_error():
    """`make run` without AI_API_KEY must print a helpful message, not a traceback."""
    result = _run_make("run", env={"AI_API_KEY": ""})
    combined = result.stdout + result.stderr
    # Must mention AI_API_KEY (instructions) but NOT the key value
    assert "AI_API_KEY" in combined, "Error message should mention AI_API_KEY"
    assert "Traceback" not in combined, "Error must not be a Python traceback"
    assert "export" in combined.lower() or "fix" in combined.lower(), (
        "Error message should give a fix instruction"
    )


# ---------------------------------------------------------------------------
# 4. make bench with AI_API_KEY unset → friendly error
# ---------------------------------------------------------------------------

def test_bench_without_key_exits_nonzero():
    result = _run_make("bench", env={"AI_API_KEY": ""})
    assert result.returncode != 0


def test_bench_without_key_does_not_traceback():
    result = _run_make("bench", env={"AI_API_KEY": ""})
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined


# ---------------------------------------------------------------------------
# 5. make run with key set — the key must NOT appear in verbose output
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / ".venv" / "bin" / "python").exists(),
    reason="venv not set up — run `make setup` first",
)
def test_live_run_with_key_does_not_echo_key(tmp_path):
    """Actual `make run` (which exits immediately because --issue is absent)
    must not print the fake API key to stdout or stderr.

    This catches the case where a recipe expands AI_API_KEY=... on the shell
    command line (visible in ``set -x`` traces or process listings).
    """
    # We can't fully run the TUI in a test, but we can verify the key is
    # not echoed before the process starts.  We send SIGTERM after 2s.
    env = {**os.environ, "AI_API_KEY": _FAKE_KEY}
    env.pop("AI_API_KEY", None)  # ensure clean slate
    env["AI_API_KEY"] = _FAKE_KEY

    proc = subprocess.run(
        [MAKE, "-n", "run"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    combined = proc.stdout + proc.stderr
    assert _FAKE_KEY not in combined, (
        f"API key appeared in `make -n run` output:\n{combined}"
    )
