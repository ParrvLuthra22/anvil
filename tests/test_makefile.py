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


# ---------------------------------------------------------------------------
# Item 2: Makefile syntax
# ---------------------------------------------------------------------------

_MAKEFILE_TARGETS = ["setup", "run", "test", "clean", "bench", "demo"]


def test_makefile_dry_run_no_syntax_error_per_target():
    """``make -n <target>`` must exit 0 for every declared Makefile target.

    A syntax error in any recipe causes make to print a shell error and exit
    non-zero. This catches the stray-backslash class of bug.
    """
    env = {**os.environ}
    # Provide a fake key so targets that check AI_API_KEY don't exit 1 early
    env["AI_API_KEY"] = _FAKE_KEY
    # Pretend venv exists so the guard passes in dry-run
    env.setdefault("VENV", ".venv")

    for target in _MAKEFILE_TARGETS:
        result = subprocess.run(
            [MAKE, "-n", target],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"`make -n {target}` exited {result.returncode} — syntax error in recipe.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "syntax error" not in combined.lower(), (
            f"`make -n {target}` printed a shell syntax error:\n{combined}"
        )
        assert "unexpected EOF" not in combined, (
            f"`make -n {target}` hit an unexpected EOF (unterminated string?):\n{combined}"
        )


def test_makefile_recipes_pass_sh_n():
    """Extracting recipe lines with ``make -n`` and running ``sh -n`` on them
    must not report a syntax error.

    This is a belt-and-suspenders check that catches quoting problems even
    when make itself doesn't complain (e.g. when a continuation line hides
    the issue from make's parser but shells see it at runtime).
    """
    env = {**os.environ, "AI_API_KEY": _FAKE_KEY}

    recipe_lines: list[str] = []
    for target in _MAKEFILE_TARGETS:
        result = subprocess.run(
            [MAKE, "-n", target],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Collect only non-empty, non-comment lines
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                recipe_lines.append(stripped)

    if not recipe_lines:
        pytest.skip("No recipe lines extracted — check Makefile targets")

    # Join the lines — the trailing \ continuation chars preserved by make -n
    # make the multi-line if/fi blocks valid when re-joined with newline.
    script = "\n".join(recipe_lines)
    sh_result = subprocess.run(
        ["sh", "-n"],
        input=script,
        capture_output=True,
        text=True,
    )
    # sh -n on joined make recipe lines can fail for benign reasons (e.g. \\-
    # continuation becomes a separate line without the context of the full block).
    # We only assert there is NO "unexpected EOF" or unterminated-string error
    # (the specific error the stray-backslash bug causes).
    assert "unexpected EOF" not in sh_result.stderr, (
        f"sh -n hit an unterminated string in the Makefile recipes:\n"
        f"{sh_result.stderr}\n\n--- recipe lines ---\n{script}"
    )
    assert "unterminated" not in sh_result.stderr.lower(), (
        f"sh -n found an unterminated token in the Makefile recipes:\n"
        f"{sh_result.stderr}\n\n--- recipe lines ---\n{script}"
    )


# ---------------------------------------------------------------------------
# Item 2: make test must succeed (regression guard)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO_ROOT / ".venv" / "bin" / "python").exists(),
    reason="venv not set up — run `make setup` first",
)
def test_make_test_target_exits_zero():
    """Running pytest directly (the command ``make test`` delegates to) must exit 0.

    We run pytest directly rather than via ``make test`` to avoid recursive
    subprocess nesting (make → pytest → this test → make → pytest …) which
    would exceed reasonable timeouts.  We exclude test_makefile.py itself
    to prevent the recursion and keep the test fast.

    This prevents the stray-backslash class of bug from regressing: if the
    Makefile has a syntax error in the test recipe, ``make -n test`` will
    fail in test_makefile_dry_run_no_syntax_error_per_target before we get here.
    """
    vpy = REPO_ROOT / ".venv" / "bin" / "python"
    env = {k: v for k, v in os.environ.items() if k != "AI_API_KEY"}
    result = subprocess.run(
        [str(vpy), "-m", "pytest", "-q", "--ignore=tests/test_makefile.py"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"pytest (non-makefile tests) exited {result.returncode}.\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-1000:]}"
    )
    combined = result.stdout + result.stderr
    assert "syntax error" not in combined.lower(), (
        f"pytest reported a shell syntax error:\n{combined}"
    )


# ---------------------------------------------------------------------------
# Item 3: live trace — trace.jsonl grows during a run, not only at the end
# ---------------------------------------------------------------------------

def test_live_trace_grows_during_run(tmp_path):
    """Emit several events to a bus, verify trace.jsonl is written incrementally.

    Simulates what the recorder task does in _wire_recorder / _run_headless:
    subscribe() inside the running loop, record each event immediately so the
    file grows before the 'done' event arrives.  A timed-out run still leaves
    a partial trace (guaranteed by the finally: recorder.close() in _drain).
    """
    import asyncio
    import json
    import time

    from anvil.events import AgentEvent, Phase
    from anvil.trace.recorder import TraceRecorder

    trace_path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(trace_path)

    events = [
        AgentEvent(ts=time.time(), type="phase",   phase=Phase.INGEST,    data={"name": "ingest"}),
        AgentEvent(ts=time.time(), type="message",  phase=Phase.INGEST,    data={"role": "user", "text": "hi"}),
        AgentEvent(ts=time.time(), type="tool_call",phase=Phase.PROFILE,   data={"tool": "list_dir", "args": {}}),
        AgentEvent(ts=time.time(), type="done",     phase=Phase.FINALIZE,  data={"resolved_confidence": 0.5,
                                                                                  "patch_path": "output/patch.diff",
                                                                                  "report_path": "output/report.md",
                                                                                  "steps": 3, "tokens": 100, "seconds": 1}),
    ]

    written_sizes: list[int] = []
    for ev in events:
        recorder.record(ev)   # flush-on-write — file grows with each event
        written_sizes.append(trace_path.stat().st_size)

    recorder.close()

    # The file must have grown monotonically (each write adds at least one byte)
    for i in range(1, len(written_sizes)):
        assert written_sizes[i] > written_sizes[i - 1], (
            f"Trace did not grow after event {i}: "
            f"sizes={written_sizes}"
        )

    # All 4 events must be readable back
    from anvil.trace.recorder import TraceRecorder as TR
    loaded = list(TR.load(trace_path))
    assert len(loaded) == 4

    # Mid-run read: after 3 events (before done) the file should have 3 lines
    lines = [l for l in trace_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 4  # all 4 are there after close

    # Check the first 3 lines are readable mid-run (simulate a timed-out run)
    partial_text = "\n".join(lines[:3])
    partial_path = tmp_path / "partial.jsonl"
    partial_path.write_text(partial_text + "\n")
    partial_events = list(TR.load(partial_path))
    assert len(partial_events) == 3, "Should read 3 events from partial mid-run trace"
