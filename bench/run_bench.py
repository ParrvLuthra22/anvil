#!/usr/bin/env python3
"""ANVIL benchmark runner.

Reads ``bench/issues.yaml``, runs each issue headlessly via the same pipeline
as ``python -m anvil --headless``, and prints a summary table.

Usage::

    AI_API_KEY=... python bench/run_bench.py
    AI_API_KEY=... python bench/run_bench.py --issues bench/my_issues.yaml
    AI_API_KEY=... python bench/run_bench.py --timeout 600

Results are written to ``bench/results.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure the project src is importable when run directly
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml  # noqa: E402 (must come after sys.path tweak)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BenchIssue:
    url: str
    description: str = ""
    test_cmd: str | None = None


@dataclass
class BenchResult:
    issue: BenchIssue
    resolved: bool = False
    steps: int = 0
    tokens: int = 0
    seconds: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _load_issues(path: Path) -> list[BenchIssue]:
    """Parse the YAML issue list."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    issues: list[BenchIssue] = []
    for entry in raw.get("issues", []):
        issues.append(
            BenchIssue(
                url=entry["url"],
                description=entry.get("description", entry["url"]),
                test_cmd=entry.get("test_cmd"),
            )
        )
    return issues


def _run_one(issue: BenchIssue, timeout: int) -> BenchResult:
    """Run one issue headlessly in a subprocess; parse the done event from trace."""
    result = BenchResult(issue=issue)
    run_dir = Path("output") / f"bench_{int(time.time())}_{issue.url.split('/')[-1]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "trace.jsonl"

    env = {**os.environ, "ANVIL_OUTPUT_DIR": str(run_dir)}
    cmd = [
        sys.executable, "-m", "anvil",
        "--issue", issue.url,
        "--headless",
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )
        result.seconds = round(time.time() - start, 1)
        if proc.returncode != 0:
            result.error = (proc.stderr or proc.stdout or "non-zero exit")[:200]
            return result
    except subprocess.TimeoutExpired:
        result.seconds = timeout
        result.error = f"timeout after {timeout}s"
        return result
    except Exception as exc:
        result.seconds = round(time.time() - start, 1)
        result.error = str(exc)[:200]
        return result

    # Parse trace for done event
    default_trace = Path("output") / "trace.jsonl"
    for tp in [trace_path, default_trace]:
        if tp.exists():
            try:
                from anvil.trace.recorder import TraceRecorder
                for ev in TraceRecorder.load(tp):
                    if ev.type == "llm_usage":
                        result.tokens += ev.data.get("total_tokens", 0)
                    elif ev.type == "done":
                        result.steps = ev.data.get("steps", 0)
                        result.tokens = ev.data.get("tokens", result.tokens)
                        result.seconds = ev.data.get("seconds", result.seconds)
                        conf = ev.data.get("resolved_confidence", 0)
                        result.resolved = conf >= 0.6
                break
            except Exception:
                pass

    # Optionally run the caller-supplied test command to verify
    if issue.test_cmd and not result.error:
        try:
            check = subprocess.run(
                issue.test_cmd,
                shell=True,
                timeout=120,
                capture_output=True,
                cwd=str(_ROOT),
            )
            result.resolved = check.returncode == 0
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _render_table(results: list[BenchResult]) -> str:
    rows = []
    rows.append("| # | Issue | Description | Resolved | Steps | Tokens | Time (s) |")
    rows.append("|---|-------|-------------|----------|-------|--------|----------|")
    for i, r in enumerate(results, 1):
        resolved = "✓ Yes" if r.resolved else ("✗ No" if not r.error else f"⚠ {r.error[:40]}")
        short_url = r.issue.url.split("github.com/")[-1]
        rows.append(
            f"| {i} | [{short_url}]({r.issue.url}) "
            f"| {r.issue.description[:50]} "
            f"| {resolved} "
            f"| {r.steps} "
            f"| {r.tokens:,} "
            f"| {r.seconds} |"
        )
    return "\n".join(rows)


def _print_table(results: list[BenchResult]) -> None:
    cols = ["#", "Description", "Resolved", "Steps", "Tokens", "Time(s)"]
    widths = [3, 50, 10, 6, 9, 8]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    print("\n" + header)
    print(sep)
    for i, r in enumerate(results, 1):
        resolved = "YES" if r.resolved else ("NO" if not r.error else f"ERR")
        print(
            "  ".join(
                str(v).ljust(w)
                for v, w in zip(
                    [i, r.issue.description[:50], resolved, r.steps, f"{r.tokens:,}", r.seconds],
                    widths,
                )
            )
        )
    passed = sum(1 for r in results if r.resolved)
    print(f"\nResolved: {passed}/{len(results)}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ANVIL benchmark runner")
    parser.add_argument(
        "--issues",
        type=Path,
        default=Path("bench/issues.yaml"),
        help="Path to the YAML issue list (default: bench/issues.yaml)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        metavar="SECONDS",
        help="Per-issue wall-clock timeout in seconds (default: 900)",
    )
    args = parser.parse_args()

    if not os.environ.get("AI_API_KEY"):
        print("ERROR: AI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    issues = _load_issues(args.issues)
    if not issues:
        print("No issues found in", args.issues, file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(issues)} benchmark issue(s) with timeout={args.timeout}s each…\n")
    results: list[BenchResult] = []
    for issue in issues:
        print(f"  → {issue.description or issue.url}")
        r = _run_one(issue, args.timeout)
        results.append(r)
        status = "✓" if r.resolved else "✗"
        print(f"    {status}  steps={r.steps}  tokens={r.tokens:,}  time={r.seconds}s")

    _print_table(results)

    # Write markdown results
    out = Path("bench/results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    md_content = (
        f"# ANVIL Benchmark Results\n\n"
        f"Generated: {timestamp}\n\n"
        f"{_render_table(results)}\n"
    )
    out.write_text(md_content, encoding="utf-8")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
