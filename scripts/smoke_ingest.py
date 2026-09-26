#!/usr/bin/env python3
"""smoke_ingest.py — end-to-end smoke test for the Akshat modules.

Usage:
    python scripts/smoke_ingest.py <github-issue-url>

Steps:
1. Parse the issue URL.
2. Fetch the issue metadata from GitHub (unauthenticated).
3. Clone the repository (shallow).
4. Profile the repo (language, install/test commands).
5. Print the repo map.
6. Create a WorktreeSandbox (or DockerSandbox if Docker is available).
7. Run the detected test command and print the result.
8. Clean up.

Example:
    python scripts/smoke_ingest.py https://github.com/psf/requests/issues/6456
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Ensure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anvil.repo.ingest import clone_repo, fetch_issue, parse_issue_url
from anvil.repo.profile import profile_repo, repo_map
from anvil.sandbox import make_sandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _err(msg: str) -> None:
    print(f"  ✗  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1

    url = sys.argv[1]

    # --- Step 1: Parse URL ---
    _section("Step 1: Parsing issue URL")
    try:
        ref = parse_issue_url(url)
        _ok(f"owner={ref.owner}  repo={ref.repo}  issue=#{ref.number}")
    except ValueError as exc:
        _err(f"Invalid URL: {exc}")
        return 1

    # --- Step 2: Fetch issue ---
    _section("Step 2: Fetching issue from GitHub (unauthenticated)")
    t0 = time.monotonic()
    ref = fetch_issue(ref)
    elapsed = time.monotonic() - t0
    if ref.title:
        _ok(f"Title: {ref.title!r}  ({elapsed:.1f}s)")
        _ok(f"Body preview: {ref.body[:120]!r}...")
        _ok(f"Comments fetched: {len(ref.comments)}")
    else:
        _warn(f"Could not fetch issue metadata: {ref.body[:200]}")

    # --- Step 3: Clone ---
    _section("Step 3: Cloning repository (shallow)")
    with tempfile.TemporaryDirectory(prefix="anvil_smoke_") as tmpdir:
        repo_dest = Path(tmpdir) / ref.repo
        try:
            t0 = time.monotonic()
            repo_root = clone_repo(ref, repo_dest)
            elapsed = time.monotonic() - t0
            _ok(f"Cloned to {repo_root}  ({elapsed:.1f}s)")
        except RuntimeError as exc:
            _err(f"Clone failed: {exc}")
            return 1

        # --- Step 4: Profile ---
        _section("Step 4: Profiling repository")
        profile = profile_repo(repo_root)
        _ok(f"Languages:    {profile.languages}")
        _ok(f"Primary:      {profile.primary_language}")
        _ok(f"Install cmd:  {profile.install_cmd!r}")
        _ok(f"Test cmd:     {profile.test_cmd!r}")
        _ok(f"Framework:    {profile.test_framework!r}")
        if profile.notes:
            _warn(f"Notes: {profile.notes}")

        # --- Step 5: Repo map ---
        _section("Step 5: Repository map")
        rmap = repo_map(repo_root, max_chars=4000)
        print(rmap)

        # --- Step 6: Create sandbox ---
        _section("Step 6: Creating sandbox")
        work_dir = Path(tmpdir) / f"{ref.repo}_work"
        config = {"sandbox": "auto", "tool_output_char_cap": 4000}
        sandbox = make_sandbox(config, repo_root, profile=profile)
        sandbox_type = type(sandbox).__name__
        _ok(f"Sandbox type: {sandbox_type}  root={sandbox.root}")

        # --- Step 7: Run install + tests ---
        if profile.install_cmd:
            _section(f"Step 7a: Running install: {profile.install_cmd!r}")
            t0 = time.monotonic()
            result = sandbox.exec(profile.install_cmd, timeout=180)
            elapsed = time.monotonic() - t0
            if result.exit_code == 0:
                _ok(f"Install succeeded  ({elapsed:.1f}s)")
            else:
                _warn(f"Install exited {result.exit_code}  ({elapsed:.1f}s)")
                if result.stderr:
                    print(result.stderr[:500])

        if profile.test_cmd:
            _section(f"Step 7b: Running tests: {profile.test_cmd!r}")
            t0 = time.monotonic()
            result = sandbox.exec(profile.test_cmd, timeout=300)
            elapsed = time.monotonic() - t0
            status = "PASSED" if result.exit_code == 0 else "FAILED"
            _ok(f"Tests {status}  exit={result.exit_code}  ({elapsed:.1f}s)")
            if result.stdout:
                print("\n--- stdout (last 1000 chars) ---")
                print(result.stdout[-1000:])
            if result.stderr:
                print("\n--- stderr (last 500 chars) ---")
                print(result.stderr[-500:])
        else:
            _warn("No test command detected — skipping test run.")

        # --- Step 8: Cleanup ---
        _section("Step 8: Cleanup")
        sandbox.close()
        _ok("Sandbox closed. tmpdir will be removed on exit.")

    _section("Done ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
