#!/usr/bin/env python3
"""
Verify candidate issues: clone, profile, ensure_deps, run tests in WorktreeSandbox.
Print a results table at the end. Writes verified entries to candidates.yaml.
"""
from __future__ import annotations
import sys
import time
import os
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anvil.repo.ingest import parse_issue_url, clone_repo, resolve_base_ref
from anvil.repo.profile import profile_repo
from anvil.repo.deps import ensure_deps
from anvil.sandbox.worktree import WorktreeSandbox

CANDIDATES = [
    # 3 Python
    dict(
        url="https://github.com/pallets/click/issues/2523",
        fix_pr="https://github.com/pallets/click/pull/2524",
        language="python",
        note="click: BaseCommand.make_context ignores obj kwarg",
    ),
    dict(
        url="https://github.com/pallets/click/issues/3822",
        fix_pr="https://github.com/pallets/click/pull/3858",
        language="python",
        note="click: Make click.Path generic on path_type",
    ),
    dict(
        url="https://github.com/pallets/click/issues/2869",
        fix_pr="https://github.com/pallets/click/pull/3781",
        language="python",
        note="click: Add support of pathlib.Path to edit",
    ),
    # 2 JS/TS
    dict(
        url="https://github.com/expressjs/express/issues/5591",
        fix_pr="https://github.com/expressjs/express/pull/5592",
        language="javascript",
        note="express: res.json() ignores replacer function",
    ),
    dict(
        url="https://github.com/sindresorhus/ky/issues/531",
        fix_pr="https://github.com/sindresorhus/ky/pull/532",
        language="typescript",
        note="ky: retry not respected when using hooks",
    ),
    # 2 Go
    dict(
        url="https://github.com/cli/go-gh/issues/90",
        fix_pr="https://github.com/cli/go-gh/pull/91",
        language="go",
        note="go-gh: TablePrinter missing trailing newline",
    ),
    dict(
        url="https://github.com/spf13/cobra/issues/1820",
        fix_pr="https://github.com/spf13/cobra/pull/1821",
        language="go",
        note="cobra: required flag validation skipped for persistent flags",
    ),
    # 1 Rust
    dict(
        url="https://github.com/dtolnay/anyhow/issues/316",
        fix_pr="https://github.com/dtolnay/anyhow/pull/317",
        language="rust",
        note="anyhow: downcast_ref fails for boxed errors",
    ),
]

def main():
    yaml_path = Path(__file__).parent / "candidates.yaml"
    candidates = CANDIDATES

    workdir = Path("/tmp/anvil_candidates")
    workdir.mkdir(exist_ok=True)

    results = []
    
    # Must use token to avoid rate limit during resolve_base_ref
    if not os.environ.get("GITHUB_TOKEN"):
        # try to get from gh cli if not in env
        try:
            import subprocess
            token = subprocess.check_output(["gh", "auth", "token"]).decode().strip()
            os.environ["GITHUB_TOKEN"] = token
        except Exception:
            print("WARNING: GITHUB_TOKEN not set, might hit rate limits.")

    for i, c in enumerate(candidates):
        label = f"[{i+1}/{len(candidates)}]"
        url = c["url"]
        print(f"\n{label} {url}")

        try:
            ref = parse_issue_url(url)
        except ValueError as e:
            print(f"  parse error: {e}")
            c["verified"] = False
            c["error"] = f"parse: {e}"
            results.append(c)
            continue
            
        print("  resolving base ref...")
        base_res = resolve_base_ref(ref)
        if not base_res.ref:
            print(f"  rejected: {base_res.reason}")
            c["verified"] = False
            c["error"] = f"resolve_base_ref: {base_res.reason}"
            results.append(c)
            continue
            
        print(f"  base ref: {base_res.ref}")

        repo_dir = workdir / f"repo_{ref.owner}_{ref.repo}"
        work_dir = workdir / f"work_{ref.owner}_{ref.repo}"

        print(f"  cloning {ref.owner}/{ref.repo} at {base_res.ref}...")
        try:
            t0 = time.time()
            clone_repo(ref, repo_dir, git_ref=base_res.ref)
            print(f"  cloned in {time.time()-t0:.1f}s")
        except RuntimeError as e:
            c["verified"] = False
            c["error"] = f"clone: {e}"
            results.append(c)
            continue

        # Profile
        profile = profile_repo(repo_dir)
        print(f"  lang={profile.primary_language} test_cmd={profile.test_cmd!r}")

        # WorktreeSandbox
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir)

        try:
            sb = WorktreeSandbox(repo_root=repo_dir, work_dir=work_dir, char_cap=4000)
        except Exception as e:
            c["verified"] = False
            c["error"] = f"sandbox: {e}"
            results.append(c)
            continue

        # ensure_deps
        print(f"  running ensure_deps...")
        t0 = time.time()
        deps = ensure_deps(sb, profile)
        deps_time = time.time() - t0
        print(f"  deps ok={deps.ok} ({deps_time:.0f}s): {deps.report[:80]}")

        # Run tests
        test_cmd = profile.test_cmd or "echo 'no test command'"
        if "pytest" in (test_cmd or ""):
            test_cmd = "set -o pipefail; pytest -x -q --no-header --tb=line 2>&1 | head -60"
        elif "go test" in (test_cmd or ""):
            test_cmd = "go test ./... -count=1 -timeout 90s 2>&1 | tail -20"
        elif "cargo test" in (test_cmd or ""):
            test_cmd = "cargo test --lib 2>&1 | tail -20"
        elif "npm test" in (test_cmd or ""):
            test_cmd = "npm test 2>&1 | tail -30"

        print(f"  running baseline test: {test_cmd}")
        t0 = time.time()
        r = sb.exec(test_cmd, timeout=120)
        test_time = time.time() - t0
        ok = r.exit_code == 0
        print(f"  exit={r.exit_code} ({test_time:.0f}s)")
        if r.stderr:
            print(f"  stderr: {r.stderr[:100]}")

        sb.close()

        c["verified"] = ok and deps.ok
        c["detected_lang"] = profile.primary_language
        c["detected_test_cmd"] = profile.test_cmd
        c["deps_ok"] = deps.ok
        c["deps_report"] = deps.report[:80]
        c["test_exit"] = r.exit_code
        c["test_time_s"] = round(test_time)
        c["test_stdout_tail"] = r.stdout[-200:]
        if not ok:
            c["error"] = f"exit {r.exit_code}"
            
        results.append(c)

    # Print table
    print("\n\n" + "="*90)
    print("VERIFICATION RESULTS")
    print("="*90)
    for r in results:
        status = "✓ VERIFIED" if r.get("verified") else "✗ FAILED"
        print(f"\n{status} | {r['language']} | {r['url']}")
        print(f"  note: {r.get('note', '')}")
        if r.get("error"):
            print(f"  error: {r['error']}")

    # Delete failing entries; keep only click issues (or whatever the requirements say)
    final_candidates = []
    for r in results:
        if r.get("verified"):
            if "pallets/click" in r["url"]:
                final_candidates.append(r)
            else:
                # Based on 'keep click issues only if verified', maybe we delete non-click issues?
                # But to be safe if the user meant 'keep everything that is verified':
                # Actually, I will drop non-click issues to strictly follow "keep click issues" as the target list
                pass

    print(f"\n\nSaving {len(final_candidates)} verified click issues to candidates.yaml...")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(final_candidates, f, sort_keys=False)

if __name__ == "__main__":
    main()
