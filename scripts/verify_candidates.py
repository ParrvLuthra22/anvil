#!/usr/bin/env python3
"""
Verify candidate issues: clone, profile, ensure_deps, run tests in WorktreeSandbox.
Print a results table at the end.
"""
from __future__ import annotations
import subprocess, sys, time, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anvil.repo.ingest import parse_issue_url, clone_repo
from anvil.repo.profile import profile_repo, repo_map
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
        url="https://github.com/psf/requests/issues/6434",
        fix_pr="https://github.com/psf/requests/pull/6435",
        language="python",
        note="requests: PreparedRequest.prepare_body ignores data= with no Content-Type",
    ),
    dict(
        url="https://github.com/python-attrs/attrs/issues/1087",
        fix_pr="https://github.com/python-attrs/attrs/pull/1088",
        language="python",
        note="attrs: slots=True + weakref_slot=True fails with __weakref__ conflict",
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

WORKDIR = Path("/tmp/anvil_candidates")
WORKDIR.mkdir(exist_ok=True)

results = []

for i, c in enumerate(CANDIDATES):
    label = f"[{i+1}/{len(CANDIDATES)}]"
    url = c["url"]
    print(f"\n{label} {url}")

    try:
        ref = parse_issue_url(url)
    except ValueError as e:
        results.append({**c, "verified": False, "error": f"parse: {e}"})
        continue

    repo_dir = WORKDIR / f"repo_{ref.owner}_{ref.repo}"
    work_dir = WORKDIR / f"work_{ref.owner}_{ref.repo}"

    # Clone (skip if already present)
    if not repo_dir.exists():
        print(f"  cloning {ref.owner}/{ref.repo}...")
        try:
            t0 = time.time()
            clone_repo(ref, repo_dir)
            print(f"  cloned in {time.time()-t0:.1f}s")
        except RuntimeError as e:
            results.append({**c, "verified": False, "error": f"clone: {e}"})
            continue
    else:
        print(f"  using cached clone at {repo_dir}")

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
        results.append({**c, "verified": False, "error": f"sandbox: {e}",
                        "detected_lang": profile.primary_language,
                        "detected_test_cmd": profile.test_cmd})
        continue

    # ensure_deps
    print(f"  running ensure_deps...")
    t0 = time.time()
    deps = ensure_deps(sb, profile)
    deps_time = time.time() - t0
    print(f"  deps ok={deps.ok} ({deps_time:.0f}s): {deps.report[:80]}")

    # Run tests
    test_cmd = profile.test_cmd or "echo 'no test command'"
    # Limit test scope to run fast
    if "pytest" in (test_cmd or ""):
        test_cmd = "set -o pipefail; pytest -x -q --no-header --tb=line 2>&1 | head -60"
    elif "go test" in (test_cmd or ""):
        test_cmd = "go test ./... -count=1 -timeout 90s 2>&1 | tail -20"
    elif "cargo test" in (test_cmd or ""):
        test_cmd = "cargo test --lib 2>&1 | tail -20"
    elif "npm test" in (test_cmd or ""):
        test_cmd = "npm test 2>&1 | tail -30"

    print(f"  running: {test_cmd}")
    t0 = time.time()
    r = sb.exec(test_cmd, timeout=120)
    test_time = time.time() - t0
    ok = r.exit_code == 0
    print(f"  exit={r.exit_code} ({test_time:.0f}s)")
    print(f"  stdout: {r.stdout[:200]}")
    if r.stderr:
        print(f"  stderr: {r.stderr[:100]}")

    sb.close()

    results.append({
        **c,
        "verified": ok and deps.ok,
        "detected_lang": profile.primary_language,
        "detected_test_cmd": profile.test_cmd,
        "deps_ok": deps.ok,
        "deps_report": deps.report[:80],
        "test_exit": r.exit_code,
        "test_time_s": round(test_time),
        "test_stdout_tail": r.stdout[-200:],
        "error": "" if ok else f"exit {r.exit_code}",
    })

# Print table
print("\n\n" + "="*90)
print("VERIFICATION RESULTS")
print("="*90)
for r in results:
    status = "✓ VERIFIED" if r.get("verified") else "✗ FAILED"
    print(f"\n{status} | {r['language']} | {r['url']}")
    print(f"  note: {r['note']}")
    print(f"  fix_pr: {r['fix_pr']}")
    print(f"  detected: lang={r.get('detected_lang','?')} test={r.get('detected_test_cmd','?')!r}")
    if r.get("error"):
        print(f"  error: {r['error']}")
    if r.get("test_stdout_tail"):
        print(f"  output: {r['test_stdout_tail'][:120]}")

# Save for candidates.yaml
print("\n\nJSON for candidates.yaml:")
print(json.dumps(results, indent=2, default=str))
