#!/usr/bin/env python3
"""ANVIL benchmark scorer."""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from anvil.repo.profile import profile_repo


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int = 120,
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{cwd}:{base_env.get('PYTHONPATH', '')}".rstrip(":")
    if env:
        base_env.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=base_env,
    )


def _score_instance(instance: dict, run_result: dict, work_dir: Path) -> dict:
    patch_path_str = run_result.get("patch_path", "")
    patch_path = Path(patch_path_str) if patch_path_str else None

    # 1. Error checks
    if run_result.get("error"):
        if "timeout" in run_result["error"].lower():
            run_result["category"] = "timeout"
        else:
            run_result["category"] = "harness_error"
        run_result["resolved"] = False
        return run_result

    if not patch_path or not patch_path.exists():
        run_result["category"] = "no_patch"
        run_result["resolved"] = False
        return run_result

    patch_text = patch_path.read_text(encoding="utf-8").strip()
    if not patch_text:
        run_result["category"] = "empty_patch"
        run_result["resolved"] = False
        return run_result

    # 2. Setup fresh worktree / clone
    repo_url = instance["repo"]
    base_commit = instance["base_commit"]
    repo_dir = work_dir / instance["instance_id"]

    if repo_url.startswith("file://"):
        local_path = repo_url[7:]
    elif (_ROOT / repo_url).exists():
        local_path = str((_ROOT / repo_url).resolve())
    elif Path(repo_url).exists():
        local_path = str(Path(repo_url).resolve())
    else:
        local_path = repo_url

    try:
        clone_res = _run_cmd(["git", "clone", local_path, str(repo_dir)], cwd=work_dir)
        if clone_res.returncode != 0:
            run_result["category"] = "harness_error"
            run_result["resolved"] = False
            run_result["error"] = f"Clone failed: {clone_res.stderr or clone_res.stdout}"
            return run_result

        co_res = _run_cmd(["git", "checkout", base_commit], cwd=repo_dir)
        if co_res.returncode != 0:
            run_result["category"] = "harness_error"
            run_result["resolved"] = False
            run_result["error"] = f"Checkout failed: {co_res.stderr or co_res.stdout}"
            return run_result
    except Exception as e:
        run_result["category"] = "harness_error"
        run_result["resolved"] = False
        run_result["error"] = f"Workspace setup failed: {e}"
        return run_result

    # 3. Apply agent's patch
    check_agent = _run_cmd(
        ["git", "apply", "--check", "--allow-empty", str(patch_path.resolve())], cwd=repo_dir
    )
    if check_agent.returncode != 0:
        run_result["category"] = "patch_does_not_apply"
        run_result["resolved"] = False
        return run_result

    apply_agent = _run_cmd(
        ["git", "apply", "--allow-empty", str(patch_path.resolve())], cwd=repo_dir
    )
    if apply_agent.returncode != 0:
        run_result["category"] = "patch_does_not_apply"
        run_result["resolved"] = False
        return run_result

    # 4. Check for syntax errors introduced by the patch
    for py_file in repo_dir.rglob("*.py"):
        if ".git" in py_file.parts or ".venv" in py_file.parts:
            continue
        try:
            py_compile.compile(str(py_file), doraise=True)
        except py_compile.PyCompileError:
            run_result["category"] = "patch_does_not_apply"
            run_result["resolved"] = False
            return run_result

    # 5. Apply test_patch
    test_patch_text = instance.get("test_patch", "")
    if test_patch_text:
        tp_path = repo_dir / "test_patch.diff"
        tp_path.write_text(test_patch_text, encoding="utf-8")
        apply_test = _run_cmd(["git", "apply", "--allow-empty", "test_patch.diff"], cwd=repo_dir)
        if apply_test.returncode != 0:
            print(f"Warning: test_patch failed to apply for {instance['instance_id']}")

    # 6. Profile repo for install/test commands if needed
    profile = profile_repo(repo_dir)
    if profile.install_cmd:
        _run_cmd(profile.install_cmd.split(), cwd=repo_dir, timeout=300)

    # 7. Run PASS_TO_PASS and FAIL_TO_PASS with the recorded test_cmd
    f2p = instance.get("FAIL_TO_PASS", [])
    p2p = instance.get("PASS_TO_PASS", [])

    base_test_cmd = profile.test_cmd.split() if profile.test_cmd else [sys.executable, "-m", "pytest", "-q"]

    # Run PASS_TO_PASS first
    if p2p:
        cmd = base_test_cmd + p2p
        res = _run_cmd(cmd, cwd=repo_dir)
        if res.returncode != 0:
            run_result["category"] = "p2p_regression"
            run_result["resolved"] = False
            return run_result

    # Run FAIL_TO_PASS
    if f2p:
        cmd = base_test_cmd + f2p
        res = _run_cmd(cmd, cwd=repo_dir)
        if res.returncode != 0:
            run_result["category"] = "f2p_fail"
            run_result["resolved"] = False
            return run_result

    # All tests passed!
    run_result["category"] = "resolved"
    run_result["resolved"] = True
    return run_result


def _print_summary_table(results: list[dict]) -> None:
    cols = ["#", "Instance ID", "Category", "Resolved", "Time(s)", "Steps", "Tokens"]
    widths = [3, 24, 16, 10, 8, 7, 10]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)

    print("\n" + "=" * len(header))
    print("ANVIL Benchmark Score Summary".center(len(header)))
    print("=" * len(header))
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        iid = r.get("instance_id", "?")[:24]
        cat = r.get("category", "unknown")[:16]
        res = "YES" if r.get("resolved") else "NO"
        dur = f"{r.get('duration', 0.0):.1f}"
        steps = str(r.get("steps", 0))
        toks = f"{r.get('tokens', 0):,}"
        print(
            "  ".join(
                str(v).ljust(w)
                for v, w in zip([i, iid, cat, res, dur, steps, toks], widths)
            )
        )

    print(sep)
    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved"))
    rate = (resolved / total) if total else 0.0
    print(f"Total: {total} | Resolved: {resolved} ({rate:.1%})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANVIL benchmark scorer")
    parser.add_argument("--instances", type=Path, default=Path("bench/instances.json"))
    parser.add_argument("--results", type=Path, default=Path("bench/results.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("bench/score.json"))
    args = parser.parse_args()

    instances = {
        i["instance_id"]: i
        for i in json.loads(args.instances.read_text(encoding="utf-8"))
    }

    results = []
    if args.results.exists():
        for line in args.results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass

    scored = []
    with tempfile.TemporaryDirectory() as td:
        work_dir = Path(td)
        for r in results:
            inst = instances.get(r["instance_id"])
            if not inst:
                continue

            print(f"Scoring {r['instance_id']}...")
            r_scored = _score_instance(inst, r, work_dir)
            scored.append(r_scored)
            print(f"  -> {r_scored['category']} (resolved: {r_scored['resolved']})")

    # Output bench/score.json
    total = len(scored)
    resolved_count = sum(1 for r in scored if r.get("resolved"))
    cat_counts: dict[str, int] = {}
    for r in scored:
        c = r.get("category", "unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1

    score_data = {
        "instances": scored,
        "summary": {
            "total": total,
            "resolved": resolved_count,
            "resolved_rate": (resolved_count / total) if total else 0.0,
            "categories": cat_counts,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(score_data, indent=2), encoding="utf-8")

    # Rewrite results.jsonl with scored fields
    lines = [json.dumps(r) + "\n" for r in scored]
    args.results.write_text("".join(lines), encoding="utf-8")

    # Print summary table to stdout
    _print_summary_table(scored)


if __name__ == "__main__":
    main()
