#!/usr/bin/env python3
"""Build bench/instances.json from SWE-bench Lite.

Downloads SWE-bench Lite from HuggingFace, filters to preferred pure-Python
repos, runs a GOLD CHECK on each candidate (clone → install → test_patch →
FAIL_TO_PASS must fail → gold patch → FAIL_TO_PASS must pass +
PASS_TO_PASS sample must pass), writes passing instances to
bench/instances.json (no gold patch inside), gold patches go to
bench/gold/<id>.patch.

Usage::

    python scripts/swebench_subset.py [--max-instances N] [--dry-run]

Requirements: pip install datasets   (already installed in .venv)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from math import ceil
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREFERRED_REPOS: list[str] = [
    "psf/requests",
    "pallets/flask",
    "pytest-dev/pytest",
    "pylint-dev/pylint",
    "sphinx-doc/sphinx",
    "mwaskom/seaborn",
    "pydata/xarray",
    "django/django",
    "sympy/sympy",
]
SKIP_REPOS: set[str] = {
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
}
TARGET_MIN = 8
TARGET_MAX = 12
TEST_TIMEOUT = 120
MAX_PTP_SAMPLE = 5
PYTHON_CANDIDATES = ["python3.12", "python3.11", "python3.10", "python3.9", "python3"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_swebench_lite() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        sys.exit("ERROR: pip install datasets")
    print("Downloading SWE-bench Lite…", flush=True)
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except Exception as exc:
        sys.exit(f"ERROR: download failed: {exc}\n>>> Report to Parrv immediately <<<")
    rows = [dict(r) for r in ds]
    print(f"  {len(rows)} instances loaded.", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_test_list(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return list(raw)
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [x.strip() for x in raw.splitlines() if x.strip()]


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


from anvil.repo.ingest import clone_repo, IssueRef
from anvil.sandbox.worktree import WorktreeSandbox
from anvil.repo.profile import profile_repo
from anvil.repo.deps import ensure_deps

def _apply_patch(patch_text: str, sandbox: WorktreeSandbox) -> tuple[bool, str]:
    """Apply a unified diff using patch -p1."""
    # Write patch to a temporary file in the sandbox
    sandbox.write_file(".anvil_temp.patch", patch_text)
    res = sandbox.exec("patch -p1 --forward --batch --no-backup-if-mismatch < .anvil_temp.patch", timeout=30)
    sandbox.exec("rm -f .anvil_temp.patch")
    if res.exit_code == 0:
        return True, ""
    
    # Try git apply as fallback
    sandbox.write_file(".anvil_temp.patch", patch_text)
    res2 = sandbox.exec("git apply --reject .anvil_temp.patch", timeout=30)
    sandbox.exec("rm -f .anvil_temp.patch")
    return res2.exit_code == 0, (res2.stderr or res.stderr)[:400]


def _build_test_cmd(venv_python: str, repo: str, test_ids: list[str]) -> str:
    if repo.startswith("django/"):
        modules: list[str] = []
        for tid in test_ids:
            path = tid.split("::")[0]
            path = path.removeprefix("tests/").removesuffix(".py")
            mod = path.replace("/", ".")
            if mod not in modules:
                modules.append(mod)
        return f"{venv_python} tests/runtests.py --verbosity=2 {' '.join(modules)}"
    ids = " ".join(test_ids)
    return f"{venv_python} -m pytest -x -q --tb=short {ids}"


def _difficulty_hint(patch: str) -> str:
    adds = sum(1 for ln in patch.splitlines()
               if ln.startswith("+") and not ln.startswith("+++"))
    if adds <= 5:   return "trivial"
    if adds <= 20:  return "easy"
    if adds <= 60:  return "medium"
    return "hard"





# ---------------------------------------------------------------------------
# Gold check — operates directly on repo_dir (no WorktreeSandbox)
# ---------------------------------------------------------------------------
def gold_check(inst: dict[str, Any], workdir: Path, verbose: bool) -> dict[str, Any]:
    """Run full 6-step gold check using WorktreeSandbox. Never raises."""
    iid    = inst["instance_id"]
    repo   = inst["repo"]
    base   = inst["base_commit"]
    tpatch = inst["test_patch"]
    gpatch = inst["patch"]
    ftp    = _parse_test_list(inst["FAIL_TO_PASS"])
    ptp    = _parse_test_list(inst["PASS_TO_PASS"])[:MAX_PTP_SAMPLE]

    def _fail(msg: str, install_s: float = 0.0) -> dict[str, Any]:
        return {"ok": False, "error": msg, "notes": "",
                "install_seconds": install_s, "venv_python": None, "test_cmd": ""}

    owner, repo_name = repo.split("/", 1)
    # Use a persistent cache for the clone
    repo_cache_dir = workdir / repo.replace("/", "_")
    work_dir = workdir / iid

    t0 = time.time()

    # [1] Clone
    if verbose:
        print(f"    [1] clone {repo}@{base[:8]}", flush=True)
    
    if not repo_cache_dir.exists():
        repo_cache_dir.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://github.com/{owner}/{repo_name}.git"
        res = subprocess.run(["git", "clone", clone_url, str(repo_cache_dir)], capture_output=True, text=True)
        if res.returncode != 0:
            return _fail(f"clone: {res.stderr[:200]}")

    sb = WorktreeSandbox(repo_root=repo_cache_dir, work_dir=work_dir)
    try:
        # make sure we are at base_commit
        checkout_res = sb.exec(f"git checkout {base}", timeout=30)
        if checkout_res.exit_code != 0:
            return _fail(f"checkout: {checkout_res.stderr[:200]}")

        # [2] Install deps
        if verbose:
            print("    [2] install deps", flush=True)
        t_deps = time.time()
        profile = profile_repo(work_dir)
        deps_res = ensure_deps(sb, profile)
        install_s = round(time.time() - t_deps, 1)
        if not deps_res.ok:
            return _fail(f"deps: {deps_res.report}", install_s)
        
        venv_python = deps_res.venv_python or "python3" # Fallback if not python? 
        
        ftp_cmd = _build_test_cmd(venv_python, repo, ftp)
        ptp_cmd = _build_test_cmd(venv_python, repo, ptp) if ptp else None

        # [3] Apply test_patch
        if verbose:
            print("    [3] apply test_patch", flush=True)
        ok_tp, tp_err = _apply_patch(tpatch, sb)
        if not ok_tp:
            return {"ok": False, "error": f"test_patch: {tp_err}", "notes": "",
                    "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}

        # [4] FAIL_TO_PASS must FAIL before gold
        if verbose:
            print(f"    [4] FAIL_TO_PASS (must fail pre-gold): {ftp_cmd[:70]}", flush=True)
        pre_res = sb.exec(ftp_cmd, timeout=TEST_TIMEOUT)
        pre_pass = pre_res.exit_code == 0
        pre_out = (pre_res.stdout + "\\n" + pre_res.stderr).strip()
        if pre_pass:
            return {"ok": False, "error": "FAIL_TO_PASS passed before gold (test doesn't cover bug)",
                    "notes": pre_out[-300:], "install_seconds": install_s,
                    "venv_python": venv_python, "test_cmd": ftp_cmd}
        elif pre_res.timed_out:
            return {"ok": False, "error": "FAIL_TO_PASS timed out pre-gold",
                    "notes": pre_out[-300:], "install_seconds": install_s,
                    "venv_python": venv_python, "test_cmd": ftp_cmd}

        # [5] Apply gold patch
        if verbose:
            print("    [5] apply gold patch", flush=True)
        ok_gp, gp_err = _apply_patch(gpatch, sb)
        if not ok_gp:
            return {"ok": False, "error": f"gold patch: {gp_err}", "notes": "",
                    "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}

        # [6] FAIL_TO_PASS must PASS after gold
        if verbose:
            print("    [6] FAIL_TO_PASS (must pass post-gold)", flush=True)
        post_res = sb.exec(ftp_cmd, timeout=TEST_TIMEOUT)
        post_pass = post_res.exit_code == 0
        post_out = (post_res.stdout + "\\n" + post_res.stderr).strip()
        if not post_pass:
            return {"ok": False, "error": "FAIL_TO_PASS still fails after gold patch",
                    "notes": post_out[-300:], "install_seconds": install_s,
                    "venv_python": venv_python, "test_cmd": ftp_cmd}
        elif post_res.timed_out:
            return {"ok": False, "error": "FAIL_TO_PASS timed out post-gold",
                    "notes": post_out[-300:], "install_seconds": install_s,
                    "venv_python": venv_python, "test_cmd": ftp_cmd}

        # [7] PASS_TO_PASS smoke
        if ptp and ptp_cmd:
            if verbose:
                print(f"    [7] PASS_TO_PASS smoke ({len(ptp)} tests)", flush=True)
            ptp_res = sb.exec(ptp_cmd, timeout=TEST_TIMEOUT)
            ptp_pass = ptp_res.exit_code == 0
            ptp_out = (ptp_res.stdout + "\\n" + ptp_res.stderr).strip()
            if not ptp_pass:
                return {"ok": False, "error": "PASS_TO_PASS smoke failed after gold",
                        "notes": ptp_out[-300:], "install_seconds": install_s,
                        "venv_python": venv_python, "test_cmd": ftp_cmd}
            elif ptp_res.timed_out:
                return {"ok": False, "error": "PASS_TO_PASS timed out",
                        "notes": ptp_out[-300:], "install_seconds": install_s,
                        "venv_python": venv_python, "test_cmd": ftp_cmd}

        elapsed = round(time.time() - t0, 1)
        return {"ok": True, "error": "", "notes": f"install={install_s:.0f}s total={elapsed:.0f}s",
                "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}
    finally:
        sb.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build bench/instances.json from SWE-bench Lite.")
    ap.add_argument("--max-instances", type=int, default=TARGET_MAX)
    ap.add_argument("--repos", default="",
                    help="Comma-separated repos (overrides PREFERRED_REPOS)")
    ap.add_argument("--workdir", default="",
                    help="Persistent dir for clones (default: system temp)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verbose", dest="verbose", action="store_false", default=True)
    args = ap.parse_args()

    target_repos = (
        [r.strip() for r in args.repos.split(",") if r.strip()]
        if args.repos else PREFERRED_REPOS
    )
    repo_order = {r: i for i, r in enumerate(target_repos)}

    bench_dir = _ROOT / "bench"
    gold_dir  = bench_dir / "gold"
    inst_path = bench_dir / "instances.json"
    gold_dir.mkdir(parents=True, exist_ok=True)

    rows = load_swebench_lite()
    candidates = sorted(
        [r for r in rows
         if r["repo"] in target_repos and r["repo"] not in SKIP_REPOS],
        key=lambda r: (repo_order.get(r["repo"], 99), r["instance_id"]),
    )
    print(f"Candidates: {len(candidates)} from "
          f"{len(set(r['repo'] for r in candidates))} repos\n")

    if args.dry_run:
        for c in candidates[:30]:
            print(f"  {c['instance_id']}  ({c['repo']})")
        print(f"  … {len(candidates)} total")
        return

    use_tmp = not args.workdir
    workdir = (Path(args.workdir) if args.workdir
               else Path(tempfile.mkdtemp(prefix="anvil_swe_")))
    workdir.mkdir(parents=True, exist_ok=True)
    if use_tmp:
        print(f"Temp workdir: {workdir}\n")

    verified: list[dict[str, Any]] = []
    table:    list[dict[str, Any]] = []
    seen:     set[str] = set()
    per_repo_cap = ceil(args.max_instances / 3)

    def _process(inst: dict[str, Any]) -> None:
        iid  = inst["instance_id"]
        repo = inst["repo"]
        print(f"\n{'='*72}\n  {iid}  ({repo})\n{'='*72}", flush=True)
        gc = gold_check(inst, workdir, args.verbose)
        table.append({
            "instance": iid, "repo": repo,
            "install_s": int(gc["install_seconds"]),
            "gold_ok": "✓ PASS" if gc["ok"] else "✗ FAIL",
            "why_dropped": "" if gc["ok"] else gc["error"][:72],
        })
        if gc["ok"]:
            ftp = _parse_test_list(inst["FAIL_TO_PASS"])
            ptp = _parse_test_list(inst["PASS_TO_PASS"])[:MAX_PTP_SAMPLE]
            verified.append({
                "instance_id": iid, "repo": repo,
                "base_commit": inst["base_commit"],
                "problem_statement": inst["problem_statement"],
                "test_patch": inst["test_patch"],
                "FAIL_TO_PASS": ftp, "PASS_TO_PASS": ptp,
                "test_cmd": gc["test_cmd"],
                "difficulty_hint": _difficulty_hint(inst["patch"]),
            })
            (gold_dir / f"{iid}.patch").write_text(inst["patch"], encoding="utf-8")
            print(f"  ✓  {gc['notes']}", flush=True)
        else:
            print(f"  ✗  {gc['error']}", flush=True)
            if gc.get("notes"):
                print(f"     {gc['notes'][-200:]}", flush=True)

    # Pass 1 — spread across repos
    for inst in candidates:
        if len(verified) >= args.max_instances:
            break
        iid  = inst["instance_id"]
        repo = inst["repo"]
        if iid in seen:
            continue
        repo_n = sum(1 for v in verified if v["repo"] == repo)
        if repo_n >= per_repo_cap and len(verified) < TARGET_MIN:
            continue
        seen.add(iid)
        _process(inst)

    # Pass 2 — fill to TARGET_MIN
    if len(verified) < TARGET_MIN:
        print(f"\n[Pass 2: {len(verified)} verified; filling to {TARGET_MIN}]", flush=True)
        for inst in candidates:
            if len(verified) >= TARGET_MIN:
                break
            iid = inst["instance_id"]
            if iid in seen:
                continue
            seen.add(iid)
            _process(inst)

    # Write instances.json
    inst_path.write_text(json.dumps(verified, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\n→ Wrote {len(verified)} instances to {inst_path}")
    print(f"→ Gold patches in {gold_dir}/")

    # Summary table
    W = [38, 26, 10, 10, 72]
    HDR = ["instance", "repo", "install_s", "gold_check", "why_dropped"]
    SEP = "  ".join("-" * w for w in W)
    total_w = sum(W) + 2 * (len(W) - 1)
    print(f"\n{'='*total_w}")
    print("GOLD-CHECK SUMMARY  — share this table with Parrv and Sneha")
    print(f"{'='*total_w}")
    print("  ".join(h.ljust(w) for h, w in zip(HDR, W)))
    print(SEP)
    for r in table:
        print("  ".join([
            r["instance"][:W[0]].ljust(W[0]),
            r["repo"][:W[1]].ljust(W[1]),
            str(r["install_s"]).ljust(W[2]),
            r["gold_ok"].ljust(W[3]),
            r["why_dropped"][:W[4]],
        ]))
    print(SEP)
    repos_done = sorted(set(v["repo"] for v in verified))
    print(f"\nVerified: {len(verified)} across {len(repos_done)} repos:")
    for rp in repos_done:
        n = sum(1 for v in verified if v["repo"] == rp)
        print(f"  {rp}: {n}")
    if len(verified) < TARGET_MIN:
        print(f"\nWARNING: only {len(verified)} verified (target {TARGET_MIN}–{TARGET_MAX}).")

    if use_tmp:
        print(f"\nCleaning up {workdir}…")
        shutil.rmtree(workdir, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
