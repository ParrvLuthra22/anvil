#!/usr/bin/env python3
"""Build bench/instances.json from SWE-bench Lite.

Downloads SWE-bench Lite from HuggingFace (datasets library), filters to
preferred pure-Python repos, runs a GOLD CHECK for each candidate instance
inside a WorktreeSandbox, and writes only verified instances to
bench/instances.json.  Gold patches are saved separately to
bench/gold/<instance_id>.patch (never included in instances.json).

Usage::

    python scripts/swebench_subset.py [--max-instances N] [--dry-run]
    python scripts/swebench_subset.py --help

Requirements: pip install datasets
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

# ---------------------------------------------------------------------------
# Path setup — make src/ importable when run as a script
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from anvil.repo.deps import DepsResult, ensure_deps  # noqa: E402
from anvil.repo.profile import profile_repo           # noqa: E402
from anvil.sandbox.worktree import WorktreeSandbox    # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
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

# Heavy compile deps or broken CI — skip unconditionally
SKIP_REPOS: set[str] = {
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "astropy/astropy",
}

TARGET_MIN = 8
TARGET_MAX = 12
TEST_TIMEOUT = 120        # seconds per individual test run
DEPS_TIMEOUT = 300        # seconds for ensure_deps
MAX_PTP_SAMPLE = 5        # how many PASS_TO_PASS tests to smoke-check


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_swebench_lite() -> list[dict[str, Any]]:
    """Load SWE-bench Lite 'test' split from HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: 'datasets' package not found.\n"
            "Fix: pip install datasets\nThen re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Downloading SWE-bench Lite from HuggingFace…", flush=True)
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    except Exception as exc:
        print(
            f"ERROR: Failed to download SWE-bench Lite: {exc}\n"
            "Check your internet connection / HF_TOKEN.\n"
            ">>> Report this failure to Parrv immediately. <<<",
            file=sys.stderr,
        )
        sys.exit(1)

    rows: list[dict[str, Any]] = [dict(row) for row in ds]
    print(f"  Loaded {len(rows)} instances.", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_test_list(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return list(raw)
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [x.strip() for x in raw.splitlines() if x.strip()]


def _build_test_cmd(repo: str, test_ids: list[str], venv_python: str | None) -> str:
    """Return the correct test command for a repo and list of test IDs."""
    py = venv_python or "python3"
    if repo.startswith("django/"):
        # test_ids look like tests/aggregation/tests.py::Class::method
        modules: list[str] = []
        for tid in test_ids:
            path = tid.split("::")[0]
            path = path.removeprefix("tests/").removesuffix(".py")
            mod = path.replace("/", ".")
            if mod not in modules:
                modules.append(mod)
        return f"{py} tests/runtests.py --verbosity=2 {' '.join(modules)}"
    # Default: pytest with explicit test IDs
    ids = " ".join(test_ids)
    return f"{py} -m pytest -x -q --tb=short {ids}"


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _clone_at_commit(repo: str, base_commit: str, dest: Path) -> tuple[bool, str]:
    """Clone *repo* from GitHub at *base_commit* into *dest*.  Returns (ok, error)."""
    url = f"https://github.com/{repo}.git"
    env = _git_env()
    dest.mkdir(parents=True, exist_ok=True)

    # Step 1: init + shallow fetch of exact SHA
    subprocess.run(["git", "init"], cwd=dest, capture_output=True, env=env, timeout=15)
    r = subprocess.run(
        ["git", "fetch", "--depth", "1", url, base_commit],
        cwd=dest, capture_output=True, text=True, timeout=90, env=env,
    )
    if r.returncode == 0:
        co = subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=dest, capture_output=True, text=True, timeout=30, env=env,
        )
        if co.returncode == 0:
            return True, ""

    # Step 2: full shallow clone then checkout
    shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    r2 = subprocess.run(
        ["git", "clone", "--depth", "50", url, str(dest)],
        capture_output=True, text=True, timeout=180, env=env,
    )
    if r2.returncode != 0:
        return False, f"git clone failed: {r2.stderr[:300]}"
    co2 = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=dest, capture_output=True, text=True, timeout=30, env=env,
    )
    if co2.returncode != 0:
        return False, f"git checkout failed: {co2.stderr[:200]}"
    return True, ""


def _apply_patch_via_sandbox(sb: WorktreeSandbox, patch_text: str, label: str) -> tuple[bool, str]:
    """Write patch to sandbox and apply via git apply.  Returns (ok, stderr)."""
    tmp_patch = f".anvil_{label}.patch"
    sb.write_file(tmp_patch, patch_text)
    r = sb.exec(f"git apply --index {tmp_patch}", timeout=30)
    if r.exit_code == 0:
        return True, ""
    # Try without --index (some repos need this)
    r2 = sb.exec(f"git apply {tmp_patch}", timeout=30)
    return r2.exit_code == 0, r2.stderr[:300]


def _run_tests(sb: WorktreeSandbox, cmd: str) -> tuple[bool, str]:
    r = sb.exec(cmd, timeout=TEST_TIMEOUT)
    out = (r.stdout + "\n" + r.stderr).strip()
    return (r.exit_code == 0 and not r.timed_out), out[-600:]


def _difficulty_hint(patch: str) -> str:
    additions = sum(1 for ln in patch.splitlines()
                    if ln.startswith("+") and not ln.startswith("+++"))
    if additions <= 5:   return "trivial"
    if additions <= 20:  return "easy"
    if additions <= 60:  return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Gold check
# ---------------------------------------------------------------------------
def gold_check(inst: dict[str, Any], workdir: Path, verbose: bool) -> dict[str, Any]:
    """Full 6-step gold check.  Never raises; returns a result dict."""
    iid   = inst["instance_id"]
    repo  = inst["repo"]
    base  = inst["base_commit"]
    tpatch = inst["test_patch"]
    gpatch = inst["patch"]
    ftp   = _parse_test_list(inst["FAIL_TO_PASS"])
    ptp   = _parse_test_list(inst["PASS_TO_PASS"])[:MAX_PTP_SAMPLE]

    def fail(msg: str, notes: str = "") -> dict[str, Any]:
        return {"ok": False, "error": msg, "notes": notes,
                "install_seconds": 0.0, "venv_python": None, "test_cmd": ""}

    repo_dir = workdir / f"repo_{iid}"
    work_dir = workdir / f"work_{iid}"
    for d in (repo_dir, work_dir):
        if d.exists():
            shutil.rmtree(d)

    t_start = time.time()

    # 1. Clone
    if verbose: print(f"    [1] clone {repo}@{base[:8]}", flush=True)
    ok, err = _clone_at_commit(repo, base, repo_dir)
    if not ok:
        return fail(f"clone: {err}")

    # 2. ensure_deps
    if verbose: print("    [2] ensure_deps", flush=True)
    try:
        sb = WorktreeSandbox(repo_root=repo_dir, work_dir=work_dir, char_cap=8000)
    except Exception as exc:
        return fail(f"sandbox init: {exc}")

    profile = profile_repo(repo_dir)
    t_deps = time.time()
    deps: DepsResult = ensure_deps(sb, profile)
    install_s = round(time.time() - t_deps, 1)

    if not deps.ok and not deps.skipped:
        sb.close()
        return {"ok": False, "error": f"deps: {deps.report[:200]}", "notes": "",
                "install_seconds": install_s, "venv_python": None, "test_cmd": ""}

    venv_python = deps.venv_python
    ftp_cmd = _build_test_cmd(repo, ftp, venv_python)
    ptp_cmd = _build_test_cmd(repo, ptp, venv_python) if ptp else None

    # 3. Apply test_patch
    if verbose: print("    [3] apply test_patch", flush=True)
    ok_tp, tp_err = _apply_patch_via_sandbox(sb, tpatch, "test")
    if not ok_tp:
        sb.close()
        return {"ok": False, "error": f"test_patch apply: {tp_err}",
                "install_seconds": install_s, "venv_python": venv_python,
                "test_cmd": ftp_cmd, "notes": ""}

    # 4. FAIL_TO_PASS must FAIL before gold patch
    if verbose: print(f"    [4] FAIL_TO_PASS pre-gold (must fail): {ftp_cmd[:70]}", flush=True)
    passed_pre, pre_out = _run_tests(sb, ftp_cmd)
    if passed_pre:
        sb.close()
        return {"ok": False,
                "error": "FAIL_TO_PASS passed BEFORE gold patch (test doesn't cover the bug)",
                "install_seconds": install_s, "venv_python": venv_python,
                "test_cmd": ftp_cmd, "notes": pre_out}

    # 5. Apply gold patch
    if verbose: print("    [5] apply gold patch", flush=True)
    ok_gp, gp_err = _apply_patch_via_sandbox(sb, gpatch, "gold")
    if not ok_gp:
        sb.close()
        return {"ok": False, "error": f"gold patch apply: {gp_err}",
                "install_seconds": install_s, "venv_python": venv_python,
                "test_cmd": ftp_cmd, "notes": ""}

    # 6. FAIL_TO_PASS must PASS after gold patch
    if verbose: print("    [6] FAIL_TO_PASS post-gold (must pass)", flush=True)
    passed_post, post_out = _run_tests(sb, ftp_cmd)
    if not passed_post:
        sb.close()
        return {"ok": False, "error": "FAIL_TO_PASS still fails after gold patch",
                "install_seconds": install_s, "venv_python": venv_python,
                "test_cmd": ftp_cmd, "notes": post_out}

    # 7. PASS_TO_PASS smoke-check
    if ptp and ptp_cmd:
        if verbose: print(f"    [7] PASS_TO_PASS smoke ({len(ptp)} tests)", flush=True)
        ptp_pass, ptp_out = _run_tests(sb, ptp_cmd)
        if not ptp_pass:
            sb.close()
            return {"ok": False, "error": "PASS_TO_PASS smoke failed after gold patch",
                    "install_seconds": install_s, "venv_python": venv_python,
                    "test_cmd": ftp_cmd, "notes": ptp_out}

    sb.close()
    elapsed = round(time.time() - t_start, 1)
    return {"ok": True, "error": "", "notes": f"install={install_s:.0f}s total={elapsed:.0f}s",
            "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}


# Small hack so fail() can accept keyword install_seconds
# (already handled inline above — the function signature above ignores it)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build bench/instances.json from SWE-bench Lite.")
    ap.add_argument("--max-instances", type=int, default=TARGET_MAX)
    ap.add_argument("--repos", default="",
                    help="Comma-separated repos to include (overrides PREFERRED_REPOS)")
    ap.add_argument("--workdir", default="",
                    help="Persistent directory for clones (default: auto temp)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print candidates without running checks")
    ap.add_argument("--no-verbose", dest="verbose", action="store_false", default=True)
    args = ap.parse_args()

    target_repos = (
        [r.strip() for r in args.repos.split(",") if r.strip()]
        if args.repos else PREFERRED_REPOS
    )

    bench_dir  = _ROOT / "bench"
    gold_dir   = bench_dir / "gold"
    inst_path  = bench_dir / "instances.json"
    gold_dir.mkdir(parents=True, exist_ok=True)

    # --- load & filter ---
    rows = load_swebench_lite()
    repo_order = {r: i for i, r in enumerate(target_repos)}
    candidates = sorted(
        [r for r in rows if r["repo"] in target_repos and r["repo"] not in SKIP_REPOS],
        key=lambda r: (repo_order.get(r["repo"], 99), r["instance_id"]),
    )
    print(f"Candidates: {len(candidates)} from {len(set(r['repo'] for r in candidates))} repos\n")

    if args.dry_run:
        for c in candidates:
            print(f"  {c['instance_id']}  ({c['repo']})")
        return

    # --- workdir ---
    use_tmp = not args.workdir
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="anvil_swe_"))
    workdir.mkdir(parents=True, exist_ok=True)
    if use_tmp:
        print(f"Temp workdir: {workdir}\n")

    verified: list[dict[str, Any]] = []
    table: list[dict[str, Any]]    = []
    per_repo_cap = ceil(args.max_instances / 3)

    def _process(inst: dict[str, Any]) -> bool:
        """Run gold check, record result. Returns True if verified."""
        iid  = inst["instance_id"]
        repo = inst["repo"]
        print(f"\n{'='*72}")
        print(f"  {iid}  ({repo})")
        print(f"{'='*72}")
        gc = gold_check(inst, workdir, args.verbose)
        table.append({
            "instance": iid,
            "repo": repo,
            "install_s": int(gc["install_seconds"]),
            "gold_ok": "✓ PASS" if gc["ok"] else "✗ FAIL",
            "why_dropped": "" if gc["ok"] else gc["error"][:72],
        })
        if gc["ok"]:
            ftp = _parse_test_list(inst["FAIL_TO_PASS"])
            ptp = _parse_test_list(inst["PASS_TO_PASS"])[:MAX_PTP_SAMPLE]
            verified.append({
                "instance_id": iid,
                "repo": repo,
                "base_commit": inst["base_commit"],
                "problem_statement": inst["problem_statement"],
                "test_patch": inst["test_patch"],
                "FAIL_TO_PASS": ftp,
                "PASS_TO_PASS": ptp,
                "test_cmd": gc["test_cmd"],
                "difficulty_hint": _difficulty_hint(inst["patch"]),
            })
            (gold_dir / f"{iid}.patch").write_text(inst["patch"], encoding="utf-8")
            print(f"  ✓  {gc['notes']}")
            return True
        else:
            print(f"  ✗  {gc['error']}")
            return False

    already_seen: set[str] = set()

    # First pass — spread across repos (per_repo_cap limit)
    for inst in candidates:
        if len(verified) >= args.max_instances:
            break
        iid  = inst["instance_id"]
        repo = inst["repo"]
        if already_seen.intersection({iid}):
            continue
        repo_n = sum(1 for v in verified if v["repo"] == repo)
        if repo_n >= per_repo_cap and len(verified) < TARGET_MIN:
            continue  # skip — get more repos first
        already_seen.add(iid)
        _process(inst)

    # Second pass — fill to TARGET_MIN if needed (no per-repo cap)
    if len(verified) < TARGET_MIN:
        print(f"\n[Second pass — only {len(verified)} verified; filling to {TARGET_MIN}]")
        for inst in candidates:
            if len(verified) >= TARGET_MIN:
                break
            iid = inst["instance_id"]
            if iid in already_seen:
                continue
            already_seen.add(iid)
            _process(inst)

    # --- write instances.json ---
    inst_path.write_text(json.dumps(verified, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n\nWrote {len(verified)} instances → {inst_path}")
    print(f"Gold patches → {gold_dir}/")

    # --- summary table ---
    W = [36, 26, 10, 10, 72]
    HDR = ["instance", "repo", "install_s", "gold_check", "why_dropped"]
    SEP = "  ".join("-" * w for w in W)
    print("\n" + "=" * (sum(W) + 2 * len(W)))
    print("GOLD-CHECK SUMMARY TABLE")
    print("=" * (sum(W) + 2 * len(W)))
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
    if len(verified) < TARGET_MIN:
        print(f"\nWARNING: only {len(verified)} verified (target {TARGET_MIN}–{TARGET_MAX}).")
    repos_done = sorted(set(v["repo"] for v in verified))
    print(f"\nVerified: {len(verified)} instances across {len(repos_done)} repos:")
    for rp in repos_done:
        n = sum(1 for v in verified if v["repo"] == rp)
        print(f"  {rp}: {n}")

    if use_tmp:
        print(f"\nCleaning temp workdir {workdir}…")
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n>>> Table above should be shared with Parrv and Sneha. <<<")


if __name__ == "__main__":
    main()
