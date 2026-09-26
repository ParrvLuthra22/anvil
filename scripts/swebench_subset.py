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


def _clone_at_commit(repo: str, base_commit: str, dest: Path) -> tuple[bool, str]:
    """Clone repo at a specific SHA. Returns (ok, error_msg)."""
    url = f"https://github.com/{repo}.git"
    env = _git_env()
    dest.mkdir(parents=True, exist_ok=True)

    # Try: init + shallow fetch of exact SHA
    subprocess.run(["git", "init"], cwd=dest, capture_output=True, env=env, timeout=15)
    r = subprocess.run(
        ["git", "fetch", "--depth", "1", url, base_commit],
        cwd=dest, capture_output=True, text=True, timeout=120, env=env,
    )
    if r.returncode == 0:
        co = subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=dest, capture_output=True, text=True, timeout=30, env=env,
        )
        if co.returncode == 0:
            return True, ""

    # Fallback: full shallow clone + checkout
    shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    r2 = subprocess.run(
        ["git", "clone", "--depth", "50", url, str(dest)],
        capture_output=True, text=True, timeout=180, env=env,
    )
    if r2.returncode != 0:
        return False, f"git clone: {r2.stderr[:300]}"
    co2 = subprocess.run(
        ["git", "checkout", base_commit],
        cwd=dest, capture_output=True, text=True, timeout=30, env=env,
    )
    if co2.returncode != 0:
        return False, f"git checkout: {co2.stderr[:200]}"
    return True, ""


def _pick_python(repo_dir: Path) -> str | None:
    """Return path to a working python3 interpreter."""
    for cand in PYTHON_CANDIDATES:
        r = subprocess.run(["which", cand], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


def _create_venv(repo_dir: Path, interp: str) -> tuple[str | None, str]:
    """Create .anvil_venv inside repo_dir. Returns (python_path, error)."""
    venv_dir = repo_dir / ".anvil_venv"
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    r = subprocess.run(
        [interp, "-m", "venv", str(venv_dir)],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        return None, f"venv create: {r.stderr[:200]}"
    return str(venv_dir / "bin" / "python"), ""


# Common test infrastructure packages always installed into the venv.
_TEST_PKGS = ["pytest", "pytest-xdist", "pytest-timeout", "hypothesis"]
# Repo-specific extra requirements files to install if present.
_REQ_FILES = [
    "requirements-test.txt", "requirements_test.txt",
    "requirements-dev.txt",  "requirements_dev.txt",
    "requirements-testing.txt",
]


def _pip_install(venv_python: str, repo_dir: Path) -> tuple[bool, str]:
    """Install repo + test infrastructure in venv. Returns (ok, report)."""
    env = _git_env()
    env.pop("GITHUB_TOKEN", None)

    # Upgrade pip (silently)
    subprocess.run(
        [venv_python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        capture_output=True, cwd=repo_dir, timeout=60, env=env,
    )

    # Install project: try extras in order, settle for plain -e .
    installed_extras = False
    for extras in [".[test]", ".[testing]", ".[dev]", ".[tests]", "."]:
        r = subprocess.run(
            [venv_python, "-m", "pip", "install", "--quiet", "-e", extras],
            capture_output=True, text=True, cwd=repo_dir, timeout=180, env=env,
        )
        if r.returncode == 0:
            installed_extras = True
            break

    if not installed_extras:
        # Last resort: setup.py develop
        setup_py = repo_dir / "setup.py"
        if setup_py.exists():
            r2 = subprocess.run(
                [venv_python, "setup.py", "develop"],
                capture_output=True, text=True, cwd=repo_dir, timeout=180, env=env,
            )
            if r2.returncode != 0:
                return False, f"setup.py develop failed: {r2.stderr[-300:]}"
        else:
            return False, f"all install attempts failed: {r.stderr[-300:]}"

    # Always install pytest + common test infrastructure
    subprocess.run(
        [venv_python, "-m", "pip", "install", "--quiet"] + _TEST_PKGS,
        capture_output=True, cwd=repo_dir, timeout=120, env=env,
    )

    # Install any repo-level requirements-test / requirements-dev files
    for req_name in _REQ_FILES:
        req_path = repo_dir / req_name
        if req_path.exists():
            subprocess.run(
                [venv_python, "-m", "pip", "install", "--quiet", "-r", req_name],
                capture_output=True, cwd=repo_dir, timeout=120, env=env,
            )

    return True, "install OK"


def _apply_patch(patch_text: str, repo_dir: Path) -> tuple[bool, str]:
    """Apply a unified diff to repo_dir using 'patch -p1'. Returns (ok, stderr)."""
    env = _git_env()
    r = subprocess.run(
        ["patch", "-p1", "--forward", "--batch", "--no-backup-if-mismatch"],
        input=patch_text, text=True, capture_output=True,
        cwd=repo_dir, timeout=30, env=env,
    )
    if r.returncode == 0:
        return True, ""
    # Try git apply as fallback
    r2 = subprocess.run(
        ["git", "apply", "--reject", "-"],
        input=patch_text, text=True, capture_output=True,
        cwd=repo_dir, timeout=30, env=env,
    )
    return r2.returncode == 0, (r2.stderr or r.stderr)[:400]


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


def _run_cmd(cmd: str, cwd: Path, timeout: int) -> tuple[bool, str]:
    env = _git_env()
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=cwd, timeout=timeout, env=env,
    )
    out = (r.stdout + "\n" + r.stderr).strip()
    return r.returncode == 0, out[-800:]


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
    """Run full 6-step gold check. Never raises."""
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

    repo_dir = workdir / iid
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    t0 = time.time()

    # [1] Clone
    if verbose:
        print(f"    [1] clone {repo}@{base[:8]}", flush=True)
    ok, err = _clone_at_commit(repo, base, repo_dir)
    if not ok:
        return _fail(f"clone: {err}")

    # [2] Install deps
    if verbose:
        print("    [2] install deps", flush=True)
    interp = _pick_python(repo_dir)
    if not interp:
        return _fail("no python3 found on PATH")
    t_deps = time.time()
    venv_python, venv_err = _create_venv(repo_dir, interp)
    if not venv_python:
        return _fail(f"venv: {venv_err}")
    pip_ok, pip_report = _pip_install(venv_python, repo_dir)
    install_s = round(time.time() - t_deps, 1)
    if not pip_ok:
        return _fail(f"deps: {pip_report}", install_s)

    ftp_cmd = _build_test_cmd(venv_python, repo, ftp)
    ptp_cmd = _build_test_cmd(venv_python, repo, ptp) if ptp else None

    # [3] Apply test_patch
    if verbose:
        print("    [3] apply test_patch", flush=True)
    ok_tp, tp_err = _apply_patch(tpatch, repo_dir)
    if not ok_tp:
        return {"ok": False, "error": f"test_patch: {tp_err}", "notes": "",
                "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}

    # [4] FAIL_TO_PASS must FAIL before gold
    if verbose:
        print(f"    [4] FAIL_TO_PASS (must fail pre-gold): {ftp_cmd[:70]}", flush=True)
    try:
        pre_pass, pre_out = _run_cmd(ftp_cmd, repo_dir, TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        pre_pass, pre_out = False, "TIMEOUT"
    if pre_pass:
        return {"ok": False, "error": "FAIL_TO_PASS passed before gold (test doesn't cover bug)",
                "notes": pre_out[-300:], "install_seconds": install_s,
                "venv_python": venv_python, "test_cmd": ftp_cmd}

    # [5] Apply gold patch
    if verbose:
        print("    [5] apply gold patch", flush=True)
    ok_gp, gp_err = _apply_patch(gpatch, repo_dir)
    if not ok_gp:
        return {"ok": False, "error": f"gold patch: {gp_err}", "notes": "",
                "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}

    # [6] FAIL_TO_PASS must PASS after gold
    if verbose:
        print("    [6] FAIL_TO_PASS (must pass post-gold)", flush=True)
    try:
        post_pass, post_out = _run_cmd(ftp_cmd, repo_dir, TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        post_pass, post_out = False, "TIMEOUT"
    if not post_pass:
        return {"ok": False, "error": "FAIL_TO_PASS still fails after gold patch",
                "notes": post_out[-300:], "install_seconds": install_s,
                "venv_python": venv_python, "test_cmd": ftp_cmd}

    # [7] PASS_TO_PASS smoke
    if ptp and ptp_cmd:
        if verbose:
            print(f"    [7] PASS_TO_PASS smoke ({len(ptp)} tests)", flush=True)
        try:
            ptp_pass, ptp_out = _run_cmd(ptp_cmd, repo_dir, TEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            ptp_pass, ptp_out = False, "TIMEOUT"
        if not ptp_pass:
            return {"ok": False, "error": "PASS_TO_PASS smoke failed after gold",
                    "notes": ptp_out[-300:], "install_seconds": install_s,
                    "venv_python": venv_python, "test_cmd": ftp_cmd}

    elapsed = round(time.time() - t0, 1)
    return {"ok": True, "error": "", "notes": f"install={install_s:.0f}s total={elapsed:.0f}s",
            "install_seconds": install_s, "venv_python": venv_python, "test_cmd": ftp_cmd}


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
