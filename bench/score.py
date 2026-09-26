#!/usr/bin/env python3
"""Apply and evaluate benchmark patches in disposable checkouts."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"no_patch", "empty_patch", "patch_does_not_apply", "f2p_fail", "p2p_regression", "timeout", "harness_error"}


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run a bounded command without allowing it to prompt for input."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})


def _repo_url(value: str) -> str:
    """Resolve fixture-relative repository URLs while preserving remote URLs."""
    if value.startswith("file://"):
        return value[7:]
    candidate = Path(value)
    if not candidate.is_absolute() and (ROOT / candidate).exists():
        return str((ROOT / candidate).resolve())
    if candidate.exists():
        return str(candidate.resolve())
    return value if "://" in value else f"https://github.com/{value}"


def _command(text: str, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Execute a recorded shell command, adapting benchmark venv Python paths."""
    words = shlex.split(text)
    if not words:
        return subprocess.CompletedProcess([], 0, "", "")
    words = [str(cwd / ".anvil_venv/bin/python") if w.endswith("/.anvil_venv/bin/python") or w == ".anvil_venv/bin/python" else (sys.executable if w in {"python", "python3"} else w) for w in words]
    return _run(words, cwd, timeout)


def _test_command(test_cmd: str, targets: list[str]) -> str:
    """Reuse the recorded runner and flags, substituting just its test selection."""
    words = shlex.split(test_cmd)
    # SWE-bench test_cmd ends with selected test node ids. Preserve the runner
    # and options while replacing those nodes with the requested oracle set.
    first_target = next((i for i, word in enumerate(words) if ".py::" in word or word.endswith(".py")), len(words))
    prefix = words[:first_target]
    if prefix and prefix[0] in {"python", "python3"}:
        prefix[0] = sys.executable
    return " ".join(shlex.quote(w) for w in prefix + targets)


def _score_instance(instance: dict, run_result: dict, work_dir: Path) -> dict:
    """Score one result. Oracle fields are used only here, after agent execution."""
    result = dict(run_result)
    patch = Path(result.get("patch_path", "")) if result.get("patch_path") else None
    if result.get("error"):
        result.update(resolved=False, category="timeout" if "timeout" in str(result["error"]).lower() else "harness_error")
        return result
    if patch is None or not patch.is_file():
        result.update(resolved=False, category="no_patch")
        return result
    if not patch.read_text(encoding="utf-8").strip():
        result.update(resolved=False, category="empty_patch")
        return result

    repo = work_dir / f"repo-{instance['instance_id']}"
    try:
        clone = _run(["git", "clone", "--no-checkout", _repo_url(instance["repo"]), str(repo)], work_dir)
        if clone.returncode:
            raise RuntimeError(clone.stderr[-1000:])
        checkout = _run(["git", "checkout", "--detach", instance["base_commit"]], repo)
        if checkout.returncode:
            raise RuntimeError(checkout.stderr[-1000:])
        check = _run(["git", "apply", "--check", str(patch.resolve())], repo)
        if check.returncode:
            result.update(resolved=False, category="patch_does_not_apply")
            return result
        applied = _run(["git", "apply", str(patch.resolve())], repo)
        if applied.returncode:
            result.update(resolved=False, category="patch_does_not_apply")
            return result
        # Reject syntactically invalid Python patches before tests, matching the
        # general patch validity category without making language assumptions.
        py_compile = _run([sys.executable, "-m", "compileall", "-q", "."], repo, timeout=120)
        if py_compile.returncode:
            result.update(resolved=False, category="patch_does_not_apply")
            return result
        oracle = instance.get("test_patch", "")
        if oracle:
            oracle_path = work_dir / f"oracle-{instance['instance_id']}.diff"
            oracle_path.write_text(oracle, encoding="utf-8")
            applied_oracle = _run(["git", "apply", "--check", str(oracle_path)], repo)
            if applied_oracle.returncode:
                raise RuntimeError(f"test_patch does not apply: {applied_oracle.stderr[-800:]}")
            applied_oracle = _run(["git", "apply", str(oracle_path)], repo)
            if applied_oracle.returncode:
                raise RuntimeError(f"test_patch failed: {applied_oracle.stderr[-800:]}")

        from anvil.repo.profile import profile_repo
        install_cmd = instance.get("install_cmd") or profile_repo(repo).install_cmd
        if install_cmd:
            # Keep dependency changes inside this instance's disposable tree.
            venv_python = repo / ".anvil_venv" / "bin" / "python"
            if not venv_python.exists():
                created = _run([sys.executable, "-m", "venv", str(repo / ".anvil_venv")], repo)
                if created.returncode:
                    raise RuntimeError(f"venv creation failed: {created.stderr[-800:]}")
            install_words = shlex.split(install_cmd)
            if install_words[:2] == ["pip", "install"]:
                install_words = [str(venv_python), "-m", "pip", *install_words[1:]]
            installed = _run(install_words, repo, timeout=600)
            if installed.returncode:
                raise RuntimeError(f"dependency installation failed: {installed.stderr[-800:]}")
        test_cmd = str(instance.get("test_cmd", "python -m pytest -q"))
        for category, targets in (("p2p_regression", instance.get("PASS_TO_PASS", [])), ("f2p_fail", instance.get("FAIL_TO_PASS", []))):
            if not targets:
                continue
            cmd = _test_command(test_cmd, [str(t) for t in targets])
            try:
                tested = _command(cmd, repo)
            except subprocess.TimeoutExpired:
                result.update(resolved=False, category="timeout")
                return result
            if tested.returncode:
                result.update(resolved=False, category=category, test_output=(tested.stdout + tested.stderr)[-2000:])
                return result
        result.update(resolved=True, category="resolved")
    except subprocess.TimeoutExpired:
        result.update(resolved=False, category="timeout")
    except (OSError, RuntimeError, KeyError) as exc:
        result.update(resolved=False, category="harness_error", error=str(exc))
    return result


def _read_jsonl(path: Path) -> list[dict]:
    """Read valid JSON records from a JSONL file."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if line.strip(): rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _print_summary_table(rows: list[dict]) -> None:
    """Print compact scores and the resolved rate."""
    print("\nID                       RESOLVED  CATEGORY             STEPS  TOKENS  SECONDS  LABEL")
    for r in rows:
        print(f"{r.get('instance_id','?')[:24]:24}  {str(bool(r.get('resolved'))):8}  {r.get('category','unscored'):20.20}  {r.get('steps',0):5}  {r.get('tokens',0):6}  {r.get('seconds',r.get('duration',0)):7.1f}  {r.get('label','')}")
    count = len(rows); solved = sum(bool(r.get("resolved")) for r in rows)
    print(f"Resolved: {solved}/{count} ({solved/count:.1%})" if count else "Resolved: 0/0 (0.0%)")


def main() -> None:
    """Score all completed benchmark runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=ROOT / "bench/instances.json")
    parser.add_argument("--results", type=Path, default=ROOT / "bench/results.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "bench/score.json")
    args = parser.parse_args()
    instances = {x["instance_id"]: x for x in json.loads(args.instances.read_text(encoding="utf-8"))}
    records = _read_jsonl(args.results)
    with tempfile.TemporaryDirectory(prefix="anvil-score-") as td:
        scored = [_score_instance(instances[r["instance_id"]], r, Path(td)) for r in records if r.get("instance_id") in instances]
    args.results.write_text("".join(json.dumps(r) + "\n" for r in scored), encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"instances": scored, "summary": {"total": len(scored), "resolved": sum(bool(r.get("resolved")) for r in scored)}}, indent=2), encoding="utf-8")
    _print_summary_table(scored)


if __name__ == "__main__":
    main()
