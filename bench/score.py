#!/usr/bin/env python3
"""Apply and evaluate benchmark patches in disposable checkouts."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

from anvil.sandbox.base import Sandbox

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {"no_patch", "empty_patch", "patch_does_not_apply", "f2p_fail", "p2p_regression", "timeout", "harness_error"}


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run a bounded command without allowing it to prompt for input."""
    from anvil.sandbox.worktree import _sanitized_env

    env = _sanitized_env()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, env=env)


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


def _test_argv(test_cmd: str, targets: list[str], venv_python: Path) -> list[str]:
    """Use recorded test flags and targets with this instance's isolated Python."""
    words = shlex.split(test_cmd)
    # SWE-bench test_cmd ends with selected test node ids. Preserve the runner
    # and options while replacing those nodes with the requested oracle set.
    first_target = next((i for i, word in enumerate(words) if ".py::" in word or word.endswith(".py")), len(words))
    prefix = words[:first_target]
    if not prefix:
        raise RuntimeError("test_cmd must name a Python test runner")
    if prefix[0] in {"python", "python3", ".anvil_venv/bin/python"} or prefix[0].endswith("/.anvil_venv/bin/python"):
        prefix[0] = str(venv_python)
    elif Path(prefix[0]).name == "pytest":
        prefix = [str(venv_python), "-m", "pytest", *prefix[1:]]
    else:
        raise RuntimeError(f"test_cmd runner must use Python or pytest, got {prefix[0]!r}")
    return [*prefix, *targets]


def _instance_sandbox(repo: Path, work_dir: Path) -> Sandbox:
    """Create a real isolated worktree for dependency installation and scoring."""
    from anvil.sandbox.worktree import WorktreeSandbox

    return WorktreeSandbox(repo, work_dir)


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

        from dataclasses import replace
        from anvil.repo.deps import ensure_deps
        from anvil.repo.profile import profile_repo

        profile = profile_repo(repo)
        profile = replace(profile, primary_language="python")
        if instance.get("install_cmd"):
            profile = replace(profile, install_cmd=instance["install_cmd"])
        # All curated benchmark instances are Python. Even a minimal fixture
        # without package metadata gets an instance-local venv and pytest.
        if not profile.install_cmd:
            profile = replace(profile, install_cmd="python -m pip install pytest")

        # Make the candidate patch and hidden oracle part of the disposable
        # base checkout before creating the isolated scoring worktree.
        from anvil.sandbox.worktree import _sanitized_env

        env = {
            **_sanitized_env(),
            "GIT_AUTHOR_NAME": "ANVIL benchmark",
            "GIT_AUTHOR_EMAIL": "anvil-bench@localhost",
            "GIT_COMMITTER_NAME": "ANVIL benchmark",
            "GIT_COMMITTER_EMAIL": "anvil-bench@localhost",
        }
        staged = _run(["git", "add", "-A"], repo)
        committed = subprocess.run(
            ["git", "commit", "-m", "ANVIL score fixture"], cwd=repo,
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60,
            env=env,
        )
        if staged.returncode or committed.returncode:
            raise RuntimeError(f"could not prepare scoring worktree: {committed.stderr[-800:]}")

        sandbox = _instance_sandbox(repo, work_dir / f"score-{instance['instance_id']}")
        try:
            installed = ensure_deps(
                sandbox,
                profile,
                as_of=instance.get("base_commit_date") or None,
            )
            if not installed.ok:
                if "timed out" in installed.report.lower():
                    result.update(resolved=False, category="timeout", error=installed.report)
                    return result
                raise RuntimeError(f"dependency installation failed: {installed.report[-800:]}")
            if not installed.venv_python:
                raise RuntimeError("ensure_deps did not provide an isolated Python interpreter")
            venv_python = sandbox.root / installed.venv_python
            if not venv_python.is_file():
                raise RuntimeError(f"instance venv interpreter was not created: {venv_python}")

            py_compile = _run([str(venv_python), "-m", "compileall", "-q", "."], sandbox.root, timeout=120)
            if py_compile.returncode:
                result.update(resolved=False, category="patch_does_not_apply")
                return result

            test_cmd = str(instance.get("test_cmd", "python -m pytest -q"))
            for category, targets in (("f2p_fail", instance.get("FAIL_TO_PASS", [])), ("p2p_regression", instance.get("PASS_TO_PASS", []))):
                if not targets:
                    continue
                try:
                    tested = _run(_test_argv(test_cmd, [str(t) for t in targets], venv_python), sandbox.root, timeout=600)
                except subprocess.TimeoutExpired:
                    result.update(resolved=False, category="timeout")
                    return result
                if tested.returncode:
                    result.update(resolved=False, category=category, test_output=(tested.stdout + tested.stderr)[-2000:])
                    return result
            result.update(resolved=True, category="resolved")
        finally:
            sandbox.close()
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
