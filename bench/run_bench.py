#!/usr/bin/env python3
"""ANVIL benchmark runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_instances(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_completed(results_file: Path) -> set[str]:
    completed = set()
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "instance_id" in data:
                        completed.add(data["instance_id"])
                except Exception:
                    pass
    return completed


async def _run_instance(
    instance: dict,
    timeout: int,
    label: str,
    results_file: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    instance_id = instance["instance_id"]
    async with semaphore:
        run_dir = _REPO_ROOT / "bench" / "runs" / instance_id
        run_dir.mkdir(parents=True, exist_ok=True)

        env = {
            **os.environ,
            "ANVIL_OUTPUT_DIR": str(run_dir.resolve()),
            "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}".rstrip(":"),
        }
        if "AI_API_KEY" not in env:
            env["AI_API_KEY"] = "mock"

        repo = instance["repo"]
        if repo.startswith("file://"):
            local_p = Path(repo[7:]).resolve()
            repo = f"file://{local_p}"
        elif Path(repo).exists():
            repo = f"file://{Path(repo).resolve()}"
        elif (_REPO_ROOT / repo).exists():
            repo = f"file://{(_REPO_ROOT / repo).resolve()}"
        elif not repo.startswith("http") and "/" in repo:
            repo = f"https://github.com/{repo}"

        cmd = [
            sys.executable,
            "-m",
            "anvil",
            "--headless",
            "--repo",
            repo,
            "--ref",
            instance["base_commit"],
            "--issue-text",
            instance["problem_statement"],
        ]

        while True:
            print(f"Running {instance_id}...")
            start = time.time()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_REPO_ROOT),
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                out = stdout.decode(errors="replace")
                err = stderr.decode(errors="replace")

                # Check for rate limit
                if (
                    "429" in out
                    or "429" in err
                    or "rate limit" in out.lower()
                    or "rate limit" in err.lower()
                ):
                    print(f"Rate limit hit for {instance_id}. Sleeping 60s and retrying...")
                    await asyncio.sleep(60)
                    continue

                error = ""
                if proc.returncode != 0:
                    error = (err or out)[:200]
                break

            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except OSError:
                    pass
                error = f"timeout after {timeout}s"
                break
            except Exception as e:
                error = str(e)[:200]
                break

        duration = round(time.time() - start, 1)

        # Parse trace for done event to get tokens/steps
        trace_path = run_dir / "trace.jsonl"
        steps = 0
        tokens = 0
        if trace_path.exists():
            try:
                from anvil.trace.recorder import TraceRecorder

                for ev in TraceRecorder.load(trace_path):
                    if ev.type == "llm_usage":
                        tokens += ev.data.get("total_tokens", 0)
                    elif ev.type == "done":
                        steps = ev.data.get("steps", steps)
                        tokens = ev.data.get("tokens", tokens)
            except Exception:
                pass

        result = {
            "instance_id": instance_id,
            "label": label,
            "patch_path": str(run_dir / "patch.diff"),
            "report_path": str(run_dir / "report.md"),
            "trace_path": str(trace_path),
            "duration": duration,
            "error": error,
            "steps": steps,
            "tokens": tokens,
        }

        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        print(f"Finished {instance_id} in {duration}s. Error: {error}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="ANVIL benchmark runner")
    parser.add_argument("--instances", type=Path, default=Path("bench/instances.json"))
    parser.add_argument("--results", type=Path, default=Path("bench/results.jsonl"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", type=str)
    parser.add_argument("--label", type=str, default="default")
    args = parser.parse_args()

    results_file = args.results
    results_file.parent.mkdir(parents=True, exist_ok=True)

    instances = _load_instances(args.instances)
    if args.only:
        instances = [i for i in instances if i["instance_id"] == args.only]

    completed = _load_completed(results_file)
    to_run = [i for i in instances if i["instance_id"] not in completed]

    if args.limit:
        to_run = to_run[: args.limit]

    print(f"Running {len(to_run)} instances ({len(completed)} completed skipped)")

    semaphore = asyncio.Semaphore(args.jobs)
    tasks = [
        _run_instance(inst, args.timeout, args.label, results_file, semaphore)
        for inst in to_run
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
