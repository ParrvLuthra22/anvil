#!/usr/bin/env python3
"""ANVIL benchmark analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="ANVIL benchmark analyzer")
    parser.add_argument("--results", type=Path, default=Path("bench/results.jsonl"))
    parser.add_argument("--score", type=Path, default=Path("bench/score.json"))
    args = parser.parse_args()

    results: list[dict] = []
    if args.score.exists():
        try:
            data = json.loads(args.score.read_text(encoding="utf-8"))
            results = data.get("instances", [])
        except Exception:
            pass

    if not results and args.results.exists():
        with open(args.results, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except Exception:
                        pass

    if not results:
        print("No benchmark results found to analyze.")
        return

    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved"))

    categories: dict[str, int] = {}
    for r in results:
        cat = r.get("category", "unscored")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nANVIL Benchmark Summary")
    print("=======================")
    print(f"Total instances: {total}")
    rate = (resolved / total) if total else 0.0
    print(f"Resolved:        {resolved} ({rate:.1%})")

    print("\nCategories Breakdown:")
    for cat, count in sorted(categories.items()):
        pct = (count / total) if total else 0.0
        print(f"  {cat.ljust(20)} {count} ({pct:.1%})")

    resolved_results = [r for r in results if r.get("resolved")]
    if resolved_results:
        avg_tokens = sum(r.get("tokens", 0) for r in resolved_results) / len(resolved_results)
        avg_steps = sum(r.get("steps", 0) for r in resolved_results) / len(resolved_results)
        avg_dur = sum(r.get("duration", 0.0) for r in resolved_results) / len(resolved_results)
        print("\nAverage for resolved instances:")
        print(f"  Duration: {avg_dur:.1f}s")
        print(f"  Tokens:   {avg_tokens:,.0f}")
        print(f"  Steps:    {avg_steps:.1f}")
    print()


if __name__ == "__main__":
    main()
