"""related_tests tool: map changed files to candidate test files."""

from __future__ import annotations

import re
from pathlib import Path

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

class RelatedTestsTool:
    """Map changed source files to candidate test files."""

    name = "related_tests"
    description = (
        "Map changed source files to candidate test files based on naming, "
        "imports, and grep, and return runnable test ids for the repository's test framework."
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of source file paths that were changed.",
            },
        },
        "required": ["paths"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Find related tests for the given paths."""
        paths = args.get("paths", [])
        if not paths or not isinstance(paths, list):
            return ToolResult(ok=False, output="'paths' must be a non-empty list of strings.")

        test_files: set[str] = set()

        for path in paths:
            p = Path(path)
            stem = p.stem

            # 1. Naming heuristics: test_<stem>.*, <stem>_test.*
            res1 = sandbox.exec(f"find . -type f \\( -name '*test*{stem}*' -o -name '*{stem}*test*' \\)")
            if res1.exit_code == 0:
                for line in res1.stdout.splitlines():
                    test_files.add(line.lstrip("./"))

            # 2. Grep / Imports: search for the stem in test directories or test files
            res2 = sandbox.exec(
                f"find . -type f \\( -name '*test*' -o -name '*_test.*' -o -name 'test_.*' -o -path '*/tests/*' -o -path '*/t/*' \\) -exec grep -l '{stem}' {{}} +"
            )
            if res2.exit_code == 0:
                for line in res2.stdout.splitlines():
                    test_files.add(line.lstrip("./"))

        if not test_files:
            return ToolResult(
                ok=True,
                output="No related tests found.",
                meta={"test_ids": []},
            )

        # Convert test file paths into runnable test ids based on language
        runnable: set[str] = set()
        for t in test_files:
            if not t:
                continue
            # Basic exclusions
            if "node_modules" in t or "venv" in t or ".git" in t:
                continue
            
            if t.endswith(".go"):
                # Go test runs on packages
                runnable.add("./" + str(Path(t).parent))
            elif t.endswith(".rs"):
                # Rust cargo test doesn't take file paths natively, but we can pass the module/file stem?
                # Usually it's easier to run everything if Rust, but we can return the path
                runnable.add(t)
            else:
                # Python (pytest), JS (jest/mocha) accept file paths
                runnable.add(t)

        sorted_ids = sorted(list(runnable))[:20]

        return ToolResult(
            ok=True,
            output="Related test ids:\n" + "\n".join(f"  {t}" for t in sorted_ids),
            meta={"test_ids": sorted_ids},
        )
