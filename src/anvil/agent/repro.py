"""The one tool the agent layer contributes itself: writing a repro script under ``.anvil/``.

The tool contract has no way to create a file (``edit_file`` only replaces text in
an existing one), so the REPRODUCE phase needs this. It can only write inside
``.anvil/``, which ``outputs.filter_diff`` keeps out of the final patch.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from anvil.agent.outputs import SCRATCH_DIR
from anvil.sandbox.base import Sandbox
from anvil.tools.base import ToolResult


class WriteReproTool:
    """Create or overwrite a repro script inside the sandbox's ``.anvil/`` directory.

    ``written`` lists, in order, the paths successfully written through this tool.
    """

    name = "write_repro"
    description = (
        "Create or overwrite a repro script under .anvil/ (excluded from the final patch). "
        "`path` is relative to .anvil/, e.g. 'repro.py'. Run the script afterwards with run_cmd."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Script path relative to .anvil/, e.g. 'repro.py'."},
            "content": {"type": "string", "description": "Full contents of the script."},
        },
        "required": ["path", "content"],
    }

    def __init__(self) -> None:
        self.written: list[str] = []

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Write the script; failures come back as ``ok=False`` results, never exceptions."""
        raw, content = args.get("path"), args.get("content")
        if not isinstance(raw, str) or not isinstance(content, str):
            return ToolResult(ok=False, output="Both 'path' and 'content' are required strings.")
        path = scratch_path(raw)
        if path is None:
            return ToolResult(
                ok=False, output=f"Invalid path {raw!r}: it must be a relative file path inside {SCRATCH_DIR}/."
            )
        try:
            sandbox.write_file(path, content)
        except (OSError, ValueError) as exc:
            return ToolResult(ok=False, output=f"Could not write {path}: {exc}")
        if path not in self.written:
            self.written.append(path)
        return ToolResult(ok=True, output=f"Wrote {len(content)} characters to {path}.", meta={"path": path})


def scratch_path(raw: str) -> str | None:
    """Normalise ``raw`` to a repo-relative path inside ``.anvil/``; ``None`` if it would escape."""
    path = PurePosixPath(raw.strip().replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = path.parts if path.parts[:1] == (SCRATCH_DIR,) else (SCRATCH_DIR, *path.parts)
    return "/".join(parts) if len(parts) > 1 else None


# ---- does a failing command's output report the bug the issue describes? ---------------------------------------------

# Failures that say the script or the environment is broken rather than the code under test. They only count as the
# reported bug when the issue itself names that same error.
_ENVIRONMENT_ERRORS = (
    "modulenotfounderror", "importerror", "syntaxerror", "indentationerror", "taberror", "nameerror",
    "command not found", "no such file or directory", "can't open file", "permission denied",
    "cannot find module", "cannot find package", "undefined:",
)
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PATH_LIKE = re.compile(r"\S*[/\\]\S*")  # a file path in a traceback says where the script lives, not what is wrong
_COMMON = frozenset(
    """the and for with that this are was not but you its has have from into when then than such also can should would could
    does doesn don instead returns return expected actual got get error errors exception traceback file files line lines most
    recent call calls last module assert assertion assertionerror fail failed failing failure test tests true false none null
    print python python3 anvil repro script exit code stdout stderr raise raised import def class self use using used bug issue
    output result results value values should must shall may might will just only some any all each every been being were
    """.split()
)


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(_PATH_LIKE.sub(" ", text))} - _COMMON


def reports_the_issue(output: str, issue_text: str) -> bool:
    """Whether a failing command's ``output`` looks like the failure ``issue_text`` describes.

    Two conditions. The output must not be an environment or script error (a typo, a missing import, a wrong path) unless the
    issue names that very error. And it must share at least one distinctive word with the issue (an identifier, a message
    word; common words, tracebacks' boilerplate and file paths do not count).
    """
    out, issue = output.lower(), issue_text.lower()
    if any(term in out and term not in issue for term in _ENVIRONMENT_ERRORS):
        return False
    return bool(_words(output) & _words(issue_text))

