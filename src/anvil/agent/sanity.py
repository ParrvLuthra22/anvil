"""Checks on the patch before it is delivered (``features.patch_sanity``).

A patch that is empty, that does not apply to the original code, or that edits the tests is worth
nothing to whoever receives it, and the model that produced it has no way to notice. So before
FINALIZE the harness looks at the patch itself:

* it is taken from the diff without the harness's own files (``.anvil/``, ``.anvil_venv/``,
  ``*.egg-info``, bytecode, binary sections: see ``filter_diff``);
* changes to test files are taken out of it unless the issue is about tests;
* it must not be empty;
* ``git apply --check`` must accept it in a scratch worktree at the base commit, that is, the very
  state a reviewer or an evaluator will apply it to.

``inspect_patch`` does the looking and never raises; the orchestrator decides what a problem costs.
"""

from __future__ import annotations

import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from anvil.agent.outputs import split_patch
from anvil.sandbox.base import Sandbox

_APPLY_TIMEOUT = 60
_STDERR_CHARS = 600
_TEST_DIRS = frozenset({"tests", "test", "__tests__", "spec", "specs"})
_TEST_FILE = re.compile(
    r"^(?:test_.*\.py|.*_tests?\.py|conftest\.py|.*\.(?:test|spec)\.[cm]?[jt]sx?|.*_test\.go|.*Tests?\.(?:java|kt))$"
)
_TITLE_ABOUT_TESTS = re.compile(r"\b(?:tests?|testing|pytest|unittest|coverage|flaky)\b", re.IGNORECASE)
_BODY_ASKS_FOR_TESTS = re.compile(
    r"\b(?:add|adding|write|writing|create|creating|missing|improve|more)\s+(?:(?:a|an|some|more|new|unit|regression)\s+)*tests?\b",
    re.IGNORECASE,
)


def is_test_path(path: str) -> bool:
    """Whether ``path`` (repository-relative, ``/``-separated) is a test file or lives in a test directory."""
    pure = PurePosixPath(path)
    return any(part in _TEST_DIRS for part in pure.parts[:-1]) or bool(_TEST_FILE.match(pure.name))


def issue_is_about_tests(title: str, body: str = "") -> bool:
    """Whether the issue asks for work on the tests themselves, so that editing them is the point.

    True when the title mentions tests, testing, coverage or flakiness, or the text asks to add, write or improve
    tests. A heuristic, deliberately permissive: a bug report that merely mentions a test does not count, and being
    wrong towards "about tests" only means test edits are left in the patch.
    """
    return bool(_TITLE_ABOUT_TESTS.search(title or "") or _BODY_ASKS_FOR_TESTS.search(body or ""))


@dataclass(frozen=True)
class PatchCheck:
    """What the harness found out about a patch."""

    patch: str
    """The patch to deliver: the given one without the test files that were taken out."""
    problem: str = ""
    """Why it cannot be delivered (empty patch, does not apply); empty when it can."""
    removed_tests: tuple[str, ...] = ()
    """Test files whose changes were taken out of the patch."""
    apply_checked: bool = False
    """Whether ``git apply --check`` actually ran (it cannot outside a git checkout)."""
    skipped: str = ""
    """Why ``git apply --check`` did not run, if it did not."""

    @property
    def ok(self) -> bool:
        return not self.problem


def inspect_patch(sandbox: Sandbox, patch: str, *, allow_tests: bool) -> PatchCheck:
    """Look at ``patch`` (already free of the harness's files) as described in the module docstring."""
    removed: list[str] = []
    if not allow_tests:
        kept: list[str] = []
        for path, text in split_patch(patch):
            if is_test_path(path):
                removed.append(path)
            else:
                kept.append(text)
        patch = "".join(kept)
    if not patch.strip():
        if removed:
            problem = (
                f"The patch only changes test files ({', '.join(removed)}), which must not be edited for this issue, "
                "so nothing is left to deliver. Fix the source instead."
            )
        else:
            problem = "The patch is empty: no source file was changed."
        return PatchCheck("", problem, tuple(removed))
    applies, detail = _applies_to_base(sandbox, patch)
    if applies is False:
        return PatchCheck(patch, f"The patch does not apply to the original code (git apply --check): {detail}", tuple(removed), True)
    return PatchCheck(patch, "", tuple(removed), applies is True, "" if applies is not None else detail)


def _applies_to_base(sandbox: Sandbox, patch: str) -> tuple[bool | None, str]:
    """``git apply --check`` of ``patch`` in a scratch worktree at the base commit.

    Returns ``(True, "")``, ``(False, why)`` or ``(None, why it could not be tried)``. The scratch worktree
    lives outside the repository and is always removed again.
    """
    base = _exec(sandbox, "git rev-parse HEAD")
    if base is None or base[0] != 0 or not base[1].strip():
        return None, "the working directory is not a git checkout"
    scratch = Path(tempfile.mkdtemp(prefix="anvil-sanity-"))
    worktree = scratch / "base"
    added = False
    try:
        made = _exec(sandbox, f"git worktree add --detach {shlex.quote(str(worktree))} {shlex.quote(base[1].strip())}")
        if made is None or made[0] != 0:
            return None, f"could not create a scratch worktree at the base commit: {_tail(made[2] if made else '')}"
        added = True
        sandbox.write_file(".anvil/sanity.diff", patch)
        patch_file = Path(sandbox.root) / ".anvil" / "sanity.diff"
        checked = _exec(
            sandbox, f"git -C {shlex.quote(str(worktree))} apply --check --whitespace=nowarn {shlex.quote(str(patch_file))}"
        )
        if checked is None:
            return None, "git apply --check could not be run"
        if checked[0] != 0:
            return False, _tail(checked[2] or checked[1]) or f"exit code {checked[0]}"
        return True, ""
    except Exception as exc:  # noqa: BLE001 - a sanity check must never stop the run
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if added:
            _exec(sandbox, f"git worktree remove --force {shlex.quote(str(worktree))}")
        _exec(sandbox, "git worktree prune")
        shutil.rmtree(scratch, ignore_errors=True)


def _exec(sandbox: Sandbox, command: str) -> tuple[int, str, str] | None:
    """(exit code, stdout, stderr) of ``command`` in the sandbox, or ``None`` if it timed out or could not run."""
    try:
        result = sandbox.exec(command, timeout=_APPLY_TIMEOUT)
    except Exception:  # noqa: BLE001
        return None
    if result.timed_out:
        return None
    return result.exit_code, result.stdout, result.stderr


def _tail(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= _STDERR_CHARS else text[:_STDERR_CHARS] + "..."
