"""The run's artefacts: the patch (without harness scratch files) and the markdown report."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from anvil.agent.state import RunState

SCRATCH_DIR = ".anvil"
_DEV_NULL = "/dev/null"
_GIT_HEADER = "diff --git "


@dataclass(frozen=True)
class FileChange:
    """One file touched by a patch, with its added/removed line counts."""

    path: str
    added: int
    removed: int


def filter_diff(diff: str) -> str:
    """Return ``diff`` without the sections that touch ``.anvil/`` (the repro scripts).

    Understands ``git diff`` output and plain unified diffs. Text that is not part
    of any file section (a stray status line from a sandbox, say) is dropped, so
    the result is either empty or a patch ``git apply`` can take.
    """
    kept = ["".join(section) for section in _sections(diff) if not _is_scratch(section)]
    patch = "".join(kept)
    return patch if not patch or patch.endswith("\n") else patch + "\n"


def changed_files(patch: str) -> list[FileChange]:
    """List the files a patch changes, in order, with +/- line counts."""
    changes = []
    for section in _sections(patch):
        old, new = _paths(section)
        added = removed = 0
        in_hunk = False
        for line in section:
            if line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                added += 1
            elif in_hunk and line.startswith("-"):
                removed += 1
        changes.append(FileChange(new if new != _DEV_NULL else old, added, removed))
    return changes


def write_outputs(output_dir: Path, patch: str, report: str) -> tuple[Path, Path]:
    """Write ``patch.diff`` and ``report.md`` into ``output_dir`` (created if needed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path, report_path = output_dir / "patch.diff", output_dir / "report.md"
    patch_path.write_text(patch, encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    return patch_path, report_path


def render_report(state: RunState, patch: str, *, steps: int, tokens: int, seconds: float) -> str:
    """Render ``report.md``: issue summary, files changed, confidence, tests, budget, limitations."""
    files = changed_files(patch)
    confidence = state.confidence(bool(files))
    issue = state.issue
    title = f"{issue.title or 'untitled'} ({issue.owner}/{issue.repo}#{issue.number})" if issue else state.issue_url

    out = ["# ANVIL report", "", "## Issue", "", f"- Issue: {title}", f"- URL: {state.issue_url}"]
    out += ["", state.understanding.strip() or "_No issue summary was produced._"]

    out += ["", "## Outcome", ""]
    out.append(f"- Confidence: **{confidence}** ({state.confidence_score(bool(files)):.2f})")
    out.append(f"- Reproduced before patching: {f'yes (`{state.repro_cmd}`)' if state.repro_confirmed else 'no'}")
    out.append(f"- Verified after patching: {'yes' if state.verified else 'no'}")
    out.append(f"- Review: {state.review}")

    out += ["", "## Files changed", ""]
    out += [f"- `{f.path}` (+{f.added} / -{f.removed})" for f in files] or ["- None: no patch was produced."]

    out += ["", "## Tests run", ""]
    out += [f"- {'pass' if c.passed else 'FAIL'}: {c.label}" for c in state.checks] or ["- None recorded."]

    out += ["", "## Summary", "", state.final_summary.strip() or state.localization.strip() or "_None._"]

    out += ["", "## Budget", ""]
    out.append(f"- LLM calls: {steps}")
    out.append(f"- Tokens: {tokens}")
    out.append(f"- Wall clock: {seconds:.1f}s")
    out.append(f"- Patch attempts: {state.patch_attempts}, rollbacks: {state.rollbacks}")

    out += ["", "## Known limitations", ""]
    out += [f"- {note}" for note in state.limitations] or ["- None noted."]
    return "\n".join(out) + "\n"


# ---- diff parsing ---------------------------------------------------------------------------


def _sections(diff: str) -> list[list[str]]:
    """Split a diff into per-file sections (lists of lines); leading non-diff text is discarded."""
    lines = diff.splitlines(keepends=True)
    git_style = any(line.startswith(_GIT_HEADER) for line in lines)
    sections: list[list[str]] = []
    for i, line in enumerate(lines):
        if git_style:
            starts = line.startswith(_GIT_HEADER)
        else:
            starts = line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")
        if starts:
            sections.append([line])
        elif sections:
            sections[-1].append(line)
    return sections


def _paths(section: list[str]) -> tuple[str, str]:
    """(old path, new path) of a section, without the ``a/`` ``b/`` prefixes."""
    first = section[0]
    if first.startswith(_GIT_HEADER):
        old, _, new = first[len(_GIT_HEADER):].strip().partition(" b/")
        return _clean(re.sub(r'^"?a/', "", old)), _clean(new)
    old = section[0][4:].split("\t")[0]
    new = section[1][4:].split("\t")[0]
    return _clean(re.sub(r"^a/", "", old)), _clean(re.sub(r"^b/", "", new))


def _clean(path: str) -> str:
    return path.strip().strip('"')


def _is_scratch(section: list[str]) -> bool:
    return any(p == SCRATCH_DIR or p.startswith(SCRATCH_DIR + "/") for p in _paths(section))
