"""The closing summary the harness writes itself when a phase uses its call cap and the model does not close it.

A phase that has spent its calls is closed by the harness, whatever the model does next (``PhaseStatus.CLOSED``). What the next
phase is told about it cannot come from a model that ignored the request to summarise, so it is composed here from the
phase's own tool calls: what was read, searched, run, tested and edited, and how each ended. Deterministic, no model call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from anvil.agent.text import clip_head
from anvil.events import Phase

if TYPE_CHECKING:
    from anvil.agent.loop import ToolRecord

MAX_NOTE_CHARS = 500
MAX_ITEMS = 6
MAX_ITEM_CHARS = 100
MAX_SUMMARY_CHARS = 1500


def close_summary(phase: Phase, cap: int, records: Iterable[ToolRecord], last_note: str = "") -> str:
    """The harness's own summary of a ``phase`` that used its ``cap`` calls: ``last_note`` (the model's last words) and the calls made."""
    lines = [
        f"[Written by the harness: the {phase.value.upper()} phase used its {cap} calls and was closed without a verdict from the model.]"
    ]
    note = clip_head(" ".join((last_note or "").split()), MAX_NOTE_CHARS)
    if note:
        lines.append(f"The model's last note: {note}")
    evidence = _evidence(list(records))
    lines.extend(evidence or ["No tool call was made in this phase."])
    return clip_head("\n".join(lines), MAX_SUMMARY_CHARS)


def _evidence(records: list[ToolRecord]) -> list[str]:
    reads: list[str] = []
    searches: list[str] = []
    listed: list[str] = []
    commands: list[str] = []
    tests: list[str] = []
    edits: list[str] = []
    scripts: list[str] = []
    diffs = 0
    for record in records:
        args = record.args if isinstance(record.args, dict) else {}
        if record.tool == "read_file":
            reads.append(_read(args))
        elif record.tool == "grep":
            searches.append(f"'{_text(args.get('pattern'))}'")
        elif record.tool == "list_dir":
            listed.append(_text(args.get("path")) or ".")
        elif record.tool == "run_cmd":
            commands.append(f"`{_text(args.get('cmd'))}` ({_outcome(record)})")
        elif record.tool == "run_tests":
            tests.append(f"{_targets(args) or 'all'} ({'passed' if record.ok else 'failed'})")
        elif record.tool == "edit_file":
            edits.append(f"{_text(args.get('path'))} ({'ok' if record.ok else 'failed'})")
        elif record.tool == "write_repro":
            scripts.append(_text(args.get("path")))
        elif record.tool == "git_diff":
            diffs += 1
    lines = []
    for label, items in (
        ("Files read", reads),
        ("Searched for", searches),
        ("Directories listed", listed),
        ("Repro scripts written", scripts),
        ("Commands run", commands),
        ("Tests run", tests),
        ("Edits made", edits),
    ):
        shown = _listing(items)
        if shown:
            lines.append(f"{label}: {shown}")
    if diffs:
        lines.append(f"The diff was looked at {diffs} time(s).")
    return lines


def _read(args: dict) -> str:
    path = _text(args.get("path"))
    start, end = args.get("start"), args.get("end")
    return f"{path} (lines {start}-{end})" if start is not None and end is not None else path


def _outcome(record: ToolRecord) -> str:
    meta = record.meta if isinstance(record.meta, dict) else {}
    if meta.get("timed_out"):
        return "timed out"
    code = meta.get("exit_code")
    if record.ok:
        return "exit 0"
    return f"failed, exit {code}" if code is not None else "failed"


def _targets(args: dict) -> str:
    targets = args.get("targets") or args.get("target") or ""
    joined = " ".join(str(t) for t in targets) if isinstance(targets, (list, tuple)) else str(targets)
    return joined.strip()


def _text(value: object) -> str:
    return clip_head(" ".join(str(value if value is not None else "").split()), MAX_ITEM_CHARS)


def _listing(items: list[str]) -> str:
    unique = list(dict.fromkeys(item for item in items if item))
    shown = ", ".join(unique[:MAX_ITEMS])
    return shown + (f" and {len(unique) - MAX_ITEMS} more" if len(unique) > MAX_ITEMS else "")
