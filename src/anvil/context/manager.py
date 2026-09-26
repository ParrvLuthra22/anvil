"""Conversation history for the agent: everything the model sees goes through here.

The manager keeps the prompt inside a token budget (``max_context_tokens``,
estimated at chars/4) by compacting the history, cheapest measure first:

1. Every tool observation is truncated when it is added (head and tail kept).
2. Observations older than ``keep_steps`` model steps become a one-line summary:
   the tool, its arguments, ok/failed and the first line of output. The
   assistant's actions stay; only the payloads go.
3. Past ``summarize_threshold`` of the budget, the oldest unpinned turns are folded
   into one summary message by an injected ``summarizer`` (one LLM call), or, if
   there is none or it fails, hard-pruned into a digest of the actions taken.
4. If the prompt is still over budget: the repo map is trimmed, recent observations
   are pruned early, and the oldest turns are dropped.

Never pruned: pinned messages (the issue brief, phase summaries the caller pinned),
the latest diff (``set_diff``), the repo map (which may be *trimmed*), the
kickoff of the phase in progress, and the phase goal, which is the system message
of every prompt. If those alone exceed the budget the manager cannot help; the
budget covers messages only, so leave headroom for the tool schemas.

Tool calls and their results are kept or dropped as a unit, so the history always
stays valid OpenAI chat format.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from anvil.context.tokens import CHARS_PER_TOKEN, estimate_message_tokens, estimate_tokens
from anvil.context.truncate import clip_lines, first_line, format_call, truncate_output

logger = logging.getLogger("anvil.context")
logger.addHandler(logging.NullHandler())

DEFAULT_MAX_CONTEXT_TOKENS = 32_000
DEFAULT_TOOL_OUTPUT_CHAR_CAP = 8_000
DEFAULT_KEEP_STEPS = 8
DEFAULT_SUMMARIZE_THRESHOLD = 0.75

# Turns a transcript into a summary; typically one LLM call. May raise.
Summarizer = Callable[[str], str]

_LOW_WATERMARK = 0.5  # after folding, aim for this fraction of the budget
_RECENT_GROUPS_KEPT = 2  # the newest unpinned turns are never folded away
_MIN_FOLD_FRACTION = 0.1  # do not call the summarizer to free less than this share of the budget
_SUMMARY_MAX_CHARS = 2400
_SUMMARY_TOKENS = estimate_tokens("x" * _SUMMARY_MAX_CHARS)
_SUMMARY_COOLDOWN_STEPS = 5  # steps to wait before trying a failed summarizer again
_TRANSCRIPT_ENTRY_CHARS = 700
_TRANSCRIPT_MAX_CHARS = 24_000
_MAP_FLOOR_CHARS = 800

_PRUNED_PREFIX = "[output pruned] "
_SUMMARY_HEADER = "[Summary of earlier work]"
_DIGEST_HEADER = "[Earlier work was dropped to fit the context budget.]"
_DIGEST_ACTIONS_HEADER = "[Earlier work was dropped to fit the context budget. Actions taken, oldest first:]"
_MAP_HEADING = "## Repository overview\n"
_DIFF_HEADING = "[Latest diff of the working tree]\n"

_SUMMARY, _DIFF, _MAP = "summary", "diff", "map"


@dataclass
class _Entry:
    message: dict[str, Any]
    pinned: bool = False
    sticky: bool = False  # kept intact until its phase ends (the phase kickoff)
    tag: str = ""  # "" for an ordinary turn, else _SUMMARY, _DIFF or _MAP
    step: int = 0  # the model step this entry belongs to
    phase_run: int = 0  # which begin_phase call it was added under (0: none)
    label: str = ""  # tool observations: "grep(pattern='x') -> ok"
    line: str = ""  # tool observations: label plus the first line of output
    pruned: bool = False
    source: str = ""  # the repo map's untrimmed text
    tokens: int = 0

    def set_content(self, content: str) -> None:
        self.message["content"] = content
        self.tokens = estimate_message_tokens(self.message)


class ContextManager:
    """Ordered message history plus the compaction that keeps the prompt within budget."""

    def __init__(
        self,
        *,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        tool_output_char_cap: int = DEFAULT_TOOL_OUTPUT_CHAR_CAP,
        keep_steps: int = DEFAULT_KEEP_STEPS,
        summarize_threshold: float = DEFAULT_SUMMARIZE_THRESHOLD,
        summarizer: Summarizer | None = None,
    ) -> None:
        self._max = max_context_tokens
        self._threshold = int(max_context_tokens * summarize_threshold)
        self._cap = tool_output_char_cap
        self._keep_steps = keep_steps
        self._summarizer = summarizer
        self._entries: list[_Entry] = []
        self._calls: dict[str, tuple[str, Any, int]] = {}  # tool_call_id -> (tool, args, step)
        self._step = 0
        self._phase_runs = 0
        self._open_run = 0
        self._goal_tokens = 0
        self._no_summary_before = 0

    def __len__(self) -> int:
        return len(self._entries)

    # ---- adding to the history -------------------------------------------------------------

    def add_message(
        self, role: str, content: str, pinned: bool = False, *, ok: bool | None = None, **fields: Any
    ) -> None:
        """Append a message in OpenAI chat format.

        ``pinned`` messages are never pruned (the issue, phase summaries). ``fields``
        carries the extra keys tool calling needs: ``tool_calls`` on an assistant
        message, ``tool_call_id`` on a ``tool`` message. For a ``tool`` message,
        ``ok`` records whether the tool succeeded (used only in the one-line summary
        it turns into once it is old); its content is truncated to the output cap.
        """
        step = self._step
        label = line = ""
        if role == "assistant":
            self._step = step = self._step + 1
            self._register_calls(fields.get("tool_calls"), step)
        elif role == "tool":
            name, args, step = self._calls.get(
                str(fields.get("tool_call_id", "")), (str(fields.get("name") or "tool"), None, step)
            )
            content = truncate_output(content, self._cap)
            label, line = _describe_observation(name, args, ok, content)
        self._append(
            _Entry({"role": role, "content": content, **fields}, pinned=pinned, step=step, label=label, line=line)
        )

    def set_diff(self, diff: str) -> None:
        """Pin ``diff`` as the latest diff of the working tree, replacing the previous one.

        An empty diff removes it. A diff longer than the tool output cap is truncated.
        """
        self._entries = [e for e in self._entries if e.tag != _DIFF]
        if diff.strip():
            content = _DIFF_HEADING + truncate_output(diff.strip("\n"), self._cap)
            self._append(_Entry({"role": "user", "content": content}, pinned=True, tag=_DIFF, step=self._step))

    def set_repo_map(self, text: str) -> None:
        """Pin the repository map. It is never dropped, but is trimmed if the prompt runs out of room."""
        self._entries = [e for e in self._entries if e.tag != _MAP]
        if text.strip():
            entry = _Entry({"role": "user", "content": ""}, pinned=True, tag=_MAP, step=self._step, source=text.strip())
            self._render_map(entry, len(entry.source))
            self._entries.append(entry)

    # ---- phases ----------------------------------------------------------------------------

    def begin_phase(self, kickoff: str) -> None:
        """Start a phase with its ``kickoff`` message, which is kept intact until the phase ends.

        A phase left open by an earlier ``begin_phase`` is not compressed; its turns
        just become ordinary history.
        """
        self._phase_runs += 1
        self._open_run = self._phase_runs
        for entry in self._entries:
            entry.sticky = False
        self._append(_Entry({"role": "user", "content": kickoff}, sticky=True, step=self._step, phase_run=self._open_run))

    def end_phase(self, summary: str, *, pinned: bool = False) -> None:
        """Compress the phase in progress into ``summary``.

        Its turns (kickoff, actions, observations) are replaced by one message; pinned
        messages and the folded summaries are untouched. ``pinned`` keeps the summary
        through every later compaction. Without an open phase this only appends the summary.
        """
        run = self._open_run
        if run:
            self._entries = [e for e in self._entries if e.phase_run != run or e.pinned or e.tag]
        self._open_run = 0
        self._append(_Entry({"role": "user", "content": summary}, pinned=pinned, step=self._step))

    # ---- building the prompt ---------------------------------------------------------------

    def build_messages(self, phase_goal: str) -> list[dict[str, Any]]:
        """Return the prompt for the next model call, compacted to fit the budget.

        ``phase_goal`` is the current phase's instructions; it leads the prompt as the
        system message. The returned dicts are copies, so callers (and the LLM client)
        can never corrupt the stored history.
        """
        system = {"role": "system", "content": phase_goal}
        self._goal_tokens = estimate_message_tokens(system)
        self._compact()
        return [system, *(copy.deepcopy(e.message) for e in self._entries)]

    def token_estimate(self) -> int:
        """Estimated tokens of the prompt ``build_messages`` would return for the last phase goal it was given."""
        return self._goal_tokens + sum(e.tokens for e in self._entries)

    # ---- compaction ------------------------------------------------------------------------

    def _compact(self) -> None:
        self._prune_stale_observations()
        if self.token_estimate() > self._threshold:
            self._condense()
        if self.token_estimate() > self._max:
            self._squeeze()
        if self.token_estimate() > self._max:
            logger.warning("context still over budget after compaction: ~%d > %d", self.token_estimate(), self._max)

    def _prune_stale_observations(self) -> None:
        for entry in self._entries:
            if entry.message["role"] == "tool" and self._step - entry.step >= self._keep_steps:
                self._prune(entry)

    def _prune(self, entry: _Entry) -> None:
        if entry.pruned or entry.pinned or entry.message["role"] != "tool":
            return
        entry.pruned = True
        entry.set_content(_PRUNED_PREFIX + entry.line)

    def _condense(self) -> None:
        """Past the threshold: fold the oldest unpinned turns into one summary (or digest)."""
        candidates = self._removable_groups()[:-_RECENT_GROUPS_KEPT]
        if not any(group[0].tag != _SUMMARY for group in candidates):
            return
        if sum(e.tokens for group in candidates for e in group) < self._max * _MIN_FOLD_FRACTION:
            return
        target = int(self._max * _LOW_WATERMARK)
        chosen: list[list[_Entry]] = []
        estimate = self.token_estimate()
        for group in candidates:
            chosen.append(group)
            estimate -= sum(e.tokens for e in group)
            if estimate + _SUMMARY_TOKENS <= target:
                break
        self._fold([e for group in chosen for e in group], use_summarizer=True)

    def _squeeze(self) -> None:
        """Still over the budget: trim the map, prune recent observations early, then drop the oldest turns."""
        self._shrink_map()
        for entry in self._entries:
            if self.token_estimate() <= self._max:
                return
            self._prune(entry)
        candidates = self._removable_groups()[:-1]
        chosen: list[list[_Entry]] = []
        estimate = self.token_estimate()
        for group in candidates:
            if estimate <= self._max:
                break
            chosen.append(group)
            estimate -= sum(e.tokens for e in group)
        if chosen:
            self._fold([e for group in chosen for e in group], use_summarizer=False)

    def _shrink_map(self) -> None:
        entry = next((e for e in self._entries if e.tag == _MAP), None)
        if entry is None:
            return
        excess_chars = (self.token_estimate() - self._max) * CHARS_PER_TOKEN
        body = len(entry.message["content"]) - len(_MAP_HEADING)
        self._render_map(entry, max(_MAP_FLOOR_CHARS, body - excess_chars))

    def _render_map(self, entry: _Entry, limit: int) -> None:
        entry.set_content(_MAP_HEADING + clip_lines(entry.source, limit))

    def _removable_groups(self) -> list[list[_Entry]]:
        """History as atomic units (a message, or an assistant message with its tool results), oldest first,
        without the units that are pinned or belong to the phase in progress."""
        groups: list[list[_Entry]] = []
        owner: dict[str, list[_Entry]] = {}
        for entry in self._entries:
            message = entry.message
            call_id = str(message.get("tool_call_id", ""))
            if message["role"] == "tool" and call_id in owner:
                owner[call_id].append(entry)
                continue
            group = [entry]
            groups.append(group)
            for call in message.get("tool_calls") or []:
                owner[str(call.get("id", ""))] = group
        return [g for g in groups if not any(e.pinned or e.sticky for e in g)]

    def _fold(self, removed: list[_Entry], *, use_summarizer: bool) -> None:
        """Replace ``removed`` (oldest unpinned entries) with one summary, or a digest if it cannot be summarised."""
        text = self._summarise(removed) if use_summarizer else None
        content = f"{_SUMMARY_HEADER}\n{text}" if text else _digest(removed)
        gone = {id(e) for e in removed}
        position = next(i for i, e in enumerate(self._entries) if id(e) in gone)
        self._entries = [e for e in self._entries if id(e) not in gone]
        summary = _Entry({"role": "user", "content": content}, tag=_SUMMARY, step=removed[-1].step)
        summary.tokens = estimate_message_tokens(summary.message)
        self._entries.insert(position, summary)

    def _summarise(self, removed: list[_Entry]) -> str | None:
        """One summariser call over ``removed``; ``None`` if there is no summariser, it is cooling down, or it fails."""
        if self._summarizer is None or self._step < self._no_summary_before:
            return None
        try:
            text = self._summarizer(truncate_output(_transcript(removed), _TRANSCRIPT_MAX_CHARS))
        except Exception:  # noqa: BLE001 - any failure falls back to pruning
            logger.warning("context summariser failed; pruning instead", exc_info=True)
            text = None
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            self._no_summary_before = self._step + _SUMMARY_COOLDOWN_STEPS
            return None
        return text if len(text) <= _SUMMARY_MAX_CHARS else text[:_SUMMARY_MAX_CHARS] + " …"

    def _append(self, entry: _Entry) -> None:
        entry.phase_run = entry.phase_run or self._open_run
        entry.tokens = estimate_message_tokens(entry.message)
        self._entries.append(entry)

    def _register_calls(self, calls: Any, step: int) -> None:
        for call in calls or []:
            function = call.get("function") or {}
            name, args = str(function.get("name", "")), _decode(function.get("arguments"))
            self._calls[str(call.get("id", ""))] = (name, args, step)


def _decode(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _describe_observation(name: str, args: Any, ok: bool | None, content: str) -> tuple[str, str]:
    """(label, line) for a tool result: the call and its status, and that plus the first line of output."""
    status = {True: "ok", False: "failed", None: ""}[ok]
    call = format_call(name, args)
    label = f"{call} -> {status}" if status else call
    return label, f"{label}: {first_line(content) or '(no output)'}"


def _transcript(entries: list[_Entry]) -> str:
    """Plain-text rendering of ``entries`` for the summariser, each clipped to a sane size."""
    parts = []
    for entry in entries:
        message = entry.message
        role, content = message["role"], str(message.get("content") or "")
        if role == "tool":
            parts.append(f"[tool] {entry.line}" if entry.pruned else f"[tool] {entry.label}\n{_clip(content)}")
            continue
        text = _clip(content, _SUMMARY_MAX_CHARS if entry.tag == _SUMMARY else _TRANSCRIPT_ENTRY_CHARS)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            text += f"\n-> {format_call(str(function.get('name', '')), _decode(function.get('arguments')))}"
        parts.append(f"[{role}] {text}".rstrip())
    return "\n\n".join(parts)


def _clip(text: str, limit: int = _TRANSCRIPT_ENTRY_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def _digest(entries: list[_Entry]) -> str:
    """Deterministic stand-in for a summary: the tool calls made, oldest first, newest kept if it must be cut."""
    lines: list[str] = []
    for entry in entries:
        if entry.tag == _SUMMARY:
            lines += entry.message["content"].splitlines()[1:]
        elif entry.message["role"] == "tool":
            lines.append(f"- {entry.line}")
    if not lines:
        return _DIGEST_HEADER
    kept: list[str] = []
    used = len(_DIGEST_ACTIONS_HEADER)
    for line in reversed(lines):
        if used + len(line) + 1 > _SUMMARY_MAX_CHARS:
            kept.append("- (older actions omitted)")
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join([_DIGEST_ACTIONS_HEADER, *reversed(kept)])
