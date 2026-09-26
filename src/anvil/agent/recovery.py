"""How the harness reacts when a run goes wrong: one fixed response per class of failure.

Failures fall into the classes of ``ErrorClass``. Each has one response, and every response
is announced as an ``error`` event whose ``kind`` is the class value, so it shows in the TUI
and the trace:

=============  ==========================================================================
tool           a tool failed: the model is told to adjust its arguments or use another tool
edit           ``edit_file`` failed: the closest lines in the file plus "re-read before editing"
test_failure   ``run_tests`` failed: the key failing lines are put first, with guidance
timeout        a command timed out: the model is told to run something narrower
invalid_call   unknown tool or bad arguments: the call is not run and the schema is returned
no_tool_call   a reply without a tool call: nudged once, then the phase ends
loop           the same call 3 times in a row, or A/B/A/B: not run, corrective message,
               and the phase ends at the third strike
llm            the client's retries are exhausted: the run finalises, with advice on the cause
budget         steps, tokens or wall clock used up: the run finalises with reduced confidence
=============  ==========================================================================

Everything here is policy and text; ``PhaseRunner`` and ``Orchestrator`` call in. ``Checkpointer``
covers the last piece, rolling the working tree back through the ``Sandbox`` interface.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from anvil.agent.emitter import Emitter
from anvil.agent.outputs import changed_files, filter_diff
from anvil.agent.text import clip_head
from anvil.context.truncate import format_call
from anvil.events import Phase
from anvil.llm.errors import LLMConfigError, LLMError
from anvil.sandbox.base import Sandbox

LOOP_REPEATS = 3  # identical calls in a row that make a loop
LOOP_STRIKE_LIMIT = 3  # strikes after which the phase is ended
MAX_SILENT_REPLIES = 1  # nudges for a reply without a tool call, before the phase is ended

NUDGE = (
    "Reply with a tool call. Call phase_done(summary) if this phase's objective is met, "
    "or give_up(reason) if it cannot be met."
)

_CLOSEST_LINES = 3
_CLOSEST_CUTOFF = 0.5
_MAX_SCANNED_LINES = 10_000
_MAX_BLOCK_LINES = 8
_LINE_CHARS = 200
_DIGEST_LINES = 10


class ErrorClass(str, Enum):
    """The kinds of failure the harness recovers from; the value is the ``error`` event's ``kind``."""

    TOOL_ERROR = "tool"
    EDIT_FAILED = "edit"
    TEST_FAILURE = "test_failure"
    TIMEOUT = "timeout"
    INVALID_CALL = "invalid_call"
    NO_TOOL_CALL = "no_tool_call"
    LOOP = "loop"
    LLM_ERROR = "llm"
    BUDGET = "budget"


# ---- loop detection -------------------------------------------------------------------------


def call_signature(tool: str, args: Mapping[str, Any]) -> str:
    """A stable identity for a call: the tool and its arguments, whatever order the keys came in."""
    return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"


class LoopDetector:
    """Spots a model that is stuck: the same call ``repeats`` times in a row, or A/B/A/B.

    Every call after the pattern first appears is another strike, so a model that ignores
    the corrective message reaches ``strike_limit`` after a couple more calls.
    """

    def __init__(self, *, repeats: int = LOOP_REPEATS, strike_limit: int = LOOP_STRIKE_LIMIT) -> None:
        self._repeats = repeats
        self._strike_limit = strike_limit
        self._recent: deque[str] = deque(maxlen=4)
        self._streak = 0
        self.strikes = 0

    def observe(self, tool: str, args: Mapping[str, Any]) -> str | None:
        """Record a call. Returns a description of the loop (and adds a strike) if this call completes one."""
        sig = call_signature(tool, args)
        self._streak = self._streak + 1 if self._recent and self._recent[-1] == sig else 1
        self._recent.append(sig)
        pattern = self._pattern()
        if pattern:
            self.strikes += 1
        return pattern

    @property
    def exhausted(self) -> bool:
        """True once the strike limit is reached."""
        return self.strikes >= self._strike_limit

    def _pattern(self) -> str | None:
        recent = list(self._recent)
        if self._streak >= self._repeats:
            return f"the same call {self._streak} times in a row"
        if len(recent) >= 4 and recent[-1] == recent[-3] and recent[-2] == recent[-4] and recent[-1] != recent[-2]:
            return "alternating between the same two calls"
        return None


_LOOP_HINTS: dict[Phase, str] = {
    Phase.LOCALIZE: "search for a different identifier, read a different file or line range, or list another directory; "
    "if you already know enough, call phase_done",
    Phase.REPRODUCE: "change the repro script or its command instead of running it unchanged; "
    "if the bug cannot be reproduced, call give_up",
    Phase.PATCH: "re-read the code, form a different hypothesis and edit something else; "
    "if no fix is possible, call give_up",
    Phase.VERIFY: "the results cannot change unless the code does; "
    "diagnose the failure and call give_up(reason) with what you found",
    Phase.REVIEW: "you already have what you need: call phase_done to approve, or give_up(reason) to request changes",
}
_DEFAULT_LOOP_HINT = "use different arguments or a different tool, or end the phase with phase_done or give_up"


# ---- arguments and schemas ------------------------------------------------------------------

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_arguments(parameters: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
    """Problems with ``args`` against a JSON-Schema ``parameters`` object; empty when they are fine.

    Checks required keys and basic types. Extra arguments are tolerated: the tool ignores them.
    """
    problems = [f"missing required argument '{key}'" for key in parameters.get("required") or [] if key not in args]
    properties = parameters.get("properties") or {}
    for key, value in args.items():
        declared = (properties.get(key) or {}).get("type")
        check = _TYPE_CHECKS.get(declared) if isinstance(declared, str) else None
        if check is not None and not check(value):
            problems.append(f"argument '{key}' must be of type {declared}, got {type(value).__name__}")
    return problems


def describe_schema(name: str, parameters: Mapping[str, Any]) -> str:
    """The tool's argument schema on one line, for showing to the model."""
    return f"{name} takes (JSON Schema): {json.dumps(parameters, separators=(',', ':'), default=str)}"


def signature(name: str, parameters: Mapping[str, Any]) -> str:
    """A short call signature such as ``read_file(path, start?, end?)``."""
    required = set(parameters.get("required") or [])
    names = [key if key in required else f"{key}?" for key in (parameters.get("properties") or {})]
    return f"{name}({', '.join(names)})"


def invalid_arguments_reply(name: str, detail: str, parameters: Mapping[str, Any] | None) -> str:
    """The reply to a call whose arguments were unusable: what was wrong, and the schema to follow."""
    reply = f"Invalid arguments for '{name}': {detail}."
    if parameters is not None:
        reply += f"\n{describe_schema(name, parameters)}"
    return reply + "\nCall it again with valid arguments, or use a different tool."


def unknown_tool_reply(
    name: str, phase: Phase, offered: Mapping[str, Mapping[str, Any]], *, exists: bool = False
) -> str:
    """The reply to a call of a tool that does not exist (or, if ``exists``, is not allowed in ``phase``)."""
    reply = f"Tool '{name}' is not available in the {phase.value} phase."
    close = [] if exists else difflib.get_close_matches(name, list(offered), n=1, cutoff=0.5)
    if close:
        reply += f" Did you mean '{close[0]}'?\n{describe_schema(close[0], offered[close[0]])}"
    listing = ", ".join(signature(tool, parameters) for tool, parameters in offered.items())
    return f"{reply}\nTools you can call here: {listing}."


# ---- reviewing tool results -----------------------------------------------------------------

_EDIT_REMINDER = (
    "Re-read the file with read_file before editing again, and copy `old` verbatim from its output "
    "(indentation and blank lines included). Do not retry the same edit."
)
_TIMEOUT_ADVICE = (
    "The command timed out. Do not run it again as it is: run something narrower (a single test file or test id, "
    "a smaller input), and avoid commands that wait for input or never exit."
)
_TEST_ADVICE = (
    "Fix the cause of these failures in the code under test. Never weaken, skip or delete a test to make it pass. "
    "A failure that plainly exists without your patch is pre-existing: say so in your summary instead."
)
_TOOL_ADVICE = (
    "The tool call failed. Check the arguments against the tool's schema or use a different tool; "
    "do not repeat the same call."
)
_LOCATE_ADVICE = "Find the correct path first with list_dir or grep."

_FAILURE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^\s*(FAILED|ERROR)\b",  # pytest summary lines
        r"^\s*E\s{2,}\S",  # pytest assertion detail
        r"^\s*(FAIL|ERROR):\s",  # unittest
        r"^\s*FAIL\s+\S",  # jest, go packages
        r"^\s*--- FAIL:",  # go test
        r"panicked at|^\s*panic:",  # rust, go
        r"\btest\b.*\.\.\. FAILED",  # cargo
        r"^\s*[●✕✗×]\s",  # jest, mocha
        r"^\s*not ok\b",  # TAP
        r"^\[ERROR\]\s.*(FAIL|Tests run:|expected)",  # maven
        r"\b(AssertionError|AssertionFailedError)\b",
    )
)


def timed_out(output: str, meta: Mapping[str, Any]) -> bool:
    """Whether a tool result reports that its command timed out."""
    return bool(meta.get("timed_out")) or "[TIMED OUT" in output


def failure_digest(output: str, limit: int = _DIGEST_LINES) -> list[str]:
    """The lines of a test run's output that name what failed (across the usual test runners), in order."""
    seen: set[str] = set()
    lines: list[str] = []
    for line in output.splitlines():
        text = line.strip()
        if text not in seen and any(pattern.search(text) for pattern in _FAILURE_PATTERNS):
            seen.add(text)
            lines.append(clip_head(text, _LINE_CHARS))
            if len(lines) == limit:
                break
    return lines


def closest_lines(content: str, old: str, limit: int = _CLOSEST_LINES) -> list[tuple[int, str]]:
    """The part of ``content`` most like ``old``, as (line number, text) pairs with their real indentation.

    A multi-line ``old`` is matched as a block: the run of file lines whose text (ignoring leading and
    trailing whitespace) agrees with the most lines of ``old``, since indentation is the usual reason an
    edit does not match. Otherwise, or if no block agrees on at least two lines, the ``limit`` file lines
    most similar to the first line of ``old`` are returned, best first.
    """
    wanted = [line.strip() for line in old.splitlines() if line.strip()]
    if not wanted:
        return []
    lines = content.splitlines()[:_MAX_SCANNED_LINES]
    if len(wanted) > 1:
        block = _best_block(lines, wanted)
        if block:
            return block
    return _similar_lines(lines, wanted[0], limit)


def _best_block(lines: list[str], wanted: list[str]) -> list[tuple[int, str]]:
    """The run of non-blank file lines agreeing with the most of ``wanted``; empty if fewer than two agree."""
    present = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    stripped = [line.strip() for _, line in present]
    best_score, best_start = 0, 0
    for start in range(len(present)):
        score = sum(1 for a, b in zip(wanted, stripped[start : start + len(wanted)]) if a == b)
        if score > best_score:
            best_score, best_start = score, start
    if best_score < 2:
        return []
    return present[best_start : best_start + min(len(wanted), _MAX_BLOCK_LINES)]


def _similar_lines(lines: list[str], query: str, limit: int) -> list[tuple[int, str]]:
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(query)
    scored: list[tuple[float, int, str]] = []
    for number, line in enumerate(lines, 1):
        text = line.strip()
        if not text:
            continue
        matcher.set_seq1(text)
        if matcher.real_quick_ratio() < _CLOSEST_CUTOFF or matcher.quick_ratio() < _CLOSEST_CUTOFF:
            continue
        ratio = matcher.ratio()
        if ratio >= _CLOSEST_CUTOFF:
            scored.append((ratio, number, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(number, line) for _, number, line in scored[:limit]]


def edit_failure_feedback(args: Mapping[str, Any], output: str, sandbox: Sandbox) -> str:
    """What to append to a failed ``edit_file`` result: the closest lines in the file, and a re-read reminder.

    The closest lines are looked up here unless the tool already listed them; they carry line
    numbers and their original indentation. Never raises: an unreadable file just means no lines.
    """
    path, old = str(args.get("path") or ""), str(args.get("old") or "")
    parts: list[str] = []
    if "file not found" in output.lower():
        parts.append(_LOCATE_ADVICE)
    elif path and old and "Closest existing lines" not in output:
        try:
            content = sandbox.read_file(path)
        except Exception:  # noqa: BLE001 - the tool already said why; this is only extra help
            content = ""
        matches = closest_lines(content, old)
        if matches:
            shown = "\n".join(f"  {number}: {clip_head(line, _LINE_CHARS)}" for number, line in matches)
            parts.append(f"Closest matching lines in {path}:\n{shown}")
            if _differs_only_in_whitespace(old, matches):
                parts.append("The same text exists, but its whitespace differs: copy the indentation exactly.")
    parts.append(_EDIT_REMINDER)
    return "\n".join(parts)


def _differs_only_in_whitespace(old: str, matches: list[tuple[int, str]]) -> bool:
    """Whether some matched line has the text of a line of ``old`` but not its exact whitespace."""
    given = old.splitlines()
    texts = {line.strip() for line in given if line.strip()}
    return any(line.strip() in texts and line not in given for _, line in matches)


@dataclass(frozen=True)
class Review:
    """A tool result after recovery has looked at it: the text for the model, and what (if anything) went wrong."""

    output: str
    error_class: ErrorClass | None = None
    detail: str = ""


def review_result(
    tool: str, args: Mapping[str, Any], ok: bool, output: str, meta: Mapping[str, Any], sandbox: Sandbox
) -> Review:
    """Classify a finished tool call and add the response for its class to what the model reads.

    A command that exits non-zero is information, not an error (a repro is *meant* to fail), so
    ``run_cmd`` is only reviewed when it times out.
    """
    if timed_out(output, meta):
        detail = f"{tool} timed out; told the model to run something narrower"
        return Review(f"{output}\n\n{_TIMEOUT_ADVICE}", ErrorClass.TIMEOUT, detail)
    if ok:
        return Review(output)
    if tool == "edit_file":
        path = args.get("path") or "?"
        feedback = edit_failure_feedback(args, output, sandbox)
        detail = f"edit_file failed on {path}; sent the closest lines and a re-read reminder"
        return Review(f"{output}\n\n{feedback}", ErrorClass.EDIT_FAILED, detail)
    if tool == "run_tests":
        lines = failure_digest(output)
        digest = "Key failure lines:\n" + "\n".join(f"  {line}" for line in lines) + "\n" if lines else ""
        return Review(
            f"[test run failed]\n{digest}{_TEST_ADVICE}\n\n{output}",
            ErrorClass.TEST_FAILURE,
            f"run_tests failed ({len(lines)} key lines extracted); told the model to fix the cause, not the tests",
        )
    if tool == "run_cmd":
        return Review(output)
    advice = _LOCATE_ADVICE + " " if "not found" in output.lower() else ""
    detail = f"{tool} failed; told the model to adjust or use another tool"
    return Review(f"{output}\n\n{advice}{_TOOL_ADVICE}", ErrorClass.TOOL_ERROR, detail)


# ---- per-phase state ------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopVerdict:
    """A detected loop: the reply to give instead of running the call, and whether the phase must end."""

    reply: str
    reason: str
    exhausted: bool


class PhaseGuard:
    """The recovery state of one phase run: the loop detector and the count of tool-less replies.

    ``tools`` maps every tool offered in the phase (registry and control tools) to its JSON-Schema
    parameters. Each response is announced with an ``error`` event.
    """

    def __init__(self, emitter: Emitter, phase: Phase, tools: Mapping[str, Mapping[str, Any]]) -> None:
        self._emitter = emitter
        self._phase = phase
        self._tools = tools
        self._loops = LoopDetector()
        self._silent = 0

    def announce(self, error_class: ErrorClass, message: str) -> None:
        """Publish a recovery action as an ``error`` event."""
        self._emitter.error(error_class.value, message)

    def unknown_tool(self, name: str, *, exists: bool = False) -> str:
        """Reply for a tool that is not offered in this phase (``exists``: it is a real tool, just not allowed here)."""
        self.announce(ErrorClass.INVALID_CALL, f"{name} is not available in the {self._phase.value} phase; sent the tool list")
        return unknown_tool_reply(name, self._phase, self._tools, exists=exists)

    def invalid_arguments(self, name: str, detail: str) -> str:
        """Reply for a call whose arguments could not be used; carries the tool's schema."""
        self.announce(ErrorClass.INVALID_CALL, f"{name}: {detail}; sent the schema")
        return invalid_arguments_reply(name, detail, self._tools.get(name))

    def check_arguments(self, name: str, args: Mapping[str, Any]) -> str | None:
        """The reply to give if ``args`` do not fit the tool's schema, else ``None``."""
        parameters = self._tools.get(name)
        problems = validate_arguments(parameters, args) if parameters is not None else []
        return self.invalid_arguments(name, "; ".join(problems)) if problems else None

    def repeated(self, name: str, args: Mapping[str, Any]) -> LoopVerdict | None:
        """Note a call; if it makes a loop, the corrective reply to give instead of running it."""
        pattern = self._loops.observe(name, args)
        if pattern is None:
            return None
        strikes = self._loops.strikes
        hint = _LOOP_HINTS.get(self._phase, _DEFAULT_LOOP_HINT)
        reply = (
            f"Not run: you are repeating yourself ({pattern}) and the result will not change. "
            f"Try a different approach: {hint}. (Strike {strikes} of {LOOP_STRIKE_LIMIT}; "
            f"at {LOOP_STRIKE_LIMIT} the phase is ended.)"
        )
        call = format_call(name, args)
        ended = self._loops.exhausted
        action = "ending the phase" if ended else "told the model to change approach"
        self.announce(ErrorClass.LOOP, f"{call}: {pattern}; strike {strikes}/{LOOP_STRIKE_LIMIT}; {action}")
        reason = f"the model kept repeating itself ({pattern}; last call {call})"
        return LoopVerdict(reply, reason, ended)

    def silent_reply(self) -> str | None:
        """A reply without a tool call in a phase that needs one: the nudge to send, or ``None`` to end the phase."""
        self._silent += 1
        if self._silent > MAX_SILENT_REPLIES:
            self.announce(ErrorClass.NO_TOOL_CALL, "still no tool call after the nudge; ending the phase")
            return None
        self.announce(ErrorClass.NO_TOOL_CALL, "the reply had no tool call; nudged the model to call one")
        return NUDGE

    def tool_used(self) -> None:
        """The model made a tool call: the next tool-less reply is a first offence again."""
        self._silent = 0

    def review(
        self, tool: str, args: Mapping[str, Any], ok: bool, output: str, meta: Mapping[str, Any], sandbox: Sandbox
    ) -> str:
        """The text the model should read for a finished tool call (see ``review_result``)."""
        review = review_result(tool, args, ok, output, meta, sandbox)
        if review.error_class is not None:
            self.announce(review.error_class, review.detail)
        return review.output


# ---- LLM failures ---------------------------------------------------------------------------


def llm_failure_advice(exc: LLMError) -> str:
    """What the person running the harness can do about a failed LLM call, by cause."""
    status = exc.status_code
    text = f"{exc} {exc.detail}".lower()
    if isinstance(exc, LLMConfigError):
        return "Check AI_API_KEY and the model settings in config.yaml."
    if status in (401, 403):
        return "The provider rejected the credentials: check that AI_API_KEY is valid for the configured model and base_url."
    if status == 404:
        return "The model or endpoint was not found: check model and base_url in config.yaml (or AI_MODEL and AI_BASE_URL)."
    if status == 413 or any(word in text for word in ("context length", "context window", "too many tokens", "too large")):
        return "The prompt is larger than the model accepts: lower max_context_tokens in config.yaml."
    if status == 429:
        return (
            f"The provider's rate limit or quota was still exhausted after {exc.attempts or 'several'} attempts: "
            "wait and re-run, or use another model."
        )
    if status is not None and status >= 500:
        return f"The provider kept failing after {exc.attempts or 'several'} attempts; re-run later."
    if exc.retryable:
        return f"The network or provider kept failing after {exc.attempts or 'several'} attempts; re-run later."
    return "The model call was rejected; see the message above."


# ---- checkpoint and rollback ----------------------------------------------------------------


def usable_ref(ref: object) -> bool:
    """Whether a sandbox's checkpoint result is a ref that can be rolled back to.

    Some sandboxes report a failure as an ``"error: ..."`` string instead of raising; that, ``None`` and an
    empty string are not refs.
    """
    return isinstance(ref, str) and bool(ref.strip()) and not ref.strip().lower().startswith("error")


class Checkpointer:
    """Checkpoints and rolls back the working tree through the ``Sandbox`` interface.

    A sandbox's checkpoint or rollback may sweep away untracked files (``git stash`` does), and the
    repro script is untracked, so the files named by ``protected_paths`` are put back if missing.
    A sandbox's checkpoint must snapshot the tree without changing it, but a ``git stash`` based one empties it.
    The edits the model has made are therefore read before the checkpoint and written back if they are gone; if
    that is impossible the checkpoint is undone and reported as failed, so a rework round never starts on a tree
    that silently lost the patch it is meant to improve.

    Nothing here raises: a failed checkpoint or rollback is announced and reported by its result.
    """

    def __init__(self, sandbox: Sandbox, emitter: Emitter, protected_paths: Callable[[], Iterable[str]]) -> None:
        self._sandbox = sandbox
        self._emitter = emitter
        self._protected = protected_paths

    def checkpoint(self, label: str) -> str | None:
        """Snapshot the tree; returns the ref, or ``None`` if the sandbox could not (it raised, or returned no ref)."""
        saved = self._save()
        edits = self._edits()
        self._emitter.tool_call("checkpoint", {"label": label})
        try:
            ref = self._sandbox.checkpoint(label)
        except Exception as exc:  # noqa: BLE001
            self._emitter.tool_result("checkpoint", False, str(exc))
            self._emitter.error("sandbox", f"checkpoint failed: {exc}")
            return None
        self._restore(saved)
        if not usable_ref(ref):
            self._emitter.tool_result("checkpoint", False, str(ref))
            self._emitter.error("sandbox", f"checkpoint failed: the sandbox returned {str(ref)[:200]!r} instead of a ref")
            return None
        if not self._edits_survived(edits):
            self._sandbox_undo(ref)
            self._restore(saved)
            self._emitter.tool_result("checkpoint", False, "the checkpoint emptied the working tree")
            self._emitter.error("sandbox", "checkpoint failed: it emptied the working tree and the edits could not be restored")
            return None
        self._emitter.tool_result("checkpoint", True, str(ref))
        return ref

    def rollback(self, ref: str | None, tried: Sequence[str] = ()) -> bool:
        """Reset the tree to ``ref``; ``tried`` are the approaches the model is about to be told were abandoned.

        A ref that is not usable is refused without touching the sandbox: rolling back to nothing would
        discard the working tree and restore nothing.
        """
        if not usable_ref(ref):
            return False
        saved = self._save()
        self._emitter.tool_call("rollback", {"ref": ref})
        try:
            self._sandbox.rollback(ref)
        except Exception as exc:  # noqa: BLE001
            self._emitter.tool_result("rollback", False, str(exc))
            self._emitter.error("sandbox", f"rollback failed: {exc}")
            return False
        self._restore(saved)
        self._emitter.tool_result("rollback", True, f"restored {ref}")
        told = f"; the model is told about {len(tried)} approach(es) already tried" if tried else ""
        self._emitter.error("rollback", f"reset the working tree to {ref}{told}")
        return True

    def _edits(self) -> dict[str, str | None]:
        """Contents of the files that differ from the baseline (``None`` for a deleted one), read through the sandbox."""
        try:
            paths = [change.path for change in changed_files(filter_diff(self._sandbox.diff()))]
        except Exception:  # noqa: BLE001 - without a readable diff there is nothing to protect or verify
            return {}
        edits: dict[str, str | None] = {}
        for path in paths:
            try:
                edits[path] = self._sandbox.read_file(path)
            except Exception:  # noqa: BLE001 - a deleted file: it cannot be written back
                edits[path] = None
        return edits

    def _edits_survived(self, edits: Mapping[str, str | None]) -> bool:
        """After a checkpoint: are the edits still in the tree? Writes back any that are gone; ``False`` if it cannot."""
        if not edits:
            return True
        gone = self._gone(edits)
        if not gone:
            return True
        for path in gone:
            content = edits[path]
            if content is not None:
                try:
                    self._sandbox.write_file(path, content)
                except Exception:  # noqa: BLE001
                    return False
        remaining = self._gone(edits)
        if not remaining:
            self._emitter.error("sandbox", f"the checkpoint emptied the working tree; wrote back {len(gone)} edited file(s)")
        return not remaining

    def _gone(self, edits: Mapping[str, str | None]) -> list[str]:
        try:
            present = {change.path for change in changed_files(filter_diff(self._sandbox.diff()))}
        except Exception:  # noqa: BLE001
            return list(edits)
        return [path for path in edits if path not in present]

    def _sandbox_undo(self, ref: str) -> None:
        """Undo a checkpoint that damaged the tree by rolling back to the snapshot it just took."""
        try:
            self._sandbox.rollback(ref)
        except Exception:  # noqa: BLE001 - nothing more can be done; the caller reports the failure
            pass

    def _save(self) -> dict[str, str]:
        saved = {}
        for path in self._protected():
            try:
                saved[path] = self._sandbox.read_file(path)
            except Exception:  # noqa: BLE001 - already gone: nothing to preserve
                continue
        return saved

    def _restore(self, saved: Mapping[str, str]) -> None:
        for path, content in saved.items():
            try:
                self._sandbox.read_file(path)
            except Exception:  # noqa: BLE001 - missing: write it back
                self._sandbox.write_file(path, content)
