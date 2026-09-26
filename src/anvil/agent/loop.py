"""The tool loop every LLM phase runs through.

One ``PhaseRunner.run`` call is one phase: it feeds the model the phase prompt and
history, executes the tools it asks for (only those on the phase's allowlist),
and returns as soon as the model ends the phase with ``phase_done`` or
``give_up``, runs out of steps, stops using tools, or gets stuck in a loop. It never
raises for model or tool misbehaviour; only the global budget and an unusable LLM
stop a run (``BudgetExceeded`` and ``RunAborted``).

Misbehaviour is handled by ``anvil.agent.recovery``: a ``PhaseGuard`` vets each call
(unknown tool, bad arguments, repeats) and reviews each result (failed edit, failing
tests, timeout), announcing every intervention as an ``error`` event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from anvil.agent.budget import Budget
from anvil.agent.emitter import Emitter
from anvil.agent.prompts import GIVE_UP, PHASE_DONE, PhaseSpec, closing_message, control_tools
from anvil.agent.recovery import ErrorClass, PhaseGuard, llm_failure_advice
from anvil.agent.settings import AgentSettings
from anvil.agent.text import clip_head
from anvil.agent.usage import record_usage
from anvil.context import ContextManager
from anvil.llm.client import LLMClient, LLMResponse
from anvil.llm.errors import LLMError
from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool
from anvil.tools.registry import ToolRegistry

TEXT_ONLY_STEP_CAP = 3

# A gate inspects the arguments of an attempted ``phase_done``. It returns None to
# accept, or a message that is sent back to the model (the phase then continues).
Gate = Callable[[dict], "str | None"]


class RunAborted(Exception):
    """The run cannot continue (the LLM is unusable); wrap up with what exists."""


class PhaseStatus(str, Enum):
    """How a phase ended."""

    DONE = "done"
    GAVE_UP = "gave_up"
    STEP_LIMIT = "step_limit"
    STALLED = "stalled"
    LOOPED = "looped"


@dataclass
class ToolRecord:
    """One executed registry tool call and what it returned."""

    tool: str
    args: dict
    ok: bool
    output: str
    meta: dict = field(default_factory=dict)


@dataclass
class PhaseOutcome:
    """Result of one phase: how it ended, the closing text, and every tool call it made."""

    status: PhaseStatus
    summary: str = ""
    args: dict = field(default_factory=dict)
    records: list[ToolRecord] = field(default_factory=list)
    forced: bool = False  # the phase ran into its call cap and was made to close (whether or not it then did)

    @property
    def done(self) -> bool:
        """True when the model ended the phase with ``phase_done``."""
        return self.status is PhaseStatus.DONE


@dataclass
class _Call:
    id: str
    name: str
    args: dict
    error: str = ""
    ignored: int = 0  # further calls in the same (text-mode) reply that the client dropped


class PhaseRunner:
    """Runs LLM phases against one conversation, one sandbox and one budget."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        ctx: ContextManager,
        registry: ToolRegistry,
        sandbox: Sandbox,
        emitter: Emitter,
        budget: Budget,
        settings: AgentSettings,
        environment: str = "",
    ) -> None:
        self._environment = environment
        self._llm = llm
        self._ctx = ctx
        self._registry = registry
        self._sandbox = sandbox
        self._emitter = emitter
        self._budget = budget
        self._settings = settings
        self._call_seq = 0
        self._forced = False

    def run(self, spec: PhaseSpec, kickoff: str, gate: Gate | None = None) -> PhaseOutcome:
        """Run ``spec``'s phase, starting from the ``kickoff`` message.

        ``gate`` (optional) vets ``phase_done`` calls; see ``Gate``. Raises
        ``BudgetExceeded`` when a global budget is hit and ``RunAborted`` when the
        LLM fails for good.
        """
        self._ctx.begin_phase(kickoff)
        self._emitter.message("user", kickoff)
        schemas = self._schemas(spec)
        guard = PhaseGuard(self._emitter, spec.phase, _parameters_by_name(schemas))
        records: list[ToolRecord] = []
        steps = self._settings.max_steps_per_phase
        if spec.text_only:
            steps = min(steps, TEXT_ONLY_STEP_CAP)
        cap = self._settings.call_cap(spec.phase.value)  # a cap of ``steps`` or more never comes up: the hard limit is first
        self._forced = False

        for step in range(steps):
            # ``step`` is how many calls have been made. A cap of N gives the model N calls of its own, and call N+1 is the harness's forced close, so a phase makes at most N+1 calls.
            if cap is not None and step == cap:
                schemas = self._force_close(spec, cap)
            response = self._ask(spec, schemas)
            calls = [self._normalise(raw) for raw in response.tool_calls or []]
            self._record_assistant(response.text, calls)
            if not calls:
                if spec.text_only:
                    return PhaseOutcome(PhaseStatus.DONE, response.text.strip(), records=records, forced=self._forced)
                if self._forced:
                    reason = f"call cap ({cap}) reached and the closing call made no tool call: {clip_head(response.text.strip(), 300)}"
                    return PhaseOutcome(PhaseStatus.STEP_LIMIT, reason, records=records, forced=True)
                nudge = guard.silent_reply()
                if nudge is None:
                    reason = f"the model stopped calling tools: {clip_head(response.text.strip(), 300)}"
                    return PhaseOutcome(PhaseStatus.STALLED, reason, records=records)
                self._ctx.add_message("user", nudge)
                self._emitter.message("user", nudge)
                continue
            guard.tool_used()
            outcome = self._execute(spec, calls, gate, records, guard)
            if outcome is not None:
                outcome.forced = self._forced
                return outcome
            if self._forced:
                return PhaseOutcome(
                    PhaseStatus.STEP_LIMIT, f"call cap ({cap}) reached and the closing call did not end the phase",
                    records=records, forced=True,
                )
        return PhaseOutcome(PhaseStatus.STEP_LIMIT, f"phase step limit ({steps}) reached", records=records)

    def _force_close(self, spec: PhaseSpec, cap: int) -> list[dict]:
        """Tell the model its calls are used up and offer it nothing but the two ways to end the phase."""
        self._forced = True
        message = closing_message(spec.phase, cap)
        self._ctx.add_message("user", message)
        self._emitter.message("user", message)
        return control_tools(spec.phase)

    # ---- model call -------------------------------------------------------------------------

    def _schemas(self, spec: PhaseSpec) -> list[dict]:
        allowed = [s for s in self._registry.schemas() if _schema_name(s) in spec.tools]
        return allowed + control_tools(spec.phase)

    def _ask(self, spec: PhaseSpec, schemas: list[dict]) -> LLMResponse:
        self._budget.charge_step()
        try:
            response = self._llm.chat(self._ctx.build_messages(spec.system_prompt + self._environment), schemas)
        except LLMError as exc:
            advice = llm_failure_advice(exc)
            self._emitter.error(ErrorClass.LLM_ERROR.value, f"{exc} {advice}")
            raise RunAborted(f"LLM call failed: {exc} {advice}") from exc
        record_usage(response, self._budget, self._emitter, self._settings)
        return response

    def _normalise(self, raw: dict) -> _Call:
        self._call_seq += 1
        args = raw.get("args")
        return _Call(
            id=str(raw.get("id") or f"call_{self._call_seq}"),
            name=str(raw.get("tool") or raw.get("name") or ""),
            args=args if isinstance(args, dict) else {},
            error=str(raw.get("error") or ""),
            ignored=int(raw.get("ignored_calls") or 0),
        )

    def _record_assistant(self, text: str, calls: list[_Call]) -> None:
        if text.strip():
            self._emitter.message("assistant", text)
        fields: dict[str, Any] = {}
        if calls:
            fields["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.args, default=str)},
                }
                for c in calls
            ]
        self._ctx.add_message("assistant", text, **fields)

    # ---- tool execution ---------------------------------------------------------------------

    def _execute(
        self, spec: PhaseSpec, calls: list[_Call], gate: Gate | None, records: list[ToolRecord], guard: PhaseGuard
    ) -> PhaseOutcome | None:
        """Run the calls of one reply in order; return the outcome if one of them ended the phase."""
        outcome: PhaseOutcome | None = None
        for call in calls:
            if outcome is not None:
                self._ctx.add_message("tool", "Skipped: the phase already ended.", tool_call_id=call.id)
                continue
            self._emitter.tool_call(call.name, call.args)
            if call.ignored:
                guard.announce(
                    ErrorClass.INVALID_CALL,
                    f"the reply had {call.ignored + 1} tool calls; only the first ({call.name}) was run",
                )
            outcome = self._execute_one(spec, call, gate, records, guard)
        return outcome

    def _execute_one(
        self, spec: PhaseSpec, call: _Call, gate: Gate | None, records: list[ToolRecord], guard: PhaseGuard
    ) -> PhaseOutcome | None:
        control = call.name in (PHASE_DONE, GIVE_UP)
        if self._forced and not control:
            self._reply(call, False, "This phase's calls are used up: only phase_done or give_up can be called now.")
            return None
        if not control:
            verdict = guard.repeated(call.name, call.args)
            if verdict is not None:
                self._reply(call, False, verdict.reply)
                if verdict.exhausted:
                    return PhaseOutcome(PhaseStatus.LOOPED, verdict.reason, records=records)
                return None
        if call.error:
            self._reply(call, False, guard.invalid_arguments(call.name, call.error))
            return None
        if call.name == PHASE_DONE:
            rejection = gate(call.args) if gate else None
            if rejection:
                self._reply(call, False, rejection)
                return None
            self._reply(call, True, "Phase complete.")
            return PhaseOutcome(PhaseStatus.DONE, str(call.args.get("summary", "")).strip(), call.args, records)
        if call.name == GIVE_UP:
            self._reply(call, True, "Understood.")
            return PhaseOutcome(PhaseStatus.GAVE_UP, str(call.args.get("reason", "")).strip(), call.args, records)

        tool, refusal = self._resolve(spec, call, guard)
        if tool is None:
            self._reply(call, False, refusal)
            return None
        ok, output, meta, crashed = self._run_tool(tool, call, guard)
        records.append(ToolRecord(call.name, call.args, ok, output, meta))
        if not crashed:
            output = guard.review(call.name, call.args, ok, output, meta, self._sandbox)
        self._reply(call, ok, output)
        return None

    def _resolve(self, spec: PhaseSpec, call: _Call, guard: PhaseGuard) -> tuple[Tool | None, str]:
        """The tool to run for ``call``, or ``None`` and the reply explaining why it cannot run."""
        try:
            tool = self._registry.get(call.name)
        except LookupError:
            tool = None
        if call.name not in spec.tools:
            return None, guard.unknown_tool(call.name, exists=tool is not None)
        if tool is None:
            return None, f"Unknown tool '{call.name}'."
        problem = guard.check_arguments(call.name, call.args)
        if problem:
            return None, problem
        return tool, ""

    def _run_tool(self, tool: Tool, call: _Call, guard: PhaseGuard) -> tuple[bool, str, dict, bool]:
        """Run ``tool``; returns ok, output, meta and whether it crashed (a crash is not the tool's own verdict)."""
        try:
            result = tool.run(call.args, self._sandbox)
        except Exception as exc:  # noqa: BLE001 - a crashing tool must not end the run
            message = f"{type(exc).__name__}: {exc}"
            guard.announce(ErrorClass.TOOL_ERROR, f"{call.name} crashed: {message}")
            return False, f"Tool '{call.name}' crashed ({message}). Try a different approach.", {}, True
        return result.ok, result.output, dict(result.meta), False

    def _reply(self, call: _Call, ok: bool, output: str) -> None:
        """Answer a tool call in both the history and the event stream."""
        if call.ignored:
            output += (
                f"\n\nNote: your reply contained {call.ignored} more tool call(s). Only this first one was run; "
                "call one tool per reply and wait for its result."
            )
        self._emitter.tool_result(call.name, ok, output)
        self._ctx.add_message("tool", output, tool_call_id=call.id, ok=ok)


def _schema_name(schema: dict) -> str:
    return str(schema.get("function", schema).get("name", ""))


def _parameters_by_name(schemas: list[dict]) -> dict[str, dict]:
    """Each offered tool's JSON-Schema parameters, by tool name."""
    return {_schema_name(s): dict(s.get("function", s).get("parameters") or {}) for s in schemas}

