"""The tool loop every LLM phase runs through.

One ``PhaseRunner.run`` call is one phase: it feeds the model the phase prompt and
history, executes the tools it asks for (only those on the phase's allowlist),
and returns as soon as the model ends the phase with ``phase_done`` or
``give_up``, runs out of steps, or stops using tools. It never raises for
model or tool misbehaviour; only the global budget and an unusable LLM stop a run
(``BudgetExceeded`` and ``RunAborted``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from anvil.agent.budget import Budget
from anvil.agent.emitter import Emitter
from anvil.agent.prompts import GIVE_UP, PHASE_DONE, PhaseSpec, control_tools
from anvil.agent.settings import AgentSettings
from anvil.agent.text import clip_head, clip_middle
from anvil.context import ContextManager
from anvil.llm.client import LLMClient, LLMResponse
from anvil.llm.errors import LLMError
from anvil.sandbox.base import Sandbox
from anvil.tools.registry import ToolRegistry

TEXT_ONLY_STEP_CAP = 3
MAX_SILENT_REPLIES = 2
NUDGE = (
    "Reply with a tool call. Call phase_done(summary) if this phase's objective is met, "
    "or give_up(reason) if it cannot be met."
)

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
    ) -> None:
        self._llm = llm
        self._ctx = ctx
        self._registry = registry
        self._sandbox = sandbox
        self._emitter = emitter
        self._budget = budget
        self._settings = settings
        self._call_seq = 0

    def run(self, spec: PhaseSpec, kickoff: str, gate: Gate | None = None) -> PhaseOutcome:
        """Run ``spec``'s phase, starting from the ``kickoff`` message.

        ``gate`` (optional) vets ``phase_done`` calls; see ``Gate``. Raises
        ``BudgetExceeded`` when a global budget is hit and ``RunAborted`` when the
        LLM fails for good.
        """
        self._ctx.add_message("user", kickoff)
        self._emitter.message("user", kickoff)
        schemas = self._schemas(spec)
        records: list[ToolRecord] = []
        silent_replies = 0
        steps = self._settings.max_steps_per_phase
        if spec.text_only:
            steps = min(steps, TEXT_ONLY_STEP_CAP)

        for _ in range(steps):
            response = self._ask(spec, schemas)
            calls = [self._normalise(raw) for raw in response.tool_calls or []]
            self._record_assistant(response.text, calls)
            if not calls:
                if spec.text_only:
                    return PhaseOutcome(PhaseStatus.DONE, response.text.strip(), records=records)
                silent_replies += 1
                if silent_replies > MAX_SILENT_REPLIES:
                    reason = f"the model stopped calling tools: {clip_head(response.text.strip(), 300)}"
                    return PhaseOutcome(PhaseStatus.STALLED, reason, records=records)
                self._ctx.add_message("user", NUDGE)
                self._emitter.message("user", NUDGE)
                continue
            silent_replies = 0
            outcome = self._execute(spec, calls, gate, records)
            if outcome is not None:
                return outcome
        return PhaseOutcome(PhaseStatus.STEP_LIMIT, f"phase step limit ({steps}) reached", records=records)

    # ---- model call -------------------------------------------------------------------------

    def _schemas(self, spec: PhaseSpec) -> list[dict]:
        allowed = [s for s in self._registry.schemas() if _schema_name(s) in spec.tools]
        return allowed + control_tools(spec.phase)

    def _ask(self, spec: PhaseSpec, schemas: list[dict]) -> LLMResponse:
        self._budget.charge_step()
        try:
            response = self._llm.chat(self._ctx.build_messages(spec.system_prompt), schemas)
        except LLMError as exc:
            self._emitter.error("llm", str(exc))
            raise RunAborted(f"LLM call failed: {exc}") from exc
        usage = response.usage or {}
        prompt, completion = _count(usage, "prompt_tokens"), _count(usage, "completion_tokens")
        total = _count(usage, "total_tokens") or prompt + completion
        self._budget.add_tokens(total)
        self._emitter.usage(prompt, completion, total, self._settings.cost_estimate(prompt, completion))
        return response

    def _normalise(self, raw: dict) -> _Call:
        self._call_seq += 1
        args = raw.get("args")
        return _Call(
            id=str(raw.get("id") or f"call_{self._call_seq}"),
            name=str(raw.get("tool") or raw.get("name") or ""),
            args=args if isinstance(args, dict) else {},
            error=str(raw.get("error") or ""),
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
        self, spec: PhaseSpec, calls: list[_Call], gate: Gate | None, records: list[ToolRecord]
    ) -> PhaseOutcome | None:
        """Run the calls of one reply in order; return the outcome if one of them ended the phase."""
        outcome: PhaseOutcome | None = None
        for call in calls:
            if outcome is not None:
                self._ctx.add_message("tool", "Skipped: the phase already ended.", tool_call_id=call.id)
                continue
            self._emitter.tool_call(call.name, call.args)
            outcome = self._execute_one(spec, call, gate, records)
        return outcome

    def _execute_one(
        self, spec: PhaseSpec, call: _Call, gate: Gate | None, records: list[ToolRecord]
    ) -> PhaseOutcome | None:
        if call.error:
            self._reply(call, False, f"Invalid arguments for '{call.name}': {call.error}")
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

        ok, output, meta = self._run_tool(spec, call)
        records.append(ToolRecord(call.name, call.args, ok, output, meta))
        self._reply(call, ok, output)
        return None

    def _run_tool(self, spec: PhaseSpec, call: _Call) -> tuple[bool, str, dict]:
        if call.name not in spec.tools:
            offered = ", ".join([*spec.tools, PHASE_DONE, GIVE_UP])
            message = f"Tool '{call.name}' is not available in the {spec.phase.value} phase. Use one of: {offered}."
            return False, message, {}
        try:
            tool = self._registry.get(call.name)
        except LookupError:
            return False, f"Unknown tool '{call.name}'.", {}
        try:
            result = tool.run(call.args, self._sandbox)
        except Exception as exc:  # noqa: BLE001 - a crashing tool must not end the run
            message = f"{type(exc).__name__}: {exc}"
            self._emitter.error("tool", f"{call.name} crashed: {message}")
            return False, f"Tool '{call.name}' crashed ({message}). Try a different approach.", {}
        return result.ok, clip_middle(result.output, self._settings.tool_output_char_cap), dict(result.meta)

    def _reply(self, call: _Call, ok: bool, output: str) -> None:
        """Answer a tool call in both the history and the event stream."""
        self._emitter.tool_result(call.name, ok, output)
        self._ctx.add_message("tool", output, tool_call_id=call.id)


def _schema_name(schema: dict) -> str:
    return str(schema.get("function", schema).get("name", ""))


def _count(usage: dict, key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
