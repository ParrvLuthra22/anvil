"""PhaseRunner: the tool loop behind every LLM phase."""

import pytest

from anvil.agent.budget import Budget, BudgetExceeded
from anvil.agent.emitter import Emitter
from anvil.agent.loop import PhaseRunner, PhaseStatus, RunAborted
from anvil.agent.prompts import PHASE_SPECS
from anvil.agent.recovery import MAX_SILENT_REPLIES, NUDGE
from anvil.agent.settings import AgentSettings
from anvil.context import ContextManager
from anvil.events import EventBus, Phase
from anvil.llm.client import LLMResponse
from anvil.llm.errors import LLMError
from tests.fakes import FakeRegistry, FakeSandbox, RecordingLLM, call, crashing, default_tools, done, give_up, project_sandbox, reply

LOCALIZE = PHASE_SPECS[Phase.LOCALIZE]
PATCH = PHASE_SPECS[Phase.PATCH]
UNDERSTAND = PHASE_SPECS[Phase.UNDERSTAND]


class Harness:
    """A runner wired to fakes, with the history and the event stream exposed."""

    def __init__(self, script, *, registry=None, sandbox=None, llm=None, **settings):
        self.llm = llm or RecordingLLM(script)
        self.sandbox = sandbox or project_sandbox()
        self.settings = AgentSettings(**settings)
        self.ctx = ContextManager(tool_output_char_cap=self.settings.tool_output_char_cap)
        self.budget = Budget(self.settings)
        bus = EventBus()
        self.queue = bus.subscribe()
        self._seen = []
        self.runner = PhaseRunner(
            llm=self.llm,
            ctx=self.ctx,
            registry=registry or FakeRegistry(),
            sandbox=self.sandbox,
            emitter=Emitter(bus),
            budget=self.budget,
            settings=self.settings,
        )

    def run(self, spec=LOCALIZE, kickoff="go", gate=None):
        return self.runner.run(spec, kickoff, gate)

    @property
    def history(self):
        return self.ctx.build_messages("goal")[1:]

    def events(self, kind=None):
        while not self.queue.empty():
            self._seen.append(self.queue.get_nowait())
        return [e for e in self._seen if kind is None or e.type == kind]


def tool_results(harness):
    return [m["content"] for m in harness.history if m["role"] == "tool"]


# ---- ending a phase -------------------------------------------------------------------------


def test_phase_done_ends_the_phase_with_its_summary_and_arguments():
    h = Harness([reply(call("grep", pattern="add")), done("found it in calc.py", extra=1)])
    outcome = h.run()
    assert outcome.done and outcome.status is PhaseStatus.DONE
    assert outcome.summary == "found it in calc.py"
    assert outcome.args == {"summary": "found it in calc.py", "extra": 1}
    assert [r.tool for r in outcome.records] == ["grep"]


def test_give_up_ends_the_phase_unsuccessfully_with_its_reason():
    outcome = Harness([give_up("nothing matches")]).run()
    assert outcome.status is PhaseStatus.GAVE_UP and not outcome.done
    assert outcome.summary == "nothing matches"


def test_history_is_valid_openai_tool_calling_format():
    h = Harness([reply(call("grep", pattern="add"), text="searching"), done()])
    h.run()
    roles = [m["role"] for m in h.history]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assistant, tool = h.history[1], h.history[2]
    (tc,) = assistant["tool_calls"]
    assert tc["type"] == "function" and tc["function"]["name"] == "grep"
    assert tc["function"]["arguments"] == '{"pattern": "add"}'
    assert tool["tool_call_id"] == tc["id"]


def test_the_phase_prompt_and_only_allowed_tool_schemas_are_sent():
    h = Harness([done()])
    h.run(LOCALIZE)
    messages, tools = h.llm.calls[0]
    assert messages[0] == {"role": "system", "content": LOCALIZE.system_prompt}
    assert {t["function"]["name"] for t in tools} == {"list_dir", "grep", "read_file", "phase_done", "give_up"}


def test_reproduce_gets_the_write_repro_schema_only_if_registered():
    h = Harness([done()])
    h.run(PHASE_SPECS[Phase.REPRODUCE])
    assert "write_repro" not in {t["function"]["name"] for t in h.llm.calls[0][1]}


# ---- tools ----------------------------------------------------------------------------------


def test_tools_outside_the_phase_allowlist_are_refused_and_do_not_run():
    h = Harness([reply(call("edit_file", **{"path": "calc.py", "old": "a - b", "new": "a + b"})), done()])
    h.run(LOCALIZE)
    (result,) = tool_results(h)[:1]
    assert "not available in the localize phase" in result and "list_dir" in result
    assert "a - b" in h.sandbox.files["calc.py"]


def test_an_allowlisted_tool_missing_from_the_registry_is_reported_not_raised():
    h = Harness([reply(call("grep", pattern="x")), done()], registry=FakeRegistry([]))
    outcome = h.run()
    assert outcome.done
    assert tool_results(h)[0] == "Unknown tool 'grep'."


def test_a_crashing_tool_becomes_a_failed_result_and_the_phase_carries_on():
    registry = FakeRegistry([crashing("read_file", RuntimeError("disk on fire")), *default_tools()[:2]])
    h = Harness([reply(call("read_file", path="calc.py")), done("worked around it")], registry=registry)
    outcome = h.run()
    assert outcome.done and outcome.summary == "worked around it"
    assert "crashed (RuntimeError: disk on fire)" in tool_results(h)[0]
    (error,) = h.events("error")
    assert error.data["kind"] == "tool" and "disk on fire" in error.data["message"]
    assert outcome.records[0].ok is False


def test_tool_output_is_capped_before_it_reaches_the_model():
    sandbox = FakeSandbox({"big.txt": "line\n" * 5000})
    h = Harness([reply(call("read_file", path="big.txt")), done()], sandbox=sandbox, tool_output_char_cap=1000)
    h.run()
    (output,) = tool_results(h)[:1]
    assert len(output) <= 1000 and "lines omitted" in output
    assert output.startswith("line\nline") and output.endswith("line\n")


def test_invalid_json_arguments_are_reported_and_the_tool_is_not_run():
    bad = {"id": "c1", "tool": "edit_file", "args": {}, "error": "arguments were not a valid JSON object"}
    h = Harness([LLMResponse("", [bad], {}), done()], sandbox=project_sandbox())
    h.run(PATCH)
    assert "Invalid arguments for 'edit_file'" in tool_results(h)[0]
    assert "a - b" in h.sandbox.files["calc.py"]


def test_tool_events_are_emitted_in_order_with_usage():
    h = Harness([reply(call("grep", pattern="add"), tokens=200), done()])
    h.run()
    kinds = [e.type for e in h.events()]
    assert kinds == ["message", "llm_usage", "tool_call", "tool_result", "llm_usage", "tool_call", "tool_result"]
    usage = h.events("llm_usage")[0].data
    assert usage["total_tokens"] == 200 and usage["cost_estimate"] == 0.0


def test_registry_tool_records_exclude_control_tools_and_keep_meta():
    h = Harness([reply(call("run_cmd", cmd="pytest")), done()])
    outcome = h.run(PATCH)
    (record,) = outcome.records
    assert record.tool == "run_cmd" and record.ok is False and record.meta == {"exit_code": 1, "timed_out": False}


# ---- several calls in one reply -------------------------------------------------------------


def test_calls_after_phase_done_are_skipped_but_still_answered():
    h = Harness([reply(call("phase_done", summary="early"), call("grep", pattern="add"))])
    outcome = h.run()
    assert outcome.summary == "early" and outcome.records == []
    assert len(tool_results(h)) == 2 and "Skipped" in tool_results(h)[1]


def test_several_ordinary_calls_in_one_reply_all_run():
    h = Harness([reply(call("grep", pattern="add"), call("list_dir", path=".")), done()])
    outcome = h.run()
    assert [r.tool for r in outcome.records] == ["grep", "list_dir"]


# ---- gate -----------------------------------------------------------------------------------


def test_gate_rejection_keeps_the_phase_running_until_the_model_complies():
    seen = []

    def gate(args):
        seen.append(args["summary"])
        return "not good enough" if args["summary"] == "first try" else None

    h = Harness([done("first try"), done("second try")])
    outcome = h.run(gate=gate)
    assert outcome.done and outcome.summary == "second try"
    assert seen == ["first try", "second try"]
    assert tool_results(h) == ["not good enough", "Phase complete."]


# ---- misbehaving models ---------------------------------------------------------------------


def test_a_reply_without_tool_calls_gets_a_nudge_and_the_phase_can_recover():
    h = Harness([reply(text="I think it is in calc.py"), done("ok")])
    outcome = h.run()
    assert outcome.done
    assert h.history[2] == {"role": "user", "content": NUDGE}


def test_a_model_that_never_uses_tools_stalls_the_phase():
    h = Harness([reply(text=f"chatter {i}") for i in range(MAX_SILENT_REPLIES + 1)])
    outcome = h.run()
    assert outcome.status is PhaseStatus.STALLED
    assert "stopped calling tools" in outcome.summary and "chatter" in outcome.summary
    assert h.llm.remaining == 0


def test_the_silent_reply_counter_resets_after_a_tool_call():
    script = [reply(text="a"), reply(call("grep", pattern="x")), reply(text="b"), reply(call("grep", pattern="y")), done()]
    assert Harness(script).run().done


def test_a_phase_stops_at_its_step_limit():
    h = Harness([reply(call("grep", pattern="add")) for _ in range(10)], max_steps_per_phase=4)
    outcome = h.run()
    assert outcome.status is PhaseStatus.STEP_LIMIT
    assert h.llm.remaining == 6 and h.budget.steps == 4


# ---- text-only phases -----------------------------------------------------------------------


def test_text_only_phase_accepts_a_plain_text_answer_as_its_summary():
    outcome = Harness([reply(text="  add() subtracts instead of adding.  ")]).run(UNDERSTAND)
    assert outcome.done and outcome.summary == "add() subtracts instead of adding."


def test_text_only_phase_also_accepts_phase_done():
    assert Harness([done("analysis")]).run(UNDERSTAND).summary == "analysis"


def test_text_only_phase_gets_at_most_three_steps_and_no_registry_tools():
    h = Harness([reply(call("grep", pattern="x")) for _ in range(6)])
    outcome = h.run(UNDERSTAND)
    assert outcome.status is PhaseStatus.STEP_LIMIT and h.budget.steps == 3
    assert {t["function"]["name"] for t in h.llm.calls[0][1]} == {"phase_done", "give_up"}


# ---- budgets and failures -------------------------------------------------------------------


def test_tokens_are_charged_to_the_global_budget():
    h = Harness([reply(call("grep", pattern="x"), tokens=300), done()])
    h.run()
    assert h.budget.tokens == 400 and h.budget.steps == 2


def test_missing_usage_is_tolerated():
    h = Harness([LLMResponse("", [call("phase_done", summary="s")], {})])
    assert h.run().done and h.budget.tokens == 0


def test_the_global_step_budget_propagates_out_of_the_phase():
    h = Harness([reply(call("grep", pattern="x")) for _ in range(5)], max_total_steps=2)
    with pytest.raises(BudgetExceeded) as info:
        h.run()
    assert info.value.kind == "steps" and h.llm.remaining == 3


def test_an_llm_failure_aborts_the_run_and_is_reported():
    class Failing:
        def chat(self, messages, tools=None):
            raise LLMError("HTTP 401", status_code=401)

    h = Harness([], llm=Failing())
    with pytest.raises(RunAborted, match="HTTP 401"):
        h.run()
    (error,) = h.events("error")
    assert error.data["kind"] == "llm"


def test_call_ids_are_generated_when_the_model_layer_supplies_none():
    h = Harness([LLMResponse("", [{"tool": "grep", "args": {"pattern": "x"}}], {}), done()])
    h.run()
    assistant, tool = h.history[1], h.history[2]
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] != ""


def test_kickoff_message_is_added_to_history_and_announced():
    h = Harness([done()])
    h.run(kickoff="Begin phase LOCALIZE.")
    assert h.history[0] == {"role": "user", "content": "Begin phase LOCALIZE."}
    assert h.events("message")[0].data == {"role": "user", "text": "Begin phase LOCALIZE."}
