"""Each failure class through the tool loop, with a scripted model and an in-memory sandbox."""

from __future__ import annotations

import pytest

from anvil.agent.loop import PhaseStatus, RunAborted
from anvil.agent.prompts import PHASE_SPECS
from anvil.agent.recovery import LOOP_STRIKE_LIMIT, NUDGE
from anvil.events import Phase
from anvil.llm.client import LLMResponse
from anvil.llm.errors import LLMConfigError, LLMError
from anvil.sandbox.base import ExecResult
from anvil.tools.base import ToolResult
from tests.fakes import (
    BUGGY,
    FakeRegistry,
    FakeSandbox,
    FakeTool,
    call,
    default_tools,
    done,
    project_exec,
    project_files,
    project_sandbox,
    reply,
    schema,
)
from tests.test_agent_loop import Harness, tool_results
from tests.test_context import assert_valid

LOCALIZE = PHASE_SPECS[Phase.LOCALIZE]
REPRODUCE = PHASE_SPECS[Phase.REPRODUCE]
PATCH = PHASE_SPECS[Phase.PATCH]
VERIFY = PHASE_SPECS[Phase.VERIFY]
UNDERSTAND = PHASE_SPECS[Phase.UNDERSTAND]


def kinds(h: Harness) -> list[str]:
    return [e.data["kind"] for e in h.events("error")]


def messages_of(h: Harness, kind: str) -> list[str]:
    return [e.data["message"] for e in h.events("error") if e.data["kind"] == kind]


def counting_grep() -> tuple[FakeTool, list[dict]]:
    """A grep that records every real execution."""
    runs: list[dict] = []

    def grep(args: dict, sandbox) -> ToolResult:
        runs.append(args)
        return ToolResult(True, "calc.py:1:def add(a, b):")

    return FakeTool("grep", grep, parameters=schema(["pattern"], pattern="string")), runs


def registry_with(tool: FakeTool) -> FakeRegistry:
    return FakeRegistry([tool, *[t for t in default_tools() if t.name != tool.name]])


# ---- loops ----------------------------------------------------------------------------------


def test_the_third_identical_call_in_a_row_is_not_run_and_the_model_is_told_to_change_approach():
    grep, runs = counting_grep()
    h = Harness([reply(call("grep", pattern="add"))] * 3 + [done("found it")], registry=registry_with(grep))
    outcome = h.run()

    assert outcome.done and len(runs) == 2, "the repeated call must not execute"
    blocked = tool_results(h)[2]
    assert blocked.startswith("Not run: you are repeating yourself (the same call 3 times in a row)")
    assert "Try a different approach" in blocked and "Strike 1 of 3" in blocked
    assert kinds(h) == ["loop"]
    assert_valid(h.ctx.build_messages("goal"))


def test_two_identical_calls_are_fine():
    grep, runs = counting_grep()
    h = Harness([reply(call("grep", pattern="add"))] * 2 + [done()], registry=registry_with(grep))
    assert h.run().done and len(runs) == 2 and kinds(h) == []


def test_an_a_b_a_b_pattern_blocks_the_fourth_call():
    grep, runs = counting_grep()
    a, b = call("grep", pattern="add"), call("read_file", path="calc.py")
    h = Harness([reply(a), reply(b), reply(a), reply(b), done()], registry=registry_with(grep))
    outcome = h.run()
    assert outcome.done and len(runs) == 2
    assert "alternating between the same two calls" in tool_results(h)[3]
    assert [r.tool for r in outcome.records] == ["grep", "read_file", "grep"], "the fourth call never ran"


def test_a_model_that_keeps_repeating_after_the_warnings_has_its_phase_ended_at_strike_three():
    grep, runs = counting_grep()
    script = [reply(call("grep", pattern="add"))] * 8 + [done("never reached")]
    h = Harness(script, registry=registry_with(grep))
    outcome = h.run(LOCALIZE)

    assert outcome.status is PhaseStatus.LOOPED and not outcome.done
    assert "the model kept repeating itself" in outcome.summary and "grep(pattern='add')" in outcome.summary
    assert len(runs) == 2
    assert kinds(h) == ["loop"] * LOOP_STRIKE_LIMIT
    assert "ending the phase" in messages_of(h, "loop")[-1]
    assert [f"Strike {n} of 3" in r for n, r in enumerate(tool_results(h)[2:], 1)] == [True] * 3
    assert h.llm.remaining == 4, "the phase ended without asking the model again"
    assert_valid(h.ctx.build_messages("goal"))


def test_changing_course_between_loops_keeps_the_phase_alive_but_the_strikes_add_up():
    grep, _ = counting_grep()
    x, y = call("grep", pattern="x"), call("grep", pattern="y")
    script = [reply(x)] * 3 + [reply(y)] + [reply(x)] * 2 + [reply(y)] * 2 + [done("ok")]
    h = Harness(script, registry=registry_with(grep))
    assert h.run().done
    assert kinds(h) == ["loop"]


def test_repeated_rejections_of_phase_done_are_the_gates_business_not_a_loop():
    attempts = []

    def gate(args):
        attempts.append(args)
        return None if len(attempts) == 5 else "not yet"

    h = Harness([done("same")] * 5)
    outcome = h.run(gate=gate)
    assert outcome.done and len(attempts) == 5 and kinds(h) == []


def test_repeating_an_invalid_call_is_a_loop_too():
    h = Harness([reply(call("frobnicate", x=1))] * 5 + [done()])
    outcome = h.run()
    assert outcome.status is PhaseStatus.LOOPED
    assert kinds(h) == ["invalid_call", "invalid_call"] + ["loop"] * 3


def test_a_loop_is_only_counted_within_one_phase():
    grep, runs = counting_grep()
    h = Harness([reply(call("grep", pattern="a"))] * 2 + [done()] + [reply(call("grep", pattern="a"))] * 2 + [done()], registry=registry_with(grep))
    assert h.run().done and h.run().done
    assert len(runs) == 4 and kinds(h) == []


# ---- failed edits ---------------------------------------------------------------------------


def edit(old: str, new: str = "return a + b"):
    return reply(call("edit_file", path="calc.py", old=old, new=new))


def test_a_failed_edit_comes_back_with_the_closest_lines_and_a_reminder_to_reread():
    h = Harness([edit("return a * b"), done()], sandbox=project_sandbox())
    h.run(PATCH)
    (result,) = tool_results(h)[:1]
    assert result.startswith("String not found exactly once in 'calc.py'.")
    assert "Closest matching lines in calc.py:\n  2:     return a - b" in result
    assert "Re-read the file with read_file before editing again" in result
    assert h.sandbox.files["calc.py"] == BUGGY
    assert kinds(h) == ["edit"] and "calc.py" in messages_of(h, "edit")[0]


def test_the_model_can_recover_from_a_failed_edit_and_the_next_one_raises_no_event():
    h = Harness([edit("return a * b"), edit("return a - b"), done("fixed")], sandbox=project_sandbox())
    outcome = h.run(PATCH)
    assert outcome.done and "return a + b" in h.sandbox.files["calc.py"]
    assert kinds(h) == ["edit"]
    assert [r.ok for r in outcome.records] == [False, True]


def test_an_indentation_mismatch_is_pointed_out():
    h = Harness([edit("def add(a, b):\nreturn a - b"), done()], sandbox=project_sandbox())
    h.run(PATCH)
    assert "  2:     return a - b" in tool_results(h)[0]


def test_an_edit_to_a_missing_file_says_where_to_look():
    sandbox = project_sandbox()
    h = Harness([reply(call("edit_file", path="nope.py", old="x", new="y")), done()], sandbox=sandbox)
    h.run(PATCH)
    assert "File not found: 'nope.py'" in tool_results(h)[0] and "list_dir or grep" in tool_results(h)[0]


def test_lines_a_real_edit_tool_already_listed_are_not_duplicated():
    def edit_tool(args, sandbox):
        return ToolResult(False, "String not found in 'calc.py'.\nClosest existing lines (for reference):\n    return a - b")

    tool = FakeTool("edit_file", edit_tool, parameters=schema(["path", "old", "new"], path="string", old="string", new="string"))
    h = Harness([edit("return a * b"), done()], registry=registry_with(tool), sandbox=project_sandbox())
    h.run(PATCH)
    result = tool_results(h)[0]
    assert result.count("return a - b") == 1 and "Re-read the file" in result


# ---- invalid tool calls ---------------------------------------------------------------------


def test_an_unknown_tool_is_not_run_and_the_model_gets_the_tools_it_can_use():
    h = Harness([reply(call("read_fil", path="calc.py")), done()])
    h.run(LOCALIZE)
    (result,) = tool_results(h)[:1]
    assert "Tool 'read_fil' is not available in the localize phase." in result
    assert "Did you mean 'read_file'?" in result and '"required":["path"]' in result
    assert "Tools you can call here: list_dir(path), grep(pattern, path?, file_pattern?), read_file(path, start?, end?)" in result
    assert "phase_done(summary)" in result and kinds(h) == ["invalid_call"]


def test_a_real_tool_that_the_phase_does_not_allow_is_refused_without_a_misleading_suggestion():
    h = Harness([reply(call("edit_file", path="calc.py", old="a - b", new="a + b")), done()], sandbox=project_sandbox())
    h.run(LOCALIZE)
    (result,) = tool_results(h)[:1]
    assert "not available in the localize phase" in result and "Did you mean" not in result
    assert h.sandbox.files["calc.py"] == BUGGY


def test_arguments_that_are_not_valid_json_get_the_tools_schema_back():
    bad = {"id": "c1", "tool": "edit_file", "args": {}, "error": "arguments were not a valid JSON object"}
    h = Harness([LLMResponse("", [bad], {}), done()], sandbox=project_sandbox())
    h.run(PATCH)
    result = tool_results(h)[0]
    assert "Invalid arguments for 'edit_file': arguments were not a valid JSON object." in result
    assert 'edit_file takes (JSON Schema): {"type":"object","properties":{"path":{"type":"string"}' in result
    assert kinds(h) == ["invalid_call"]


def test_a_missing_required_argument_is_caught_before_the_tool_runs():
    grep, runs = counting_grep()
    h = Harness([reply(call("grep")), reply(call("grep", pattern="add")), done()], registry=registry_with(grep))
    outcome = h.run(LOCALIZE)
    assert outcome.done and runs == [{"pattern": "add"}]
    first = tool_results(h)[0]
    assert "missing required argument 'pattern'" in first and '"required":["pattern"]' in first
    assert kinds(h) == ["invalid_call"]


def test_a_wrong_argument_type_is_caught_before_the_tool_runs():
    h = Harness([reply(call("read_file", path="calc.py", start="ten")), done()])
    h.run(LOCALIZE)
    assert "argument 'start' must be of type integer, got str" in tool_results(h)[0]


def test_an_invalid_call_does_not_end_the_phase_and_the_model_can_fix_it():
    h = Harness([reply(call("read_file")), reply(call("read_file", path="calc.py")), done("ok")])
    outcome = h.run(LOCALIZE)
    assert outcome.done and "def add" in tool_results(h)[1]


def test_control_tools_are_left_to_the_gate_and_the_phase():
    h = Harness([reply(call("phase_done"))])
    outcome = h.run(LOCALIZE)
    assert outcome.done and outcome.summary == "" and kinds(h) == []


def test_a_tool_the_registry_lost_is_still_reported_plainly():
    h = Harness([reply(call("grep", pattern="x")), done()], registry=FakeRegistry([]))
    h.run(LOCALIZE)
    assert tool_results(h)[0] == "Unknown tool 'grep'."


# ---- replies without a tool call ------------------------------------------------------------


def test_a_reply_without_a_tool_call_is_nudged_once_and_the_phase_can_carry_on():
    h = Harness([reply(text="I think it is calc.py"), reply(call("grep", pattern="add")), done("ok")])
    outcome = h.run(LOCALIZE)
    assert outcome.done
    assert h.history[2] == {"role": "user", "content": NUDGE}
    assert kinds(h) == ["no_tool_call"]


def test_a_second_reply_without_a_tool_call_ends_the_phase():
    h = Harness([reply(text="hmm"), reply(text="still thinking"), done("never reached")])
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.STALLED
    assert "stopped calling tools: still thinking" in outcome.summary
    assert kinds(h) == ["no_tool_call", "no_tool_call"] and "ending the phase" in messages_of(h, "no_tool_call")[1]
    assert h.llm.remaining == 1
    assert [m["content"] for m in h.history if m["content"] == NUDGE] == [NUDGE], "nudged exactly once"


def test_a_phase_with_no_tools_needs_no_nudge():
    outcome = Harness([reply(text="add() subtracts.")]).run(UNDERSTAND)
    assert outcome.done and outcome.summary == "add() subtracts."


# ---- failing tests, timeouts and other tool errors ------------------------------------------


def test_a_failing_test_run_leads_with_the_failing_lines_and_says_not_to_touch_the_tests():
    h = Harness([reply(call("run_tests", target="tests/test_calc.py")), done()], sandbox=project_sandbox())
    outcome = h.run(VERIFY)
    result = tool_results(h)[0]
    assert result.startswith("[test run failed]\nKey failure lines:\n  FAILED tests/test_calc.py::test_add")
    assert "Never weaken, skip or delete a test" in result
    assert result.endswith("1 failed\nFAILED tests/test_calc.py::test_add\n"), "the runner's own output follows in full"
    assert kinds(h) == ["test_failure"]
    assert outcome.records[0].output == "1 failed\nFAILED tests/test_calc.py::test_add\n", "the record keeps the raw output"


def test_passing_tests_and_failing_repros_raise_no_recovery_events():
    sandbox = project_sandbox()
    sandbox.files[".anvil/repro.py"] = "from calc import add\nassert add(2, 3) == 5\n"
    h = Harness([reply(call("run_cmd", cmd="python .anvil/repro.py")), done()], sandbox=sandbox)
    h.run(REPRODUCE)
    assert "AssertionError" in tool_results(h)[0] and kinds(h) == []


def timeout_sandbox() -> FakeSandbox:
    return FakeSandbox(project_files(), on_exec=lambda cmd, files: ExecResult(-1, "", "", True, 120.0))


@pytest.mark.parametrize("tool, args", [("run_cmd", {"cmd": "pytest"}), ("run_tests", {"target": "tests"})])
def test_a_timeout_tells_the_model_to_run_something_narrower(tool, args):
    h = Harness([reply(call(tool, **args)), done()], sandbox=timeout_sandbox())
    h.run(VERIFY)
    result = tool_results(h)[0]
    assert "[TIMED OUT after 120s]" in result and "run something narrower" in result
    assert kinds(h) == ["timeout"], "a timeout is classified as a timeout, not as a failing test"


def test_a_failing_tool_gets_advice_and_an_event():
    h = Harness([reply(call("read_file", path="nope.py")), done()])
    h.run(LOCALIZE)
    result = tool_results(h)[0]
    assert result.startswith("File not found: 'nope.py'") and "list_dir or grep" in result
    assert "do not repeat the same call" in result and kinds(h) == ["tool"]


def test_a_tool_that_crashes_is_still_one_tool_event_with_no_extra_advice():
    from tests.fakes import crashing

    registry = FakeRegistry([crashing("read_file", RuntimeError("disk on fire")), *default_tools()[:2]])
    h = Harness([reply(call("read_file", path="calc.py")), done()], registry=registry)
    h.run(LOCALIZE)
    assert tool_results(h)[0].count("Try a different approach") == 1
    assert kinds(h) == ["tool"] and "crashed" in messages_of(h, "tool")[0]


# ---- LLM failures ---------------------------------------------------------------------------


class Failing:
    def __init__(self, exc: LLMError) -> None:
        self.exc = exc

    def chat(self, messages, tools=None):
        raise self.exc


@pytest.mark.parametrize(
    "exc, advice",
    [
        (LLMError("HTTP 401", status_code=401), "rejected the credentials"),
        (LLMError("HTTP 429", status_code=429, retryable=True, attempts=5), "rate limit or quota"),
        (LLMError("HTTP 503", status_code=503, retryable=True, attempts=5), "kept failing after 5 attempts"),
        (LLMError("HTTP 413", status_code=413), "lower max_context_tokens"),
        (LLMConfigError("AI_API_KEY is not set"), "Check AI_API_KEY"),
    ],
)
def test_an_llm_failure_ends_the_run_with_advice_on_the_cause(exc, advice):
    h = Harness([], llm=Failing(exc))
    with pytest.raises(RunAborted) as info:
        h.run()
    assert advice in str(info.value) and str(exc) in str(info.value)
    (event,) = h.events("error")
    assert event.data["kind"] == "llm" and advice in event.data["message"]


def test_an_llm_failure_never_reaches_the_tools():
    grep, runs = counting_grep()
    h = Harness([], llm=Failing(LLMError("HTTP 500", status_code=500, retryable=True, attempts=5)), registry=registry_with(grep))
    with pytest.raises(RunAborted):
        h.run()
    assert runs == []
