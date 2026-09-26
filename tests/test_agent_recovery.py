"""Each failure class through the tool loop, with a scripted model and an in-memory sandbox."""

from __future__ import annotations

import pytest

from anvil.agent.loop import PhaseStatus, RunAborted
from anvil.agent.prompts import PHASE_SPECS
from anvil.agent.recovery import LOOP_STRIKE_LIMIT, NUDGE
from anvil.agent.state import REVIEW_APPROVED, CheckRun, RunState
from anvil.events import Phase
from anvil.llm.client import LLMResponse
from anvil.llm.errors import LLMConfigError, LLMError
from anvil.sandbox.base import ExecResult
from anvil.tools.base import ToolResult
from tests.fakes import (
    BUGGY,
    FIX,
    REPRO_CMD,
    FakePipeline,
    FakeRegistry,
    FakeSandbox,
    FakeTool,
    call,
    default_tools,
    done,
    project_exec,
    project_files,
    RecordingLLM,
    project_sandbox,
    reply,
    schema,
)
from tests.test_agent_loop import Harness, tool_results
from tests.test_context import assert_valid
from tests.test_orchestrator import (
    HAPPY_STEPS,
    execute,
    finalize,
    good_patch,
    happy,
    localize,
    reproduce,
    review_ok,
    understand,
    verify,
    wrong_patch,
)

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


# ---- in a full run --------------------------------------------------------------------------


def run_kinds(run) -> list[str]:
    return [e.data["kind"] for e in run.of("error")]


def stuck_grep(times: int) -> list:
    return [reply(call("grep", pattern="add")) for _ in range(times)]


def test_a_phase_that_loops_is_ended_and_the_run_carries_on_to_the_end(tmp_path):
    script = understand() + stuck_grep(5) + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path)

    assert run_kinds(run) == ["loop"] * LOOP_STRIKE_LIMIT
    assert "The fault was not localized (looped: the model kept repeating itself" in run.report
    assert "+    return a + b" in run.patch and run.llm.remaining == 0
    assert run.phases == ["ingest", "profile", "understand", "localize", "reproduce", "patch", "verify", "review", "finalize"]


def test_a_model_that_goes_silent_while_patching_is_nudged_once_then_the_attempt_is_retried(tmp_path):
    silent = [reply(text="Let me think."), reply(text="Still thinking.")]
    script = understand() + localize() + reproduce() + silent + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path)

    assert run_kinds(run) == ["no_tool_call", "no_tool_call"]
    assert "Patch attempts: 2" in run.report and "+    return a + b" in run.patch
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)
    assert run.prompt_texts().count(NUDGE) >= 1


def test_a_failed_edit_in_a_full_run_is_recovered_from_with_the_closest_lines(tmp_path):
    bad_edit = reply(call("edit_file", path="calc.py", old="return a * b", new="return a + b"))
    patch = [bad_edit, reply(call("edit_file", **FIX)), reply(call("run_cmd", cmd=REPRO_CMD)), done("changed - to +")]
    run = execute(understand() + localize() + reproduce() + patch + verify() + review_ok() + finalize(), tmp_path)

    assert run_kinds(run) == ["edit"]
    assert any("Closest matching lines in calc.py:\n  2:     return a - b" in t for t in run.prompt_texts())
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)


def test_calling_a_tool_that_does_not_exist_costs_a_step_but_not_the_run(tmp_path):
    lost = [reply(call("search_code", query="add"))]
    script = understand() + lost + localize() + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path)
    assert run_kinds(run) == ["invalid_call"]
    assert any("Tool 'search_code' is not available in the localize phase" in t for t in run.prompt_texts())
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)


def test_a_failing_test_run_in_verify_is_announced_and_leads_with_the_failing_lines(tmp_path):
    def flaky_suite(cmd, files):
        if cmd.startswith("pytest"):
            return ExecResult(1, "1 failed\n", "FAILED tests/test_other.py::test_unrelated\n", False, 0.1)
        return project_exec(cmd, files)

    pipeline = FakePipeline(FakeSandbox(project_files(), on_exec=flaky_suite))
    script = (
        understand() + localize() + reproduce() + good_patch()
        + [reply(call("run_tests")), done("only a pre-existing unrelated failure")] + review_ok() + finalize()
    )
    run = execute(script, tmp_path, pipeline=pipeline)

    assert run_kinds(run) == ["test_failure"]
    (event,) = run.of("error")
    assert event.phase.value == "verify" and "fix the cause, not the tests" in event.data["message"]
    shown = [t for t in run.prompt_texts() if t.startswith("[test run failed]")]
    assert shown and "Key failure lines:\n  FAILED tests/test_other.py::test_unrelated" in shown[0]
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6), "the failing run still costs confidence"


def test_a_rollback_resets_the_tree_announces_itself_and_the_model_learns_what_was_tried(tmp_path):
    script = (
        understand() + localize() + reproduce()
        + wrong_patch("return a - b", "return a * b", "multiplied")
        + wrong_patch("return a * b", "return a ** b", "exponent")
        + good_patch()
        + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, max_patch_attempts=2, max_rollbacks=1)

    assert run_kinds(run) == ["rollback"]
    (event,) = run.of("error")
    assert "ckpt-1" in event.data["message"] and "2 approach(es)" in event.data["message"]
    rollback_at = run.events.index(event)
    before = run.events[rollback_at - 2 : rollback_at]
    assert [(e.type, e.data["tool"]) for e in before] == [("tool_call", "rollback"), ("tool_result", "rollback")]
    rethink = next(t for t in run.prompt_texts() if "DIFFERENT hypothesis" in t)
    assert "- multiplied" in rethink and "- exponent" in rethink
    assert ".anvil/repro.py" in run.pipeline.sandbox.files, "the repro outlives the rollback"
    assert "+    return a + b" in run.patch and "a ** b" not in run.patch


def test_a_sandbox_that_cannot_checkpoint_is_reported_and_the_run_goes_on_without_rollbacks(tmp_path):
    class NoCheckpoints(FakeSandbox):
        def checkpoint(self, label):
            raise RuntimeError("not a git repository")

    pipeline = FakePipeline(NoCheckpoints(project_files(), on_exec=project_exec))
    script = understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, pipeline=pipeline)
    assert run_kinds(run) == ["sandbox"] and "not a git repository" in run.of("error")[0].data["message"]
    assert "Checkpointing failed, so rolling back was not possible." in run.report
    assert "+    return a + b" in run.patch


# ---- budgets and dying LLMs -----------------------------------------------------------------


def test_a_budget_stop_is_announced_and_finalises_with_lowered_confidence(tmp_path):
    run = execute(understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + finalize(), tmp_path, max_total_steps=HAPPY_STEPS - 3)

    (event,) = run.of("error")
    assert event.data["kind"] == "budget" and "step budget exhausted" in event.data["message"]
    assert "lowering the confidence" in event.data["message"]
    assert "confidence is lowered" in run.report
    assert "+    return a + b" in run.patch and (run.out / "report.md").exists()
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6), "verified but unreviewed: medium at best"
    assert run.pipeline.sandbox.closed and run.phases[-1] == "finalize"


@pytest.mark.parametrize("budget", [{"max_total_steps": 6}, {"max_tokens_total": 700}, {"wall_clock_seconds": 0}])
def test_every_kind_of_budget_ends_in_outputs_a_done_event_and_a_low_confidence(tmp_path, budget):
    run = execute(happy(), tmp_path, **budget)
    assert run_kinds(run) == ["budget"]
    assert run.done.data["resolved_confidence"] <= 0.3
    assert (run.out / "patch.diff").exists() and (run.out / "report.md").exists()


def test_a_run_cut_short_is_never_reported_as_high_confidence():
    state = RunState("u", repro_confirmed=True, verified=True, review=REVIEW_APPROVED, checks=[CheckRun("t", True)])
    assert state.confidence(True) == "high"
    state.halted = "budget"
    assert state.confidence(True) == "medium"
    assert state.confidence_score(True) == 0.6 and state.confidence(False) == "none"


class DyingLLM(RecordingLLM):
    """Serves the script, then fails for good after ``after`` calls (the client's retries are used up)."""

    def __init__(self, script, after: int, exc: LLMError) -> None:
        super().__init__(script)
        self.after, self.exc = after, exc

    def chat(self, messages, tools=None):
        if len(self.calls) >= self.after:
            raise self.exc
        return super().chat(messages, tools)


@pytest.mark.parametrize(
    "exc, advice",
    [
        (LLMError("HTTP 429", status_code=429, retryable=True, attempts=5), "rate limit or quota"),
        (LLMError("HTTP 401", status_code=401), "rejected the credentials"),
    ],
)
def test_an_llm_that_dies_after_the_patch_still_yields_the_patch_a_report_and_low_confidence(tmp_path, exc, advice):
    script = understand() + localize() + reproduce() + good_patch()
    llm = DyingLLM(script, after=len(script), exc=exc)
    run = execute(None, tmp_path, llm=llm)

    assert run_kinds(run) == ["llm"] and advice in run.of("error")[0].data["message"]
    assert "+    return a + b" in run.patch, "the work done before the failure is kept"
    assert "Stopped early: LLM call failed" in run.report and advice in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.3)
    assert run.pipeline.sandbox.closed and run.events[-1] is run.done


def test_a_sandbox_that_reports_checkpoint_failure_as_a_string_is_never_rolled_back_to(tmp_path):
    """Regression (audit): 'error: git stash failed' was accepted as a ref, logged ok=True, and rolled back to."""

    class StringErrors(FakeSandbox):
        rollbacks: list = []

        def checkpoint(self, label):
            return "error: git stash failed — fatal: not a git repository"

        def rollback(self, ref):
            self.rollbacks.append(ref)

    sandbox = StringErrors(project_files(), on_exec=project_exec)
    script = (
        understand() + localize() + reproduce() + wrong_patch("return a - b", "return a * b", "multiplied") + finalize()
    )
    run = execute(script, tmp_path, pipeline=FakePipeline(sandbox), max_patch_attempts=1, max_rollbacks=2)

    assert sandbox.rollbacks == [], "no rollback may be attempted with a ref that is really an error message"
    assert "a * b" in sandbox.files["calc.py"], "the model's work is still there"
    assert run_kinds(run) == ["sandbox"]
    checkpoint_result = next(e for e in run.of("tool_result") if e.data["tool"] == "checkpoint")
    assert checkpoint_result.data["ok"] is False
    assert "Checkpointing failed, so rolling back was not possible." in run.report
    assert "rollbacks: 0" in run.report
