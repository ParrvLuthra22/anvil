"""The retry that replaces a bare assert must not be undone because the model's old repro expects the assert.

Real failure (pallets__flask-4045, Qwen3-Coder): the retry wrote `raise ValueError(...)`, but the repro the model had written earlier
caught AssertionError specifically and printed "Other error", so the harness's repro re-run failed, the retry counted as failed
verification, and the assert patch was restored: the wrong exception type was delivered, and the hidden test wants ValueError.
"""

from __future__ import annotations

from anvil.agent.prompts import patch_kickoff
from anvil.sandbox.base import ExecResult
from tests.fakes import REPRO_CMD, FakePipeline, FakeRegistry, FakeSandbox, call, default_tools, done, give_up, project_files, reply
from tests.test_agent_assert_warning import GUARD, RAISE, asserting_patch, replace_the_assert
from tests.test_agent_patch_sanity import git_world, kickoffs, sanity_line
from tests.test_orchestrator import execute, finalize, localize, reproduce, review_ok, understand, verify

UPDATE_REPRO = "update .anvil/repro.py to expect the new exception type, then re-run it"
KEPT = (
    "After replacing the assert, the repro (written for the assert) no longer passed, but the repository's tests do, "
    "so the patch with the specific exception was kept."
)


def pinned_to_the_assert(tests_pass_with_raise: bool = True, registry: FakeRegistry | None = None) -> FakePipeline:
    """A project whose repro expects the assert: once calc.py raises instead, the repro fails unless it was updated."""
    world = git_world()
    inner = world.sandbox._on_exec

    def exec_(cmd: str, files: dict[str, str]) -> ExecResult:
        raised = "raise TypeError" in files["calc.py"]
        if cmd.startswith(REPRO_CMD) and raised and "expects TypeError" not in files.get(".anvil/repro.py", ""):
            return ExecResult(1, "Other error: a must be an int\n", "", False, 0.01)
        if cmd.startswith("pytest") and raised and not tests_pass_with_raise:
            return ExecResult(1, "1 failed\n", "FAILED tests/test_calc.py::test_add\n", False, 0.1)
        return inner(cmd, files)

    return FakePipeline(FakeSandbox(project_files(), on_exec=exec_), registry)


def solve_with_the_assert():
    return understand() + localize() + reproduce() + asserting_patch() + verify() + review_ok()


def verify_phase_calls(run) -> list[str]:
    """The tools called in VERIFY, in order: the harness's repro re-runs, the model's run_tests, and any run_tests of the harness's own."""
    return [e.data["tool"] for e in run.of("tool_call") if e.phase.value == "verify" and e.data["tool"] != "phase_done"]


def rolled_back(pipeline) -> bool:
    return any(e[0] == "rollback" for e in pipeline.sandbox.events)


# ---- what the model is told ------------------------------------------------------------------------------------------------


def test_the_kickoff_tells_the_model_to_update_its_repro_to_expect_the_new_exception_and_run_it_again():
    text = patch_kickoff(attempt=2, repro_cmd=REPRO_CMD, repro_output="AssertionError", feedback="a bare assert", kind="sanity_warning")
    assert UPDATE_REPRO in text
    assert "keep the repro passing" not in text, "the old instruction was the trap: a repro pinned to the assert vetoed the fix"
    assert "ValueError, TypeError, or the type the issue names" in text and "This is your last attempt" in text


def test_the_kickoff_the_model_receives_in_a_run_carries_it(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + finalize()
    run = execute(script, tmp_path, pipeline=pinned_to_the_assert())
    (told,) = kickoffs(run)
    assert UPDATE_REPRO in told


# ---- the repro still fails after the retry: the repository's tests decide ----------------------------------------------------


def test_a_repro_that_fails_on_the_new_exception_does_not_restore_the_assert_when_the_repository_tests_pass(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + finalize()
    fake = pinned_to_the_assert(tests_pass_with_raise=True)
    run = execute(script, tmp_path, pipeline=fake)

    assert "raise TypeError" in run.patch and "assert isinstance" not in run.patch, "the specific exception is delivered"
    assert not rolled_back(fake), "nothing was undone"
    assert "Verified after patching: yes" in run.report
    assert KEPT in run.report and "restored" not in run.report
    assert sanity_line(run).startswith("- Patch sanity: passed after one forced-fix retry (the first patch was flagged")
    assert "WARNING" not in sanity_line(run)
    assert run.llm.remaining == 0


def test_the_report_shows_the_failed_repro_and_the_passing_tests_and_the_confidence_is_not_high(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + finalize()
    run = execute(script, tmp_path, pipeline=pinned_to_the_assert())
    assert f"- FAIL: repro `{REPRO_CMD}` (re-run by the harness)" in run.report
    assert "- pass: run_tests (run by the harness)" in run.report
    assert "Confidence: **medium**" in run.report, "a repro that fails is a caveat even when the tests pass"


def test_the_harness_runs_the_repository_tests_itself_and_the_trace_shows_it(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + finalize()
    run = execute(script, tmp_path, pipeline=pinned_to_the_assert())
    assert verify_phase_calls(run) == ["repro", "run_tests", "repro", "run_tests"], "first VERIFY, then the retry's: repro, then the tests"
    results = [e for e in run.of("tool_result") if e.data["tool"] == "run_tests"]
    assert [r.data["ok"] for r in results] == [True, True]


def test_when_the_repository_tests_fail_too_the_earlier_patch_is_restored(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + finalize()
    fake = pinned_to_the_assert(tests_pass_with_raise=False)
    run = execute(script, tmp_path, pipeline=fake)

    assert "assert isinstance" in run.patch and "raise TypeError" not in run.patch, "the verified patch is back"
    assert rolled_back(fake)
    assert "The retry to replace the assert failed verification; the earlier patch was restored." in run.report
    assert KEPT not in run.report
    assert "Verified after patching: yes" in run.report, "the earlier patch was verified"
    assert "- FAIL" not in run.report and "- pass: run_tests" in run.report, "the checks that verified the delivered patch, not the retry's"
    assert "Confidence: **high**" in run.report
    assert "WARNING: The patch adds a bare assert" in sanity_line(run)


def test_with_no_run_tests_tool_the_harness_cannot_judge_and_restores_as_before(tmp_path):
    no_tests = FakeRegistry([t for t in default_tools() if t.name != "run_tests"])
    script = understand() + localize() + reproduce() + asserting_patch() + [done("the repro passes")] + review_ok() + replace_the_assert() + finalize()
    fake = pinned_to_the_assert(registry=no_tests)
    run = execute(script, tmp_path, pipeline=fake)
    assert "assert isinstance" in run.patch and rolled_back(fake)
    assert "The retry to replace the assert failed verification; the earlier patch was restored." in run.report
    assert "run_tests" not in verify_phase_calls(run)


def test_a_run_tests_tool_that_crashes_is_a_failed_judgement_not_a_crashed_run(tmp_path):
    from tests.fakes import crashing

    tools = [t for t in default_tools() if t.name != "run_tests"] + [crashing("run_tests")]
    script = understand() + localize() + reproduce() + asserting_patch() + [done("the repro passes")] + review_ok() + replace_the_assert() + finalize()
    fake = pinned_to_the_assert(registry=FakeRegistry(tools))
    run = execute(script, tmp_path, pipeline=fake)
    assert "assert isinstance" in run.patch and rolled_back(fake)
    (crash,) = [e for e in run.of("tool_result") if e.data["tool"] == "run_tests" and not e.data["ok"]]
    assert "run_tests failed to run: RuntimeError: tool exploded" in crash.data["output_preview"]


# ---- the repro was updated, or the failure is not about the repro -----------------------------------------------------------


def test_a_model_that_updates_its_repro_and_runs_it_needs_no_arbitration(tmp_path):
    update_repro = reply(call("edit_file", path=".anvil/repro.py", old="assert add(2, 3) == 5", new="# expects TypeError\nassert add(2, 3) == 5"))
    retry = [reply(call("edit_file", path="calc.py", old=GUARD, new=RAISE)), update_repro, reply(call("run_cmd", cmd=REPRO_CMD)), done("raises TypeError; repro updated")]
    script = solve_with_the_assert() + retry + verify() + finalize()
    fake = pinned_to_the_assert()
    run = execute(script, tmp_path, pipeline=fake)

    assert "raise TypeError" in run.patch and "assert isinstance" not in run.patch
    assert "Verified after patching: yes" in run.report and "Confidence: **high**" in run.report
    assert KEPT not in run.report and "- FAIL" not in run.report, "the repro passed, so there was nothing to arbitrate"
    assert not rolled_back(fake)
    assert run.llm.remaining == 0


def test_a_repro_that_still_passes_after_the_retry_is_the_ordinary_path(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + verify() + finalize()
    run = execute(script, tmp_path, pipeline=git_world())
    assert "raise TypeError" in run.patch and KEPT not in run.report and "Confidence: **high**" in run.report
    assert run.llm.remaining == 0


def test_a_retry_for_an_empty_patch_is_not_judged_by_the_tests(tmp_path):
    """Only the assert retry is arbitrated: for the others a repro that fails is a failed verification, as before."""
    wrong = [reply(call("edit_file", path="calc.py", old="return a - b", new="return a * b")), done("tried multiplying")]
    script = understand() + localize() + reproduce() + [give_up("no idea")] + wrong + finalize()
    fake = pinned_to_the_assert()
    run = execute(script, tmp_path, pipeline=fake, max_patch_attempts=1, max_rollbacks=0)
    assert "cannot be handed in as it is" in kickoffs(run)[0]
    assert verify_phase_calls(run) == ["repro"], "the harness re-ran the repro, it failed, and no tests were run to overrule it"
    assert "Verified after patching: no" in run.report and KEPT not in run.report


def test_a_retry_whose_repro_passes_but_whose_verify_fails_is_not_overruled_by_the_tests(tmp_path):
    """The tests are only the tie-break for a repro that fails: a VERIFY that the model itself ended with give_up stands."""
    script = solve_with_the_assert() + replace_the_assert()[:-1] + [reply(call("edit_file", path=".anvil/repro.py", old="assert add(2, 3) == 5", new="# expects TypeError\nassert add(2, 3) == 5")), reply(call("run_cmd", cmd=REPRO_CMD)), done("raises TypeError; repro updated")]
    script += [give_up("a test I read is wrong for the new exception")] + finalize()
    fake = pinned_to_the_assert()
    run = execute(script, tmp_path, pipeline=fake)

    assert verify_phase_calls(run) == ["repro", "run_tests", "repro", "give_up", "rollback"], "the harness did not run the tests to overrule the model"
    assert "assert isinstance" in run.patch and rolled_back(fake)
    assert "The retry to replace the assert failed verification; the earlier patch was restored." in run.report
    assert KEPT not in run.report
