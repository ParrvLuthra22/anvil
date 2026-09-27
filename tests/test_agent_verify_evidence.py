"""VERIFY that the model does not finish is decided from the evidence, and VERIFY -> PATCH loop-backs are bounded.

Seen with a real model (Qwen3-Coder, B2): it fixed the bug, VERIFY ran the repro and run_tests twice (all passed), then the model
read files and ran commands and sent two replies without a tool call. The phase ended "stalled", which counted as a failed
verification, and the run went back through PATCH and VERIFY for a fix that already worked (48 steps, 137k tokens).
"""

from __future__ import annotations

import pytest

from anvil.agent.state import CheckRun, RunState
from anvil.sandbox.base import ExecResult
from tests.fakes import FakePipeline, FakeSandbox, call, done, give_up, project_exec, project_files, reply
from tests.test_orchestrator import execute, finalize, good_patch, localize, reproduce, review_ok, understand

_n = iter(range(10_000))


def run_tests():
    return reply(call("run_tests", target="tests/test_calc.py"))


def wander():
    return reply(call("read_file", path=f"src/mod{next(_n)}.py"))


def stall():
    """Two replies without a tool call: the nudge, and then the phase ends as stalled."""
    return [reply(text="I believe everything is fine."), reply(text="Yes, it is done.")]


def run_with(script, tmp_path, exec_=None, **config):
    return execute(script, tmp_path, pipeline=FakePipeline(FakeSandbox(project_files(), on_exec=exec_ or project_exec)), **config)


def phases(run, name: str) -> int:
    return sum(1 for e in run.of("phase") if e.data["name"] == name)


def failing_once(tests_failing_times: int = 1):
    """The fake project, except that the first ``tests_failing_times`` pytest runs fail (a flaky or pre-existing failure)."""
    seen = {"n": 0}

    def exec_(cmd, files):
        if cmd.startswith("pytest"):
            seen["n"] += 1
            if seen["n"] <= tests_failing_times:
                return ExecResult(1, "1 failed\n", "FAILED tests/test_calc.py::test_add\n", False, 0.1)
        return project_exec(cmd, files)

    return exec_


def base():
    return understand() + localize() + reproduce() + good_patch()


# ---- an unfinished VERIFY is closed on the evidence ------------------------------------------------------------------


def test_verify_that_stalls_after_the_tests_passed_is_verified_and_does_not_send_the_run_back_to_patch(tmp_path):
    """The B2 run: run_tests passed, then the model wandered and went silent."""
    script = base() + [run_tests(), wander(), *stall()] + review_ok() + finalize()
    run = run_with(script, tmp_path)

    assert "Verified after patching: yes" in run.report
    assert phases(run, "patch") == 1, "no second PATCH: the fix that works was not re-opened"
    assert "VERIFY ended without the model's verdict (stalled); the harness closed it on the evidence" in run.report
    assert "Confidence: **high**" in run.report, "a passing repro and a passing test run are what a verdict would have rested on"
    assert run.llm.remaining == 0


def test_verify_closed_at_its_call_cap_is_decided_the_same_way(tmp_path):
    script = base() + [run_tests(), wander(), wander()] + review_ok() + finalize()  # 2 calls of its own, then it ignores the close
    run = run_with(script, tmp_path, token_saving={"phase_calls": {"verify": 2}})
    assert "Verified after patching: yes" in run.report and phases(run, "patch") == 1
    assert "VERIFY ended without the model's verdict (closed)" in run.report


def test_verify_that_hits_the_step_limit_is_decided_the_same_way(tmp_path):
    script = base() + [run_tests(), wander(), wander()] + review_ok() + finalize()
    run = run_with(script, tmp_path, features={"token_budgets": False}, max_steps_per_phase=3)
    assert "Verified after patching: yes" in run.report and phases(run, "patch") == 1
    assert "(step_limit)" in run.report


def test_when_the_last_test_run_failed_the_unfinished_verify_is_a_failed_verification_and_goes_back_to_patch(tmp_path):
    retry = [reply(call("edit_file", path="calc.py", old="return a + b", new="return a + b  # ok")), done("touched it up")]
    second_verify = [run_tests(), done("passes now")]
    script = base() + [run_tests(), wander(), *stall()] + retry + second_verify + review_ok() + finalize()
    run = run_with(script, tmp_path, failing_once())

    assert phases(run, "patch") == 2, "a failing test run is real evidence: back to PATCH"
    retry_prompt = next(t for t in run.prompt_texts() if "This is patch attempt 2" in t)
    assert "Verification did not finish (stalled) and the last test run failed" in retry_prompt
    assert "FAILED tests/test_calc.py::test_add" in retry_prompt, "the failing output is handed to the retry"
    assert "Verified after patching: yes" in run.report


def test_a_later_passing_run_outweighs_an_earlier_failing_one_within_the_phase(tmp_path):
    script = base() + [run_tests(), run_tests(), *stall()] + review_ok() + finalize()
    run = run_with(script, tmp_path, failing_once())
    assert "Verified after patching: yes" in run.report and phases(run, "patch") == 1
    assert "Confidence: **medium**" in run.report, "but the failing run is still on the record, so not high"


def test_with_no_test_run_the_repro_alone_verifies_and_the_confidence_stays_below_high(tmp_path):
    script = base() + [wander(), *stall()] + review_ok() + finalize()
    run = run_with(script, tmp_path)
    assert "Verified after patching: yes" in run.report and phases(run, "patch") == 1
    assert "ran no tests; the harness verified the patch on the repro alone" in run.report
    assert "Confidence: **medium**" in run.report


def test_with_no_repro_and_no_test_run_there_is_nothing_to_go_on_and_verification_fails(tmp_path):
    no_repro = [reply(call("write_repro", path="repro.py", content="print(1)\n")), give_up("cannot reproduce it")]
    tweak = [reply(call("edit_file", path="calc.py", old="return a + b", new="return a + b  # t1")), done("second attempt")]
    script = understand() + localize() + no_repro + good_patch() + [wander(), *stall()] + tweak + [wander(), *stall()] + finalize()
    run = run_with(script, tmp_path, max_patch_attempts=2, max_rollbacks=0)

    assert "Verified after patching: no" in run.report and "Verification still failed after 2 patch attempts" in run.report
    retry_prompt = next(t for t in run.prompt_texts() if "This is patch attempt 2" in t)
    assert "Verification did not finish (stalled) and there is no repro and no test run to go on" in retry_prompt
    assert run.llm.remaining == 0


def test_a_model_that_gives_up_in_verify_is_taken_at_its_word_even_when_the_evidence_would_pass(tmp_path):
    retry = [reply(call("edit_file", path="calc.py", old="return a + b", new="return a + b  # ok")), done("touched it up")]
    script = base() + [run_tests(), give_up("the suite feels flaky")] + retry + [run_tests(), done("fine now")] + review_ok() + finalize()
    run = run_with(script, tmp_path)
    assert phases(run, "patch") == 2, "its own verdict was 'failed'"


# ---- the loop-back is bounded ------------------------------------------------------------------------------------------


def test_a_verify_that_keeps_failing_loops_back_to_patch_at_most_twice_per_approach(tmp_path):
    """max_patch_attempts is 3: the first attempt and two loop-backs, then the run stops trying (no rollbacks here)."""

    def tweak(i: int):
        old = "return a + b" if i == 1 else f"return a + b  # t{i - 1}"
        return [reply(call("edit_file", path="calc.py", old=old, new=f"return a + b  # t{i}")), done(f"attempt {i + 1}")]

    fail = lambda i: [run_tests(), give_up(f"the test still fails ({i})")]  # noqa: E731
    script = base() + fail(1) + tweak(1) + fail(2) + tweak(2) + fail(3) + finalize()
    run = run_with(script, tmp_path, failing_once(9), max_rollbacks=0)

    assert phases(run, "patch") == 3 and phases(run, "verify") == 3, "1 attempt + 2 loop-backs, and no fourth"
    assert "Verification still failed after 3 patch attempts and 0 rollbacks" in run.report
    assert run.llm.remaining == 0, "every scripted reply was used: each VERIFY really ran"


def test_the_second_verify_closes_on_the_evidence_instead_of_reopening_the_fix(tmp_path):
    """VERIFY #1 really fails; after the loop-back VERIFY #2 passes its tests but goes silent: it closes as verified."""
    retry = [reply(call("edit_file", path="calc.py", old="return a + b", new="return a + b  # tidy")), done("second attempt")]
    script = base() + [run_tests(), give_up("the test failed")] + retry + [run_tests(), wander(), *stall()] + review_ok() + finalize()
    run = run_with(script, tmp_path, failing_once())

    assert phases(run, "patch") == 2 and phases(run, "verify") == 2, "one loop-back, and it was not followed by a second"
    assert "Verified after patching: yes" in run.report
    assert "VERIFY ended without the model's verdict (stalled); the harness closed it on the evidence" in run.report


# ---- the confidence rule ---------------------------------------------------------------------------------------------


def state(**fields) -> RunState:
    s = RunState("https://github.com/acme/calc/issues/1")
    s.repro_confirmed, s.verified, s.review = True, True, "approved"
    s.checks = [CheckRun("run_tests", True)]
    for name, value in fields.items():
        setattr(s, name, value)
    return s


REPRO_ONLY = [CheckRun("repro (re-run by the harness)", True, repository_tests=False)]


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, "high"),
        ({"checks": REPRO_ONLY}, "medium"),
        ({"checks": REPRO_ONLY, "verified": False}, "low"),
        ({"checks": REPRO_ONLY, "repro_confirmed": False}, "low"),
    ],
)
def test_verified_on_the_repro_alone_is_never_high(fields, expected):
    assert state(**fields).confidence(has_patch=True) == expected
