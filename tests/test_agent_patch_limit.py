"""A hard total of PATCH attempts per run (max_total_patch_attempts, default 3: the first attempt and two retries).

Rollbacks, approaches, the reviewer's rework round and the sanity retry all count toward it, so no combination of them can keep a
run patching. After the last attempt the run stops patching and finalizes, and report.md says why.
"""

from __future__ import annotations

import pytest

from anvil.agent.orchestrator import load_config
from anvil.agent.settings import AgentSettings
from anvil.sandbox.base import ExecResult
from tests.fakes import FakePipeline, FakeSandbox, call, done, give_up, project_exec, project_files, reply
from tests.test_agent_assert_warning import asserting_patch
from tests.test_agent_patch_sanity import git_world, kickoffs, sanity_line
from tests.test_agent_verify_evidence import base, failing_once, phases, run_tests
from tests.test_orchestrator import execute, finalize, localize, reproduce, review_ok, understand, verify


def fail(i: int):
    """A VERIFY that runs the (failing) tests and reports the failure."""
    return [run_tests(), give_up(f"the test still fails ({i})")]


def tweak(i: int):
    """An edit that keeps the fix and changes a comment, as a retry of an attempt whose tests failed."""
    old = "return a + b" if i == 1 else f"return a + b  # t{i - 1}"
    return [reply(call("edit_file", path="calc.py", old=old, new=f"return a + b  # t{i}")), done(f"attempt {i + 1}")]


def rethink(i: int):
    """An edit made after a rollback: from the original code, with a different fix."""
    return [reply(call("edit_file", path="calc.py", old="return a - b", new=f"return a + b  # r{i}")), done(f"rethought, attempt {i}")]


def run_fake(script, tmp_path, exec_=None, **config):
    fake = FakePipeline(FakeSandbox(project_files(), on_exec=exec_ or failing_once(99)))
    return execute(script, tmp_path, pipeline=fake, **config), fake


def rollbacks(fake) -> int:
    return sum(1 for e in fake.sandbox.events if e[0] == "rollback")


def three_failing_attempts():
    return base() + fail(1) + tweak(1) + fail(2) + tweak(2) + fail(3)


# ---- the default: 3 attempts, then it stops and says why ----------------------------------------------------------------


def test_a_verify_that_always_fails_stops_patching_after_the_third_attempt_and_the_run_finalizes(tmp_path):
    run, _ = run_fake(three_failing_attempts() + finalize(), tmp_path)

    assert phases(run, "patch") == 3 and phases(run, "verify") == 3, "the first attempt and two retries, and no fourth"
    assert run.llm.remaining == 0, "every scripted reply was used, including FINALIZE's: the run finalized normally"
    assert "Patch attempts: 3" in run.report and "Verified after patching: no" in run.report
    assert run.done.type == "done" and "Stopped early" not in run.report, "a limit, not a crash: the run ended in the usual way"


def test_the_report_says_why_it_stopped_and_what_it_delivered(tmp_path):
    run, _ = run_fake(three_failing_attempts() + finalize(), tmp_path)
    limits = run.report.split("Known limitations")[1]

    assert "PATCH stopped at its limit of 3 attempts per run (max_total_patch_attempts), whatever rollbacks or approaches remained" in limits
    assert "the last attempt's patch is delivered" in limits
    assert "Last verification result: the test still fails (3)" in limits, "and the last reason verification gave"
    assert "Verification still failed after 3 patch attempts" in limits
    assert "return a + b" in run.patch, "a patch is still delivered, unverified"
    assert "Confidence: **low**" in run.report


# ---- regardless of rollbacks and approaches --------------------------------------------------------------------------------


def test_rollbacks_cannot_buy_more_attempts(tmp_path):
    """max_patch_attempts=1 rolls back after every failure and max_rollbacks=5 allows it: that used to mean up to 6 attempts."""
    script = base() + fail(1) + rethink(2) + fail(2) + rethink(3) + fail(3) + finalize()
    run, fake = run_fake(script, tmp_path, max_patch_attempts=1, max_rollbacks=5)

    assert phases(run, "patch") == 3 and rollbacks(fake) == 2 and "rollbacks: 2" in run.report
    assert run.llm.remaining == 0
    assert "PATCH stopped at its limit of 3 attempts per run" in run.report


def test_a_second_approach_cannot_buy_more_attempts_either(tmp_path):
    """Two attempts per approach and three rollbacks allowed: attempt 3 is a rethink, and there is no attempt 4."""
    script = base() + fail(1) + tweak(1) + fail(2) + rethink(3) + fail(3) + finalize()
    run, fake = run_fake(script, tmp_path, max_patch_attempts=2, max_rollbacks=3)

    assert phases(run, "patch") == 3 and rollbacks(fake) == 1
    assert run.llm.remaining == 0 and "PATCH stopped at its limit of 3 attempts per run" in run.report


@pytest.mark.parametrize("total", [1, 2, 5])
def test_the_limit_is_the_configured_total(tmp_path, total):
    attempts = base() + fail(1)
    for i in range(1, total):
        attempts += tweak(i) + fail(i + 1)
    run, _ = run_fake(attempts + finalize(), tmp_path, max_total_patch_attempts=total, max_patch_attempts=9, max_rollbacks=0)

    assert phases(run, "patch") == total and run.llm.remaining == 0
    assert f"PATCH stopped at its limit of {total} attempts per run" in run.report


def test_a_run_that_verifies_within_the_limit_is_untouched_by_it(tmp_path):
    script = base() + fail(1) + tweak(1) + [run_tests(), done("passes")] + review_ok() + finalize()
    run, _ = run_fake(script, tmp_path, failing_once(1))
    assert phases(run, "patch") == 2 and "Verified after patching: yes" in run.report
    assert "PATCH stopped at its limit" not in run.report


# ---- the other things that run PATCH count toward the same total ------------------------------------------------------------


def test_the_reviewers_rework_round_is_skipped_when_the_attempts_are_used_up(tmp_path):
    """Verified on the third attempt, so the reviewer's requested changes have no attempt left to be made in."""
    changes = [reply(call("git_diff")), give_up("rename the variable")]
    script = base() + fail(1) + tweak(1) + fail(2) + tweak(2) + [run_tests(), done("passes")] + changes + finalize()
    run, _ = run_fake(script, tmp_path, failing_once(2))

    assert phases(run, "patch") == 3, "no fourth, for the rework"
    limits = run.report.split("Known limitations")[1]
    assert "The reviewer requested changes, but the limit of 3 PATCH attempts per run was already used" in limits
    assert "no rework was made; the reviewed patch is delivered as it is" in limits
    assert "Verified after patching: yes" in run.report and run.llm.remaining == 0


def test_a_rework_that_hits_the_limit_restores_the_reviewed_patch_which_is_the_best_verified_state(tmp_path):
    def first_test_run_passes(cmd, files):
        first_test_run_passes.n = getattr(first_test_run_passes, "n", 0) + (1 if cmd.startswith("pytest") else 0)
        if cmd.startswith("pytest") and first_test_run_passes.n > 1:
            return ExecResult(1, "1 failed\n", "FAILED tests/test_calc.py::test_add\n", False, 0.1)
        return project_exec(cmd, files)

    changes = [reply(call("git_diff")), give_up("add a comment")]
    script = base() + verify() + changes + tweak(1) + fail(2) + tweak(2) + fail(3) + finalize()
    run, fake = run_fake(script, tmp_path, first_test_run_passes)

    assert phases(run, "patch") == 3
    assert "Verified after patching: yes" in run.report, "the reviewed patch was verified, and that is what is delivered"
    assert "# t" not in run.patch and "+    return a + b" in run.patch, "the rework's edits were undone"
    limits = run.report.split("Known limitations")[1]
    assert "The reviewer's requested rework failed verification; the reviewed patch was restored." in limits
    assert "PATCH stopped at its limit of 3 attempts per run" in limits and "the reviewed patch it restored is delivered" in limits
    assert rollbacks(fake) == 1


def test_the_sanity_retry_needs_an_attempt_too_and_says_so_when_there_is_none(tmp_path):
    """Three attempts that give up leave the patch empty; the forced-fix retry would be a fourth attempt, so it is not made."""
    script = understand() + localize() + reproduce() + [give_up("no idea")] * 3
    run = execute(script, tmp_path, pipeline=git_world())

    assert phases(run, "patch") == 3 and kickoffs(run) == []
    assert "no retry: the limit of 3 PATCH attempts per run was already used" in sanity_line(run)
    assert sanity_line(run).startswith("- Patch sanity: FAILED: The patch is empty")
    assert "Patch attempts: 3" in run.report and run.llm.remaining == 0


def test_the_assert_warning_retry_is_skipped_too_but_the_warning_is_still_reported(tmp_path):
    fail_tests = failing_once(99)
    script = understand() + localize() + reproduce() + asserting_patch() + fail(1) + tweak(1) + fail(2) + tweak(2) + fail(3) + finalize()
    run, _ = run_fake(script, tmp_path, fail_tests)

    assert phases(run, "patch") == 3 and kickoffs(run) == []
    assert "WARNING: The patch adds a bare assert" in sanity_line(run) and "## Warnings" in run.report


# ---- the setting ---------------------------------------------------------------------------------------------------------------


def test_the_default_is_three_in_the_code_and_in_the_shipped_config():
    assert AgentSettings().max_total_patch_attempts == 3
    assert AgentSettings.from_mapping({}).max_total_patch_attempts == 3
    assert load_config()["max_total_patch_attempts"] == 3
    assert AgentSettings.from_mapping(load_config()).max_total_patch_attempts == 3


def test_it_can_be_configured():
    assert AgentSettings.from_mapping({"max_total_patch_attempts": 7}).max_total_patch_attempts == 7


@pytest.mark.parametrize("value", [0, -1, 2.5, "3", True, None])
def test_a_value_that_is_not_a_whole_number_of_at_least_one_is_a_config_error(value):
    with pytest.raises(ValueError, match="max_total_patch_attempts"):
        AgentSettings.from_mapping({"max_total_patch_attempts": value})
