"""The confidence says what the evidence supports: repository tests and an approved review earn 0.90, nothing less honest.

* A REVIEW that never finished (a weak model that stalls) is not evidence against the patch: it costs nothing, it just does not
  earn the approval bonus.
* A patch verified only on the model's own repro, with no repository test run at all, is capped at 0.50 and says so.
* `run_cmd pytest ...` (and the other runners) count as repository tests; a run of the model's own `.anvil/` scripts does not.
"""

from __future__ import annotations

import pytest

from anvil.agent.state import (
    REVIEW_APPROVED,
    REVIEW_CHANGES_REQUESTED,
    REVIEW_NOT_RUN,
    CheckRun,
    RunState,
)
from tests.fakes import REPRO_CMD, call, done, give_up, reply
from tests.test_orchestrator import execute, finalize, good_patch, localize, reproduce, review_ok, understand, verify

UNFINISHED = "inconclusive (stalled)"
NO_TESTS_NOTE = "no repository tests were run"


def state(**overrides) -> RunState:
    base = dict(repro_confirmed=True, verified=True, review=REVIEW_APPROVED, checks=[CheckRun("pytest", True)])
    return RunState(issue_url="u", **{**base, **overrides})


REPRO_CHECK = CheckRun(f"repro `{REPRO_CMD}` (re-run by the harness)", True, repository_tests=False)


# ---- the score for each combination of evidence ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, score, label",
    [
        ({}, 0.90, "high"),
        ({"review": UNFINISHED}, 0.75, "medium"),
        ({"review": REVIEW_NOT_RUN}, 0.75, "medium"),
        ({"review": REVIEW_CHANGES_REQUESTED}, 0.60, "medium"),
        ({"checks": [CheckRun("pytest", False)]}, 0.60, "medium"),
        ({"checks": [CheckRun("pytest", True), CheckRun("pytest tests/other", False)]}, 0.60, "medium"),
        ({"halted": "budget"}, 0.60, "medium"),
        ({"halted": "budget", "review": UNFINISHED}, 0.60, "medium"),
        ({"checks": [REPRO_CHECK]}, 0.50, "medium"),
        ({"checks": []}, 0.50, "medium"),
        ({"checks": [REPRO_CHECK], "review": UNFINISHED}, 0.50, "medium"),
        ({"checks": [REPRO_CHECK], "review": REVIEW_CHANGES_REQUESTED}, 0.50, "medium"),
        ({"checks": [REPRO_CHECK], "halted": "budget"}, 0.50, "medium"),
        ({"checks": [CheckRun("repro", False, repository_tests=False)]}, 0.50, "medium"),
        ({"checks": [REPRO_CHECK, CheckRun("pytest", True)]}, 0.90, "high"),
        ({"checks": [CheckRun("repro (re-run by the harness)", False, repository_tests=False), CheckRun("pytest", True)]}, 0.60, "medium"),
        ({"verified": False}, 0.30, "low"),
        ({"repro_confirmed": False}, 0.30, "low"),
        ({"sanity_failed": True}, 0.30, "low"),
        ({"verified": False, "review": UNFINISHED, "checks": []}, 0.30, "low"),
    ],
)
def test_the_score_and_label_for_each_combination_of_evidence(overrides, score, label):
    s = state(**overrides)
    assert s.confidence_score(True) == pytest.approx(score)
    assert s.confidence(True) == label


def test_no_patch_is_no_confidence():
    assert state().confidence_score(False) == 0.0 and state().confidence(False) == "none"


def test_ninety_percent_needs_repository_tests_that_passed_and_an_approved_review_and_nothing_else_reaches_it():
    for review in (REVIEW_APPROVED, UNFINISHED, REVIEW_NOT_RUN, REVIEW_CHANGES_REQUESTED):
        for tests in ([], [REPRO_CHECK], [CheckRun("pytest", True)], [CheckRun("pytest", False)], [REPRO_CHECK, CheckRun("pytest", True)]):
            for halted in ("", "budget"):
                s = state(review=review, checks=tests, halted=halted)
                earns = review == REVIEW_APPROVED and any(c.repository_tests for c in tests) and all(c.passed for c in tests) and not halted
                assert (s.confidence_score(True) == pytest.approx(0.90)) == earns, (review, tests, halted)


def test_an_unfinished_review_costs_nothing_it_only_lacks_the_bonus_and_it_is_better_than_a_review_that_asked_for_changes():
    unfinished, requested = state(review=UNFINISHED).confidence_score(True), state(review=REVIEW_CHANGES_REQUESTED).confidence_score(True)
    approved = state().confidence_score(True)
    assert requested < unfinished < approved
    assert unfinished == pytest.approx(0.75) and approved - unfinished == pytest.approx(0.15)


# ---- what report.md is told to say ----------------------------------------------------------------------------------------


def test_the_notes_for_a_patch_verified_on_the_repro_alone():
    (note,) = state(checks=[REPRO_CHECK]).confidence_notes(True)
    assert NO_TESTS_NOTE in note and "0.50" in note and "repro" in note


def test_the_notes_for_an_unfinished_review_after_repository_tests_passed():
    (note,) = state(review=UNFINISHED).confidence_notes(True)
    assert "did not finish" in note and UNFINISHED in note and "repository's tests" in note and "no changes were requested" in note
    assert "does not lower the confidence" in note and "0.90" in note


def test_there_is_nothing_to_explain_when_the_review_approved_or_asked_for_changes_or_a_test_failed():
    assert state().confidence_notes(True) == []
    assert state(review=REVIEW_CHANGES_REQUESTED).confidence_notes(True) == []
    assert state(review=UNFINISHED, checks=[CheckRun("pytest", False)]).confidence_notes(True) == []
    assert state(review=UNFINISHED, halted="budget").confidence_notes(True) == []


def test_without_repository_tests_only_the_cap_is_explained_not_the_review():
    notes = state(review=UNFINISHED, checks=[REPRO_CHECK]).confidence_notes(True)
    assert len(notes) == 1 and NO_TESTS_NOTE in notes[0]


def test_an_unverified_or_absent_patch_has_no_notes():
    assert state(verified=False, checks=[]).confidence_notes(True) == []
    assert state(checks=[REPRO_CHECK]).confidence_notes(False) == []


# ---- whole runs ------------------------------------------------------------------------------------------------------------


def repro_only_verify():
    return [reply(call("run_cmd", cmd=REPRO_CMD)), done("the repro passes")]


def run_cmd_verify(cmd="pytest tests/test_calc.py -q"):
    return [reply(call("run_cmd", cmd=cmd)), done("the tests pass")]


def outcome_line(run) -> str:
    return next(line for line in run.report.splitlines() if line.startswith("- Confidence:"))


def test_a_run_verified_by_run_tests_and_approved_is_ninety_percent(tmp_path):
    run = execute(understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + finalize(), tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.90)
    assert outcome_line(run) == "- Confidence: **high** (0.90)" and NO_TESTS_NOTE not in run.report


def test_a_run_cmd_pytest_in_verify_counts_as_repository_tests(tmp_path):
    run = execute(understand() + localize() + reproduce() + good_patch() + run_cmd_verify() + review_ok() + finalize(), tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.90)
    assert "- pass: run_cmd pytest tests/test_calc.py -q" in run.report and NO_TESTS_NOTE not in run.report


def test_a_failing_run_cmd_pytest_is_a_failing_test_run(tmp_path):
    from anvil.sandbox.base import ExecResult
    from tests.fakes import FakePipeline, FakeSandbox, project_exec, project_files

    def flaky(cmd, files):
        if cmd.startswith("pytest"):
            return ExecResult(1, "1 failed\n", "", False, 0.1)
        return project_exec(cmd, files)

    pipeline = FakePipeline(FakeSandbox(project_files(), on_exec=flaky))
    run = execute(understand() + localize() + reproduce() + good_patch() + run_cmd_verify() + review_ok() + finalize(), tmp_path, pipeline=pipeline)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.60)
    assert "- FAIL: run_cmd pytest tests/test_calc.py -q" in run.report


def test_a_run_verified_on_the_models_own_repro_only_is_capped_and_says_no_repository_tests_were_run(tmp_path):
    run = execute(understand() + localize() + reproduce() + good_patch() + repro_only_verify() + review_ok() + finalize(), tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.50), "an approved review does not lift the cap"
    assert outcome_line(run) == "- Confidence: **medium** (0.50)"
    assert NO_TESTS_NOTE in run.report and "Capped at 0.50" in run.report
    assert "Verified after patching: yes" in run.report


def test_a_verify_that_ends_at_once_with_no_tool_call_is_the_same(tmp_path):
    run = execute(understand() + localize() + reproduce() + good_patch() + [done("looks fine")] + review_ok() + finalize(), tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.50) and NO_TESTS_NOTE in run.report


def test_running_the_models_own_script_under_pytest_is_still_only_the_repro(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + run_cmd_verify("pytest .anvil/repro.py") + review_ok() + finalize()
    run = execute(script, tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.50) and NO_TESTS_NOTE in run.report


def test_a_verify_that_stalls_after_a_run_cmd_pytest_was_verified_on_repository_tests(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + [reply(call("run_cmd", cmd="pytest tests/test_calc.py -q"))] + [reply(text="hmm")] * 3 + review_ok() + finalize()
    run = execute(script, tmp_path)
    assert "Verified after patching: yes" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.90) and NO_TESTS_NOTE not in run.report


def test_a_verify_that_stalls_with_only_the_repro_is_capped(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + [reply(call("run_cmd", cmd=REPRO_CMD))] + [reply(text="hmm")] * 3 + review_ok() + finalize()
    run = execute(script, tmp_path)
    assert "Verified after patching: yes" in run.report and "verified the patch on the repro alone" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.50) and NO_TESTS_NOTE in run.report


def test_a_review_that_never_finishes_after_repository_tests_passed_costs_nothing_and_the_report_says_so(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + verify() + [reply(text="hmm")] * 3 + finalize()
    run = execute(script, tmp_path)
    assert "Review: inconclusive (stalled)" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.75)
    assert outcome_line(run) == "- Confidence: **medium** (0.75)"
    lines = run.report.splitlines()
    note = lines[lines.index(outcome_line(run)) + 1]
    assert note.startswith("  - The self-review did not finish") and "no changes were requested" in note and "does not lower the confidence" in note


def test_a_review_that_never_finishes_after_a_repro_only_verify_is_capped_not_promoted(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + repro_only_verify() + [reply(text="hmm")] * 3 + finalize()
    run = execute(script, tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.50)
    assert "self-review did not finish, but" not in run.report and NO_TESTS_NOTE in run.report


def test_a_review_that_asked_for_changes_is_still_medium_and_has_no_unfinished_note(tmp_path):
    script = (
        understand() + localize() + reproduce() + good_patch() + verify()
        + [reply(call("git_diff")), give_up("Please add a docstring.")]
        + [reply(call("edit_file", path="calc.py", old="def add(a, b):", new='def add(a, b):\n    """Add."""')), done("added")]
        + verify() + finalize()
    )
    run = execute(script, tmp_path)
    assert "Review: changes requested" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.60) and "did not finish" not in run.report.split("## Known limitations")[0]

