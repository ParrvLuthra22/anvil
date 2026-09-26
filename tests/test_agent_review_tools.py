"""REVIEW may run tests (read-only) but cannot edit or run arbitrary commands.

Seen with a real model (Qwen3-Coder): REVIEW spent its calls on run_tests, which the phase did not offer, was refused each
time, and ran out of calls without a verdict.
"""

from __future__ import annotations

import pytest

from anvil.agent.prompts import PHASE_SPECS, WEAK_PHASE_SPECS
from anvil.events import Phase
from tests.fakes import call, done, reply
from tests.test_agent_loop import Harness, tool_results
from tests.test_orchestrator import execute, finalize, good_patch, localize, reproduce, understand, verify

REVIEW = PHASE_SPECS[Phase.REVIEW]


def test_run_tests_is_offered_and_run_in_review():
    h = Harness([reply(call("run_tests")), done("the tests cover the change")])
    outcome = h.run(REVIEW)

    assert outcome.done
    assert [r.tool for r in outcome.records] == ["run_tests"], "it ran (its verdict is the fake project's: the bug is still in the tree)"
    assert not any("not available" in text for text in tool_results(h))


def test_review_still_cannot_edit_or_run_arbitrary_commands():
    h = Harness([reply(call("edit_file", path="calc.py", old="a + b", new="a * b")), reply(call("run_cmd", cmd="ls")), done("ok")])
    outcome = h.run(REVIEW)

    assert outcome.records == [], "neither call was run"
    refusals = [text for text in tool_results(h) if "not available in the review phase" in text]
    assert len(refusals) == 2
    assert "return a - b" in h.sandbox.files["calc.py"], "and nothing was edited"


@pytest.mark.parametrize("table", [PHASE_SPECS, WEAK_PHASE_SPECS], ids=["original prompts", "short prompts"])
def test_both_prompt_tables_offer_run_tests_and_say_that_nothing_can_be_edited(table):
    spec = table[Phase.REVIEW]
    assert spec.tools == ("git_diff", "read_file", "run_tests")
    first_line = next(line for line in spec.system_prompt.splitlines() if line.startswith("Phase: REVIEW"))
    assert "run_tests" in first_line, "the Tools: list in the prompt names it"
    assert "cannot edit" in spec.system_prompt


def test_a_whole_run_whose_reviewer_runs_the_tests_has_no_invalid_calls(tmp_path):
    script = (
        understand() + localize() + reproduce() + good_patch() + verify()
        + [reply(call("git_diff")), reply(call("run_tests")), done("minimal and correct; the tests pass")]
        + finalize()
    )
    run = execute(script, tmp_path)

    assert "Review: approved" in run.report
    review_errors = [e for e in run.of("error") if e.phase is Phase.REVIEW]
    assert review_errors == []
    assert any(e.data["tool"] == "run_tests" for e in run.of("tool_call") if e.phase is Phase.REVIEW)
    assert run.llm.remaining == 0
