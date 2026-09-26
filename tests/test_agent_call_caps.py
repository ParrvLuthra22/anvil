"""Per-phase call caps (features.token_budgets): after N calls the harness makes the phase close with phase_done."""

from __future__ import annotations

import pytest

from anvil.agent.loop import PhaseStatus
from anvil.agent.prompts import PHASE_SPECS, closing_message
from anvil.events import Phase
from tests.fakes import call, done, give_up, reply
from tests.test_agent_loop import LOCALIZE, PATCH, UNDERSTAND, Harness
from tests.test_orchestrator import execute, finalize, good_patch, localize, review_ok, understand, verify

REPRODUCE = PHASE_SPECS[Phase.REPRODUCE]
_n = iter(range(10_000))


def GREP():  # noqa: N802 - a call that never ends the phase; the pattern differs each time (three equal calls are a loop)
    return reply(call("grep", pattern=f"def add{next(_n)}"))
CAP3 = {"phase_call_caps": {"localize": 3, "reproduce": 3, "patch": 3}}


def offered(harness: Harness, index: int) -> set[str]:
    return {t["function"]["name"] for t in harness.llm.calls[index][1]}


def user_messages(harness: Harness) -> list[str]:
    return [m["content"] for m in harness.history if m["role"] == "user"]


# ---- the forced close ------------------------------------------------------------------------------------------


def test_after_the_cap_the_next_call_is_a_forced_close_offering_only_the_two_control_tools():
    h = Harness([GREP(), GREP(), GREP(), done("calc.py:2, a - b")], **CAP3)
    outcome = h.run(LOCALIZE)

    assert outcome.status is PhaseStatus.DONE and outcome.forced is True
    assert outcome.summary == "calc.py:2, a - b"
    assert h.budget.steps == 4, "three calls of its own, then the closing one"
    assert offered(h, 0) > {"phase_done", "give_up"} and offered(h, 2) > {"phase_done", "give_up"}
    assert offered(h, 3) == {"phase_done", "give_up"}


def test_the_model_is_told_why_and_what_to_put_in_the_closing_call():
    h = Harness([GREP(), GREP(), GREP(), done()], **CAP3)
    h.run(LOCALIZE)
    (closing,) = [m for m in user_messages(h) if m.startswith("You have used the 3 model calls")]
    assert "phase_done(summary)" in closing and "file:line locations ranked by likelihood" in closing
    assert closing in [e.data["text"] for e in h.events("message")], "the TUI sees it too"


def test_a_phase_that_ends_before_the_cap_is_not_forced_and_is_not_told_anything():
    h = Harness([GREP(), done("found it")], **CAP3)
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.DONE and outcome.forced is False and h.budget.steps == 2
    assert not any("model calls this phase allows" in m for m in user_messages(h))


def test_a_phase_that_ends_exactly_on_its_last_call_is_not_forced():
    h = Harness([GREP(), GREP(), done("on the third call")], **CAP3)
    outcome = h.run(LOCALIZE)
    assert outcome.done and outcome.forced is False and h.budget.steps == 3


def test_at_the_close_any_other_tool_call_is_refused_not_run_and_answered_in_the_history():
    h = Harness([GREP(), GREP(), GREP(), GREP(), done()], **CAP3)
    outcome = h.run(LOCALIZE)

    assert outcome.status is PhaseStatus.STEP_LIMIT and outcome.forced is True
    assert "closing call did not end the phase" in outcome.summary and "call cap (3)" in outcome.summary
    assert len(outcome.records) == 3, "the fourth grep was not run"
    assert h.budget.steps == 4, "and nothing further was asked of the model"
    last_tool_reply = [m for m in h.history if m["role"] == "tool"][-1]["content"]
    assert "only phase_done or give_up can be called now" in last_tool_reply
    assert not any(m["role"] == "assistant" and m.get("tool_calls") and not any(
        t["role"] == "tool" and t.get("tool_call_id") == c["id"] for t in h.history for c in m["tool_calls"]
    ) for m in h.history), "every call, including the refused one, has its reply: the history stays valid"


def test_a_closing_reply_without_any_tool_call_ends_the_phase_at_the_limit_without_a_nudge():
    h = Harness([GREP(), GREP(), GREP(), reply(text="I think it is calc.py line 2")], **CAP3)
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.STEP_LIMIT and outcome.forced is True
    assert "no tool call" in outcome.summary and "calc.py line 2" in outcome.summary
    assert h.budget.steps == 4


def test_give_up_is_also_a_way_to_close():
    h = Harness([GREP(), GREP(), GREP(), give_up("nothing found")], **CAP3)
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.GAVE_UP and outcome.forced is True


def test_the_gate_still_vets_a_forced_phase_done():
    calls = []

    def gate(args):
        calls.append(args)
        return "the repro does not fail"

    h = Harness([GREP(), GREP(), GREP(), done("x", repro_cmd="python .anvil/repro.py")], **CAP3)
    outcome = h.run(REPRODUCE, gate=gate)
    assert calls and outcome.status is PhaseStatus.STEP_LIMIT and outcome.forced is True
    assert "the repro does not fail" in [m["content"] for m in h.history if m["role"] == "tool"][-1]

    accepted = Harness([GREP(), GREP(), GREP(), done("x", repro_cmd="python .anvil/repro.py")], **CAP3)
    assert accepted.run(REPRODUCE, gate=lambda args: None).status is PhaseStatus.DONE


# ---- where it does not apply -----------------------------------------------------------------------------------


def test_a_cap_at_or_above_the_hard_step_limit_never_forces_anything():
    h = Harness([GREP() for _ in range(5)], max_steps_per_phase=3, phase_call_caps={"localize": 8})
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.STEP_LIMIT and outcome.summary == "phase step limit (3) reached"
    assert outcome.forced is False and h.budget.steps == 3


def test_with_the_feature_off_no_cap_applies():
    h = Harness([GREP() for _ in range(12)] + [done()], token_budgets=False, phase_call_caps={"localize": 3})
    outcome = h.run(LOCALIZE)
    assert outcome.done and outcome.forced is False and h.budget.steps == 13


def test_the_shipped_caps_apply_to_a_phase_by_name():
    h = Harness([GREP() for _ in range(8)] + [done("closed")])  # default settings: localize 8
    outcome = h.run(LOCALIZE)
    assert outcome.done and outcome.forced is True and h.budget.steps == 9
    h = Harness([GREP() for _ in range(15)] + [done("closed")])  # patch 15
    assert h.run(PATCH).forced is True and h.budget.steps == 16


def test_understand_gets_one_call_and_then_a_close():
    ends = Harness([reply(text="The issue says add() is wrong.")])
    assert ends.run(UNDERSTAND).status is PhaseStatus.DONE and ends.budget.steps == 1
    wanders = Harness([GREP(), done("the issue in my words")])
    outcome = wanders.run(UNDERSTAND)
    assert outcome.done and outcome.forced is True and wanders.budget.steps == 2


# ---- the closing message ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase, expected",
    [
        (Phase.UNDERSTAND, "in your own words"),
        (Phase.LOCALIZE, "ranked by likelihood"),
        (Phase.REPRODUCE, "repro_cmd"),
        (Phase.PATCH, "what you changed and why"),
        (Phase.VERIFY, "only if the repro and the relevant tests passed"),
        (Phase.REVIEW, "listing exactly what to change"),
    ],
)
def test_each_phase_is_told_what_its_closing_summary_should_hold(phase, expected):
    message = closing_message(phase, 7)
    assert message.startswith("You have used the 7 model calls this phase allows.") and expected in message
    assert "phase_done(summary)" in message


# ---- through the orchestrator ----------------------------------------------------------------------------------


def wandering_localize(n: int):
    """``n`` calls that do not end the phase, then the closing phase_done."""
    return [reply(call("grep", pattern=f"def add{i}")) for i in range(n)] + [done("calc.py:2 uses a - b")]


def test_the_report_says_which_phase_the_harness_closed(tmp_path):
    from tests.test_orchestrator import reproduce

    script = understand() + wandering_localize(3) + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, token_saving={"phase_calls": {"localize": 3}})
    assert "LOCALIZE used its 3-call cap and was closed by the harness with what it had (done)." in run.report
    assert "Verified after patching: yes" in run.report, "the rest of the run went on from the forced summary"


def test_a_phase_within_its_cap_leaves_no_note(tmp_path):
    from tests.test_orchestrator import happy

    run = execute(happy(), tmp_path)
    assert "call cap" not in run.report
