"""Per-phase call caps (features.token_budgets): after N calls the harness asks the model to close the phase, and if it does
not the harness closes it itself with a summary it writes from the calls made."""

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


@pytest.fixture(autouse=True)
def scripts_never_look_like_a_loop(monkeypatch):
    """Every fake-model script here must use distinct calls: three identical ones in a row are refused as a loop."""
    from anvil.agent.recovery import PhaseGuard

    real = PhaseGuard.repeated

    def strict(self, *args, **kwargs):
        verdict = real(self, *args, **kwargs)
        assert verdict is None, "the loop detector fired: this test's script repeats the same call (use distinct arguments)"
        return verdict

    monkeypatch.setattr(PhaseGuard, "repeated", strict)


def offered(harness: Harness, index: int) -> set[str]:
    return {t["function"]["name"] for t in harness.llm.calls[index][1]}


def user_messages(harness: Harness) -> list[str]:
    return [m["content"] for m in harness.history if m["role"] == "user"]


# ---- what "a cap of N" means -----------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_a_cap_of_n_gives_the_model_n_calls_of_its_own_and_call_n_plus_one_is_the_forced_close(n):
    """A cap of N gives the model N calls of its own, and call N+1 is the harness's forced close: at most N+1 calls."""
    h = Harness([GREP() for _ in range(n)] + [done("the best so far")], phase_call_caps={"localize": n})
    outcome = h.run(LOCALIZE)

    assert len(outcome.records) == n, "the model's own calls all ran"
    assert h.budget.steps == n + 1, "and the one after them was the close"
    assert outcome.done and outcome.forced is True and outcome.summary == "the best so far"
    assert offered(h, n - 1) > {"phase_done", "give_up"}, f"call {n} still had every tool"
    assert offered(h, n) == {"phase_done", "give_up"}, f"call {n + 1} had only the two ways to close"
    assert any(m.startswith(f"You have used the {n} model calls this phase allows") for m in user_messages(h))


@pytest.mark.parametrize("n", [1, 3])
def test_ending_the_phase_on_call_n_is_not_a_forced_close(n):
    h = Harness([GREP() for _ in range(n - 1)] + [done("on my own")], phase_call_caps={"localize": n})
    outcome = h.run(LOCALIZE)
    assert outcome.done and outcome.forced is False and h.budget.steps == n


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


def test_at_the_close_any_other_tool_call_is_refused_not_run_and_the_harness_closes_the_phase():
    h = Harness([GREP(), GREP(), GREP(), GREP(), done()], **CAP3)
    outcome = h.run(LOCALIZE)

    assert outcome.status is PhaseStatus.CLOSED and outcome.closed and not outcome.done and outcome.forced is True
    assert outcome.summary.startswith("[Written by the harness: the LOCALIZE phase used its 3 calls")
    assert len(outcome.records) == 3, "the fourth grep was not run"
    assert outcome.summary.count("def add") == 3, "the three that were run are in the summary, the refused one is not"
    assert h.budget.steps == 4, "and nothing further was asked of the model"
    last_tool_reply = [m for m in h.history if m["role"] == "tool"][-1]["content"]
    assert "only phase_done or give_up can be called now" in last_tool_reply
    assert not any(m["role"] == "assistant" and m.get("tool_calls") and not any(
        t["role"] == "tool" and t.get("tool_call_id") == c["id"] for t in h.history for c in m["tool_calls"]
    ) for m in h.history), "every call, including the refused one, has its reply: the history stays valid"


def test_a_closing_reply_without_any_tool_call_is_closed_by_the_harness_with_the_models_last_words_and_no_nudge():
    h = Harness([GREP(), GREP(), GREP(), reply(text="I think it is calc.py line 2")], **CAP3)
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.CLOSED and outcome.forced is True
    assert "The model's last note: I think it is calc.py line 2" in outcome.summary
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
    assert calls and outcome.status is PhaseStatus.CLOSED and outcome.forced is True, "the gate refused it, so the harness closed the phase"
    assert "the repro does not fail" in [m["content"] for m in h.history if m["role"] == "tool"][-1]

    accepted = Harness([GREP(), GREP(), GREP(), done("x", repro_cmd="python .anvil/repro.py")], **CAP3)
    assert accepted.run(REPRODUCE, gate=lambda args: None).status is PhaseStatus.DONE


# ---- a model that fights the forced close cannot leave the phase open --------------------------------------------


def READ():  # noqa: N802 - a call every capped phase offers; the path differs each time (three equal calls are a loop)
    return reply(call("read_file", path=f"src/mod{next(_n)}.py"))


FIGHTS = {
    "ignores it and keeps investigating": lambda: [READ()],
    "answers in words only": lambda: [reply(text="I am not done yet, one more look.")],
    "calls a tool that does not exist": lambda: [reply(call("teleport", where="calc.py"))],
    "calls phase_done with an empty summary": lambda: [reply(call("phase_done", summary=""))],
}
CAPPED = [Phase.LOCALIZE, Phase.REPRODUCE, Phase.PATCH, Phase.VERIFY, Phase.REVIEW]


@pytest.mark.parametrize("phase", CAPPED, ids=lambda p: p.value)
@pytest.mark.parametrize("fight", FIGHTS)
def test_a_model_that_fights_the_forced_close_cannot_leave_any_phase_open(fight, phase):
    h = Harness([READ(), READ()] + FIGHTS[fight]() + [done("never asked for")], phase_call_caps={phase.value: 2})
    outcome = h.run(PHASE_SPECS[phase])

    assert outcome.status is PhaseStatus.CLOSED and outcome.closed and outcome.forced is True
    assert h.budget.steps == 3, "two calls of its own, the forced close, and then the harness ended the phase"
    assert h.llm.remaining == 1, "nothing further was asked of the model"
    assert outcome.summary.startswith(f"[Written by the harness: the {phase.value.upper()} phase used its 2 calls")
    assert "Files read: src/mod" in outcome.summary, "the summary says what the phase did"


def test_an_empty_phase_done_that_was_not_forced_is_the_models_own_verdict_and_stays_one():
    h = Harness([READ(), done("")], phase_call_caps={"localize": 5})
    outcome = h.run(LOCALIZE)
    assert outcome.done and not outcome.closed and outcome.forced is False


def test_a_forced_phase_done_with_a_real_summary_is_still_the_models_own_close():
    h = Harness([READ(), READ(), done("calc.py:2 subtracts")], phase_call_caps={"localize": 2})
    outcome = h.run(LOCALIZE)
    assert outcome.done and not outcome.closed and outcome.summary == "calc.py:2 subtracts"


def test_the_harness_says_it_closed_the_phase_so_the_tui_and_the_trace_show_it():
    h = Harness([READ(), READ()] + FIGHTS["ignores it and keeps investigating"](), phase_call_caps={"localize": 2})
    h.run(LOCALIZE)
    texts = [e.data["text"] for e in h.events("message")]
    assert "LOCALIZE used its 2-call cap and the model did not close it; the harness closed it." in texts


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


def test_a_localize_the_model_never_closes_still_hands_the_run_a_summary_and_the_run_goes_on(tmp_path):
    from tests.test_orchestrator import reproduce

    ignoring = [reply(call("grep", pattern=f"def add{i}")) for i in range(4)]  # 3 calls of its own, then it ignores the close
    script = understand() + ignoring + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, token_saving={"phase_calls": {"localize": 3}})

    assert "LOCALIZE used its 3-call cap and the model did not close it: the harness closed it with a summary it wrote itself" in run.report
    assert "The fault was not localized" not in run.report, "a closed LOCALIZE is not a failed one"
    assert "Verified after patching: yes" in run.report
    later = "\n".join(run.prompt_texts())
    assert "[LOCALIZE summary]\n[Written by the harness: the LOCALIZE phase used its 3 calls" in later
    assert "Searched for: 'def add0'" in later, "the next phases are told what was searched"
    assert run.llm.remaining == 0


def test_the_forced_close_conversation_does_not_leak_into_later_phases(tmp_path):
    from tests.test_orchestrator import reproduce

    ignoring = [reply(call("grep", pattern=f"def add{i}")) for i in range(4)]
    script = understand() + ignoring + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, token_saving={"phase_calls": {"localize": 3}})
    messages = next(m for m, _ in run.llm.calls if "Phase: REPRODUCE" in m[0]["content"])  # the first REPRODUCE call
    reproduce_prompt = "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))
    assert "calls are used up" not in reproduce_prompt and "You have used the 3 model calls" not in reproduce_prompt

