"""The 4th identical call in a phase is refused, back-to-back or not; so is the 4th edit that fails on the same old_string.

Seen with a real model (Qwen3-Coder): the same read_file range five times in one phase, and the same failing edit_file twelve
times (`old` with the wrong indentation) between reads. The loop detector only sees repeats that are back to back.
"""

from __future__ import annotations

import pytest

from anvil.agent.loop import PhaseStatus
from anvil.agent.prompts import PHASE_SPECS
from anvil.events import Phase
from tests.fakes import call, done, reply
from tests.test_agent_loop import LOCALIZE, PATCH, Harness, tool_results

REFUSAL = "You already did this three times: act or call phase_done."
READ = lambda: reply(call("read_file", path="calc.py"))  # noqa: E731
_n = iter(range(10_000))


def other():
    """A call that differs from every other one, so it is never itself a repeat and never forms a loop pattern."""
    return reply(call("grep", pattern=f"def add{next(_n)}"))


def bad_edit(new: str = "return a + b", old: str = "        return a - b   # wrong indentation"):
    return reply(call("edit_file", path="calc.py", old=old, new=new))


def loop_errors(h: Harness):
    return [e for e in h.events("error") if e.data["kind"] == "loop"]


# ---- identical calls that are not back to back -----------------------------------------------------------------------


def test_the_fourth_identical_call_is_refused_even_with_other_calls_between():
    h = Harness([READ(), other(), READ(), other(), READ(), other(), READ(), done("enough")])
    outcome = h.run(LOCALIZE)

    assert outcome.done
    assert [r.tool for r in outcome.records].count("read_file") == 3, "three ran; the fourth was not run"
    refused = [t for t in tool_results(h) if REFUSAL in t]
    assert len(refused) == 1 and refused[0].startswith("Not run:")


def test_the_refusal_is_announced_as_a_loop_event_so_the_trace_shows_it():
    h = Harness([READ(), other(), READ(), other(), READ(), other(), READ(), done("x")])
    h.run(LOCALIZE)
    (event,) = loop_errors(h)
    assert "read_file" in event.data["message"] and "refused" in event.data["message"]


def test_identical_means_the_same_tool_with_the_same_arguments():
    ranges = [reply(call("read_file", path="calc.py", start=i, end=i + 5)) for i in range(6)]
    h = Harness(ranges + [done("x")])
    outcome = h.run(LOCALIZE)
    assert [r.tool for r in outcome.records] == ["read_file"] * 6 and not any(REFUSAL in t for t in tool_results(h))


def test_argument_order_does_not_hide_a_repeat():
    a = reply(call("read_file", path="calc.py", start=1, end=9))
    b = reply(call("read_file", end=9, start=1, path="calc.py"))
    h = Harness([a, other(), b, other(), a, other(), b, done("x")])
    outcome = h.run(LOCALIZE)
    assert [r.tool for r in outcome.records].count("read_file") == 3 and sum(REFUSAL in t for t in tool_results(h)) == 1


def test_a_call_the_model_keeps_making_keeps_being_refused_but_never_ends_the_phase_by_itself():
    script = [READ(), other(), READ(), other(), READ(), other()] + [x for _ in range(3) for x in (READ(), other())] + [done("finally")]
    h = Harness(script, token_budgets=False)
    outcome = h.run(LOCALIZE)
    assert outcome.status is PhaseStatus.DONE and outcome.summary == "finally"
    assert sum(REFUSAL in t for t in tool_results(h)) == 3, "the 4th, 5th and 6th were all refused"


def test_control_tools_are_never_refused():
    h = Harness([done("first")])
    assert h.run(LOCALIZE).done


def test_the_count_is_per_phase_a_new_phase_starts_again():
    h = Harness([READ(), other(), READ(), other(), READ(), done("one"), READ(), done("two")])
    assert h.run(LOCALIZE).done
    second = h.run(LOCALIZE)
    assert second.done and [r.tool for r in second.records] == ["read_file"], "the same call is allowed again in the next phase"


def test_the_back_to_back_loop_detector_still_works_as_before():
    h = Harness([READ(), READ(), READ(), done("x")])
    h.run(LOCALIZE)
    assert any("Strike 1 of 3" in t for t in tool_results(h)), "three in a row is still refused by the loop detector, with a strike"


# ---- failed edits with the same old_string ---------------------------------------------------------------------------


def test_the_fourth_edit_on_an_old_string_that_failed_three_times_is_refused_even_if_new_differs():
    edits = [bad_edit("return a + b"), other(), bad_edit("return b + a"), other(), bad_edit("return sum((a, b))"), other()]
    h = Harness(edits + [bad_edit("return a + b  # v4"), done("x")])
    outcome = h.run(PATCH)

    assert [r.tool for r in outcome.records].count("edit_file") == 3, "the fourth edit was never attempted"
    refused = [t for t in tool_results(h) if REFUSAL in t]
    assert len(refused) == 1
    assert "Every earlier edit with this old_string failed" in refused[0] and "read_file" in refused[0], "and it says what to do about it"


def test_it_counts_the_old_string_whatever_the_path():
    edits = []
    for path in ("calc.py", "other.py", "third.py"):
        edits += [reply(call("edit_file", path=path, old="X marks the spot", new="y")), other()]
    h = Harness(edits + [reply(call("edit_file", path="calc.py", old="X marks the spot", new="z")), done("x")])
    outcome = h.run(PATCH)
    assert [r.tool for r in outcome.records].count("edit_file") == 3 and sum(REFUSAL in t for t in tool_results(h)) == 1


def test_edits_that_succeed_are_never_refused_however_many_there_are():
    flip = [reply(call("edit_file", path="calc.py", old="return a - b", new="return a + b")), reply(call("edit_file", path="calc.py", old="return a + b", new="return a - b"))]
    h = Harness([x for i in range(4) for x in (flip[i % 2], other())] + [done("x")])
    outcome = h.run(PATCH)
    assert not any(REFUSAL in t for t in tool_results(h))
    assert [r.tool for r in outcome.records].count("edit_file") == 4


# ---- a change to the code starts the count again -------------------------------------------------------------------------


def test_a_successful_edit_clears_the_counts_so_re_running_the_repro_after_each_edit_is_fine():
    run = reply(call("run_cmd", cmd="python .anvil/repro.py"))
    script = []
    for i in range(5):  # the normal PATCH loop: edit, run the repro, edit, run the repro ...
        old, new = ("return a - b", "return a + b") if i % 2 == 0 else ("return a + b", "return a - b")
        script += [reply(call("edit_file", path="calc.py", old=old, new=new)), run]
    h = Harness(script + [done("x")])
    outcome = h.run(PATCH)
    assert [r.tool for r in outcome.records].count("run_cmd") == 5 and not any(REFUSAL in t for t in tool_results(h))


def test_a_successful_write_repro_clears_the_counts_too_the_reproduce_loop_is_write_run_fix_run():
    from anvil.agent.repro import WriteReproTool
    from tests.fakes import FakeRegistry, default_tools

    run = reply(call("run_cmd", cmd="python .anvil/repro.py"))
    script = []
    for i in range(5):
        script += [reply(call("write_repro", path="repro.py", content=f"print({i})\n")), run]
    h = Harness(script + [done("x")], registry=FakeRegistry([*default_tools(), WriteReproTool()]))
    outcome = h.run(PHASE_SPECS[Phase.REPRODUCE], gate=lambda args: None)
    assert [r.tool for r in outcome.records].count("run_cmd") == 5 and not any(REFUSAL in t for t in tool_results(h))


def test_without_an_edit_between_them_the_same_repro_run_is_refused_the_fourth_time():
    run = reply(call("run_cmd", cmd="python .anvil/repro.py"))
    h = Harness([run, other(), run, other(), run, other(), run, done("x")])
    outcome = h.run(PATCH)
    assert [r.tool for r in outcome.records].count("run_cmd") == 3 and sum(REFUSAL in t for t in tool_results(h)) == 1


def test_a_successful_edit_also_clears_the_failed_edit_counts():
    failing = [bad_edit("a"), other(), bad_edit("b"), other(), bad_edit("c"), other()]
    ok_edit = [reply(call("edit_file", path="calc.py", old="return a - b", new="return a + b")), other()]
    h = Harness(failing + ok_edit + [bad_edit("d"), done("x")])
    outcome = h.run(PATCH)
    assert [r.tool for r in outcome.records].count("edit_file") == 5 and not any(REFUSAL in t for t in tool_results(h))


def test_a_failed_edit_does_not_clear_the_counts():
    failing = [bad_edit("a"), READ(), bad_edit("b"), READ(), bad_edit("c"), READ()]
    h = Harness(failing + [READ(), bad_edit("d"), done("x")])
    h.run(PATCH)
    assert sum(REFUSAL in t for t in tool_results(h)) == 2, "the 4th read and the 4th edit"


# ---- what the model is told ------------------------------------------------------------------------------------------------


def test_the_refusal_is_short_and_says_what_to_do():
    h = Harness([READ(), other(), READ(), other(), READ(), other(), READ(), done("x")])
    h.run(LOCALIZE)
    (text,) = [t for t in tool_results(h) if REFUSAL in t]
    assert len(text) < 200 and "phase_done" in text
