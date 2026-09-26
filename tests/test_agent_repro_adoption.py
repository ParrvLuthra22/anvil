"""REPRODUCE without a phase_done: the model's own failing command is adopted as the repro when it qualifies.

Seen with a real model (Qwen3-Coder): it wrote a valid failing repro by its 3rd to 5th call and then never said it was done, in
one run for 25 calls and in two others through a forced close it ignored. The run went on as "not reproduced" (confidence capped
at low) although the evidence was in hand.
"""

from __future__ import annotations

import pytest

from anvil.agent.repro import reports_the_issue
from anvil.sandbox.base import ExecResult
from tests.fakes import (
    REPRO_CMD,
    REPRO_SCRIPT,
    FakePipeline,
    FakeSandbox,
    call,
    give_up,
    project_exec,
    project_files,
    reply,
)
from tests.test_orchestrator import execute, finalize, good_patch, localize, review_ok, understand, verify

CAP3 = {"phase_calls": {"reproduce": 3}}
_n = iter(range(10_000))


def wander():
    return reply(call("grep", pattern=f"def add{next(_n)}"))  # never ends the phase; distinct so it is never a loop


def show_the_bug(cmd: str = REPRO_CMD, script: str = REPRO_SCRIPT):
    return [reply(call("write_repro", path="repro.py", content=script)), reply(call("run_cmd", cmd=cmd))]


def rest():
    return good_patch() + verify() + review_ok() + finalize()


def run_with(script, tmp_path, exec_=None, **config):
    pipeline = FakePipeline(FakeSandbox(project_files(), on_exec=exec_ or project_exec))
    return execute(script, tmp_path, pipeline=pipeline, **config)


def adopted(run) -> bool:
    return "the harness adopted its failing command" in run.report


# ---- adopted -------------------------------------------------------------------------------------------------------


def test_a_failing_command_the_model_never_confirmed_is_adopted_when_the_forced_close_is_ignored(tmp_path):
    script = understand() + localize() + show_the_bug() + [wander(), wander()] + rest()  # 3 own calls, then the ignored close
    run = run_with(script, tmp_path, token_saving=CAP3)

    assert f"Reproduced before patching: yes (`{REPRO_CMD}`)" in run.report
    assert adopted(run) and "REPRODUCE ended without the model's phase_done (closed)" in run.report
    assert "The bug was not reproduced" not in run.report
    told = [e.data["text"] for e in run.of("message") if e.data["role"] == "system" and "adopted its failing command" in e.data["text"]]
    assert len(told) == 1, "the TUI and the trace are told, not only the report"
    assert "Confidence: **high**" in run.report, "the rest of the run went on from a confirmed repro"
    patch_prompt = next(t for t in run.prompt_texts() if "This is patch attempt 1" in t)
    assert f"Repro command: `{REPRO_CMD}`" in patch_prompt and "add(2, 3) == -1" in patch_prompt
    assert run.llm.remaining == 0


def test_the_harness_runs_the_adopted_command_itself_on_the_clean_tree(tmp_path):
    script = understand() + localize() + show_the_bug() + [wander(), wander()] + rest()
    run = run_with(script, tmp_path, token_saving=CAP3)
    reruns = [e for e in run.of("tool_call") if e.data["tool"] == "repro" and e.phase.value == "reproduce"]
    assert [e.data["args"]["cmd"] for e in reruns] == [REPRO_CMD]


@pytest.mark.parametrize(
    "ending, config, tail",
    [
        ("the hard step limit (no caps)", {"features": {"token_budgets": False}, "max_steps_per_phase": 4}, [wander, wander]),
        ("a model that stops calling tools", {"features": {"token_budgets": False}}, [lambda: reply(text="Hmm."), lambda: reply(text="Still thinking.")]),
    ],
    ids=["step limit", "stalled"],
)
def test_it_is_adopted_however_the_phase_ends_without_a_phase_done(tmp_path, ending, config, tail):
    script = understand() + localize() + show_the_bug() + [make() for make in tail] + rest()
    run = run_with(script, tmp_path, **config)
    assert f"Reproduced before patching: yes (`{REPRO_CMD}`)" in run.report, ending
    assert adopted(run)


def test_the_most_recent_qualifying_command_is_the_one_adopted(tmp_path):
    second = "python .anvil/second.py"

    def exec_(cmd, files):
        if cmd == second:
            return ExecResult(1, "", "AssertionError: add(2, 3) == -1 (second script)\n", False, 0.01)
        return project_exec(cmd, files)

    script = (
        understand() + localize() + show_the_bug()
        + [reply(call("write_repro", path="second.py", content="assert False\n")), reply(call("run_cmd", cmd=second))]
        + [wander()] + rest()
    )
    run = run_with(script, tmp_path, exec_, token_saving={"phase_calls": {"reproduce": 5}})
    assert f"Reproduced before patching: yes (`{second}`)" in run.report


def test_a_script_the_model_made_with_a_shell_command_qualifies_like_one_made_with_write_repro(tmp_path):
    make = "printf 'the repro' > .anvil/repro.py"

    def exec_(cmd, files):
        if cmd == make:
            files[".anvil/repro.py"] = REPRO_SCRIPT
            return ExecResult(0, "", "", False, 0.01)
        return project_exec(cmd, files)

    script = understand() + localize() + [reply(call("run_cmd", cmd=make)), reply(call("run_cmd", cmd=REPRO_CMD))]
    script += [wander(), wander()] + rest()
    run = run_with(script, tmp_path, exec_, token_saving=CAP3)
    assert f"Reproduced before patching: yes (`{REPRO_CMD}`)" in run.report and adopted(run)


def reruns_of(run) -> list[str]:
    return [e.data["args"]["cmd"] for e in run.of("tool_call") if e.data["tool"] == "repro" and e.phase.value == "reproduce"]


def test_a_command_that_passed_is_never_run_again_even_if_its_output_mentions_the_bug(tmp_path):
    shows = "python .anvil/show.py"

    def exec_(cmd, files):
        if cmd == shows:
            return ExecResult(0, "add(2, 3) = -1\n", "", False, 0.01)  # prints the wrong sum, exits 0: proves nothing
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [reply(call("write_repro", path="show.py", content="print(1)\n")), reply(call("run_cmd", cmd=shows))]
    script += [wander()] + rest()
    run = run_with(script, tmp_path, exec_, token_saving={"phase_calls": {"reproduce": 5}})
    assert f"Reproduced before patching: yes (`{REPRO_CMD}`)" in run.report
    assert reruns_of(run) == [REPRO_CMD], "the passing command did not use up a re-run"


def test_failures_unrelated_to_the_issue_do_not_use_up_the_reruns_that_a_matching_command_needs(tmp_path):
    unrelated = [f"python .anvil/u{i}.py" for i in range(3)]

    def exec_(cmd, files):
        if cmd in unrelated:
            return ExecResult(1, "", "AssertionError: expected 3 items in the queue\n", False, 0.01)
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [reply(call("run_cmd", cmd=c)) for c in unrelated] + [wander()] + rest()
    run = run_with(script, tmp_path, exec_, token_saving={"phase_calls": {"reproduce": 6}})
    assert f"Reproduced before patching: yes (`{REPRO_CMD}`)" in run.report
    assert reruns_of(run) == [REPRO_CMD], "the three unrelated failures were passed over without being run again"


# ---- not adopted ---------------------------------------------------------------------------------------------------


def not_adopted(run) -> None:
    assert not adopted(run)
    assert "Reproduced before patching: no" in run.report
    assert "The bug was not reproduced" in run.report and "confidence is capped at low" in run.report


def test_a_script_that_passes_shows_nothing_and_is_not_adopted(tmp_path):
    passing = "print('nothing checked')\n"  # the fake project exits 0 for a script that never calls add(2, 3)
    script = understand() + localize() + show_the_bug(script=passing) + [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, token_saving=CAP3))


def test_a_failure_of_the_environment_or_the_script_is_not_the_bug(tmp_path):
    """The command runs a script that was never written where it looks: "can't open file", not a bug in add()."""
    script = understand() + localize() + [reply(call("write_repro", path="other.py", content=REPRO_SCRIPT)), reply(call("run_cmd", cmd=REPRO_CMD))]
    script += [wander(), wander()] + rest()

    def exec_(cmd, files):  # repro.py does not exist: project_exec answers "can't open file"
        return project_exec(cmd, files)

    not_adopted(run_with(script, tmp_path, exec_, token_saving=CAP3))


def test_output_that_has_nothing_to_do_with_the_issue_is_not_adopted(tmp_path):
    unrelated = "python .anvil/unrelated.py"

    def exec_(cmd, files):
        if cmd == unrelated:
            return ExecResult(1, "", "AssertionError: expected 3 items in the queue\n", False, 0.01)
        return project_exec(cmd, files)

    script = understand() + localize() + [reply(call("write_repro", path="unrelated.py", content="assert False\n")), reply(call("run_cmd", cmd=unrelated))]
    script += [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, exec_, token_saving=CAP3))


def test_a_model_that_gave_up_is_taken_at_its_word(tmp_path):
    script = understand() + localize() + show_the_bug() + [give_up("I cannot reproduce it, it may already be fixed")] + rest()
    not_adopted(run_with(script, tmp_path))


def test_a_command_that_fails_again_differently_or_passes_when_the_harness_runs_it_is_not_adopted(tmp_path):
    seen = {"n": 0}

    def flaky(cmd, files):
        if cmd == REPRO_CMD:
            seen["n"] += 1
            if seen["n"] > 1:  # the model saw it fail; the harness's own run passes
                return ExecResult(0, "OK\n", "", False, 0.01)
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, flaky, token_saving=CAP3))


def test_a_command_that_passes_when_the_harness_runs_it_is_not_adopted_even_if_its_output_mentions_the_bug(tmp_path):
    seen = {"n": 0}

    def passes_but_prints_the_sum(cmd, files):
        if cmd == REPRO_CMD:
            seen["n"] += 1
            if seen["n"] > 1:
                return ExecResult(0, "add(2, 3) = 5\n", "", False, 0.01)  # exit 0: it does not fail, whatever it prints
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, passes_but_prints_the_sum, token_saving=CAP3))


def test_a_command_that_times_out_when_the_harness_runs_it_is_not_adopted(tmp_path):
    seen = {"n": 0}

    def hangs_the_second_time(cmd, files):
        if cmd == REPRO_CMD:
            seen["n"] += 1
            if seen["n"] > 1:
                return ExecResult(-1, "add(2, 3) = -1\n", "", True, 120.0)
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, hangs_the_second_time, token_saving=CAP3))


def test_a_command_that_timed_out_is_passed_over_without_using_up_a_rerun(tmp_path):
    slow = "python .anvil/slow.py"

    def exec_(cmd, files):
        if cmd == slow:
            return ExecResult(-1, "add(2, 3) = -1\n", "", True, 120.0)  # started to print the bug, then hung
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [reply(call("write_repro", path="slow.py", content="import time\n")), reply(call("run_cmd", cmd=slow))]
    script += [wander()] + rest()
    run = run_with(script, tmp_path, exec_, token_saving={"phase_calls": {"reproduce": 5}})
    assert f"Reproduced before patching: yes (`{REPRO_CMD}`)" in run.report
    assert reruns_of(run) == [REPRO_CMD]


def test_a_command_that_fails_again_but_not_for_the_reason_in_the_issue_is_not_adopted(tmp_path):
    seen = {"n": 0}

    def other_failure(cmd, files):
        if cmd == REPRO_CMD:
            seen["n"] += 1
            if seen["n"] > 1:  # the model saw the bug; the harness's own run fails for something else entirely
                return ExecResult(1, "", "AssertionError: expected 3 items in the queue\n", False, 0.01)
        return project_exec(cmd, files)

    script = understand() + localize() + show_the_bug() + [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, other_failure, token_saving=CAP3))


def test_a_command_that_is_not_the_script_under_dot_anvil_is_not_adopted(tmp_path):
    inline = 'python -c "from calc import add; assert add(2, 3) == 5"'

    def exec_(cmd, files):
        if cmd == inline:
            return ExecResult(1, "", "AssertionError: add(2, 3) == -1\n", False, 0.01)
        return project_exec(cmd, files)

    script = understand() + localize() + [reply(call("write_repro", path="repro.py", content=REPRO_SCRIPT)), reply(call("run_cmd", cmd=inline))]
    script += [wander(), wander()] + rest()
    not_adopted(run_with(script, tmp_path, exec_, token_saving=CAP3))


def test_at_most_two_commands_are_run_again_while_looking_for_one_to_adopt(tmp_path):
    runs = {}

    def exec_(cmd, files):
        if cmd.startswith("python .anvil/r"):
            runs[cmd] = runs.get(cmd, 0) + 1
            if runs[cmd] == 1:
                return ExecResult(1, "", "AssertionError: add(2, 3) == -1\n", False, 0.01)
            return ExecResult(0, "OK\n", "", False, 0.01)  # every re-run passes
        return project_exec(cmd, files)

    cmds = [f"python .anvil/r{i}.py" for i in range(4)]
    tries = [reply(call("run_cmd", cmd=c)) for c in cmds]
    script = understand() + localize() + [reply(call("write_repro", path="r0.py", content="assert False\n"))] + tries + [wander()] + rest()
    run = run_with(script, tmp_path, exec_, token_saving={"phase_calls": {"reproduce": 6}})

    assert not adopted(run)
    reruns = [e for e in run.of("tool_call") if e.data["tool"] == "repro" and e.phase.value == "reproduce"]
    assert len(reruns) == 2, "the two most recent failing commands were tried again, and no more"


# ---- the matching rule ---------------------------------------------------------------------------------------------

ISSUE = "add() returns the wrong sum\nadd(2, 3) returns -1 instead of 5."


@pytest.mark.parametrize(
    "output, expected",
    [
        ("[exit code 1]\n[stderr]\nAssertionError: add(2, 3) == -1", True),
        ("[stdout]\nadd(2, 3) = -1\nFAILURE: Wrong result\n", True),
        ("ADD(2, 3) RETURNED -1", True),
        ("ModuleNotFoundError: No module named 'calc'", False),
        ("NameError: name 'ad' is not defined", False),
        ("SyntaxError: invalid syntax", False),
        ("bash: python: command not found", False),
        ("python: can't open file '.anvil/repro.py': [Errno 2] No such file or directory", False),
        ("AssertionError: expected 3 items in the queue", False),
        ('Traceback (most recent call last):\n  File "/tmp/add/.anvil/repro.py", line 3, in <module>', False),
        ("", False),
    ],
    ids=["assertion", "printed", "case", "import", "name", "syntax", "not-found", "no-file", "unrelated", "traceback-path-only", "empty"],
)
def test_reports_the_issue(output, expected):
    assert reports_the_issue(output, ISSUE) is expected


def test_an_environment_error_counts_when_the_issue_reports_that_very_error():
    issue = "ImportError: cannot import name 'add' from 'calc'"
    assert reports_the_issue("ImportError: cannot import name 'add' from 'calc'", issue) is True
    assert reports_the_issue("ModuleNotFoundError: No module named 'calc'", issue) is False, "a different one still does not"


def test_a_repository_name_in_a_path_does_not_make_a_match():
    assert reports_the_issue("File \"/work/flask__4045/tests/x.py\", line 2\nKeyError: 'k'", "flask blueprint name may not contain a dot") is False
