"""End-to-end runs of run_harness against scripted models and in-memory fakes (no network, no real LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from anvil.agent.orchestrator import run_harness
from anvil.events import AgentEvent, EventBus
from anvil.llm.errors import LLMError
from anvil.sandbox.base import ExecResult
from tests.fakes import (
    FIX,
    ISSUE_URL,
    REPRO_CMD,
    REPRO_SCRIPT,
    FakePipeline,
    FakeRegistry,
    FakeSandbox,
    RecordingLLM,
    StashStyleSandbox,
    call,
    crashing,
    default_tools,
    done,
    give_up,
    project_exec,
    project_files,
    reply,
)

# ---- script building blocks (each list is the model's replies for one phase) ----------------


def understand():
    return [done("add() subtracts instead of adding")]


def localize():
    return [
        reply(call("grep", pattern="def add")),
        reply(call("read_file", path="calc.py")),
        done("calc.py:2 uses a - b"),
    ]


def reproduce(content=REPRO_SCRIPT):
    return [
        reply(call("write_repro", path="repro.py", content=content)),
        reply(call("run_cmd", cmd=REPRO_CMD)),
        done("the repro fails", repro_cmd=REPRO_CMD),
    ]


def good_patch(old=FIX["old"]):
    edit = {**FIX, "old": old}
    return [reply(call("edit_file", **edit)), reply(call("run_cmd", cmd=REPRO_CMD)), done("changed - to +")]


def wrong_patch(old, new, note):
    return [reply(call("edit_file", path="calc.py", old=old, new=new)), done(note)]


def verify():
    return [reply(call("run_tests", target="tests/test_calc.py")), done("repro and tests pass")]


def review_ok():
    return [reply(call("git_diff")), done("minimal and correct")]


def finalize():
    return [done("Fixed the operator in add().")]


def happy():
    return understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + finalize()


HAPPY_STEPS = 15
PHASE_ORDER = ["ingest", "profile", "understand", "localize", "reproduce", "patch", "verify", "review", "finalize"]


# ---- running -------------------------------------------------------------------------------


@dataclass
class Run:
    events: list[AgentEvent]
    llm: RecordingLLM
    pipeline: FakePipeline
    out: Path

    @property
    def patch(self) -> str:
        return (self.out / "patch.diff").read_text()

    @property
    def report(self) -> str:
        return (self.out / "report.md").read_text()

    def of(self, kind: str) -> list[AgentEvent]:
        return [e for e in self.events if e.type == kind]

    @property
    def done(self) -> AgentEvent:
        (event,) = self.of("done")
        return event

    @property
    def phases(self) -> list[str]:
        return [e.data["name"] for e in self.of("phase")]

    def prompt_texts(self) -> list[str]:
        """Every message content the model was ever shown."""
        return [m.get("content") or "" for messages, _ in self.llm.calls for m in messages]

    def last_tool_result(self) -> str:
        return [m["content"] for m in self.llm.calls[-1][0] if m["role"] == "tool"][-1]


def execute(script, tmp_path, *, pipeline=None, llm=None, **config) -> Run:
    out = tmp_path / "out"
    bus = EventBus()
    queue = bus.subscribe()
    llm = llm or RecordingLLM(script)
    pipeline = pipeline or FakePipeline()
    run_harness(ISSUE_URL, {"output_dir": str(out), **config}, bus, llm=llm, pipeline=pipeline)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return Run(events, llm, pipeline, out)


# ---- happy path ----------------------------------------------------------------------------


def test_happy_path_runs_every_phase_and_produces_a_verified_patch(tmp_path):
    run = execute(happy(), tmp_path)

    assert run.phases == PHASE_ORDER
    assert run.llm.remaining == 0
    assert "+    return a + b" in run.patch and "-    return a - b" in run.patch

    report = run.report
    assert "Confidence: **high** (0.90)" in report
    assert "yes (`python .anvil/repro.py`)" in report
    assert "`calc.py` (+1 / -1)" in report
    assert "pass: run_tests tests/test_calc.py" in report
    assert "Fixed the operator in add()." in report
    assert "add() subtracts instead of adding" in report
    assert "None noted." in report


def test_the_repro_script_never_reaches_the_patch_but_does_exist_in_the_sandbox(tmp_path):
    run = execute(happy(), tmp_path)
    assert ".anvil" not in run.patch
    assert ".anvil/repro.py" in run.pipeline.sandbox.files
    assert ".anvil" not in run.report.split("## Files changed")[1].split("## Tests run")[0]


def test_done_event_carries_the_contract_payload_and_comes_last(tmp_path):
    run = execute(happy(), tmp_path)
    assert run.events[-1] is run.done
    assert run.done.phase.value == "finalize"
    data = run.done.data
    assert data["resolved_confidence"] == pytest.approx(0.9)
    assert data["steps"] == HAPPY_STEPS and data["tokens"] == 100 * HAPPY_STEPS
    assert data["patch_path"] == str(run.out / "patch.diff") and data["report_path"] == str(run.out / "report.md")
    assert set(data) == {"resolved_confidence", "patch_path", "report_path", "steps", "tokens", "seconds"}


def test_events_cover_every_type_and_usage_matches_the_steps(tmp_path):
    run = execute(happy(), tmp_path)
    assert {e.type for e in run.events} == {"phase", "message", "tool_call", "tool_result", "llm_usage", "done"}
    assert len(run.of("llm_usage")) == HAPPY_STEPS
    assert not run.of("error")
    tools_called = {e.data["tool"] for e in run.of("tool_call")}
    assert {"grep", "write_repro", "edit_file", "run_tests", "git_diff", "phase_done", "repro", "checkpoint"} <= tools_called


def test_checkpoints_are_taken_at_the_start_of_reproduce_and_of_patch_and_the_sandbox_is_closed(tmp_path):
    run = execute(happy(), tmp_path)
    events = run.pipeline.sandbox.events
    assert events[:2] == [("checkpoint", "reproduce-start", "ckpt-1"), ("checkpoint", "patch-start", "ckpt-2")]
    assert events[-1] == ("close",)
    assert not any(e[0] == "rollback" for e in events)


def test_each_phase_only_saw_its_own_system_prompt_and_tools(tmp_path):
    run = execute(happy(), tmp_path)
    seen = {}
    for messages, tools in run.llm.calls:
        phase = re.search(r"Phase: ([A-Z]+)", messages[0]["content"]).group(1)
        seen.setdefault(phase, set()).update(t["function"]["name"] for t in tools)
    assert seen["UNDERSTAND"] == {"phase_done", "give_up"}
    assert seen["LOCALIZE"] == {"list_dir", "grep", "read_file", "phase_done", "give_up"}
    assert seen["PATCH"] == {"read_file", "grep", "edit_file", "run_cmd", "git_diff", "phase_done", "give_up"}
    assert seen["REVIEW"] == {"git_diff", "read_file", "phase_done", "give_up"}


def test_the_issue_reaches_the_model_as_a_pinned_untrusted_brief(tmp_path):
    run = execute(happy(), tmp_path)
    first_user = run.llm.calls[0][0][1]
    assert first_user["role"] == "user"
    assert "add() returns the wrong sum" in first_user["content"] and "<issue>" in first_user["content"]


# ---- reproduce-first -----------------------------------------------------------------------


def _no_repro_script(reproduce_replies):
    return (
        understand() + localize() + reproduce_replies + good_patch()
        + [reply(call("run_tests")), done("tests pass")] + review_ok() + finalize()
    )


def test_a_repro_that_never_fails_exhausts_reproduce_and_caps_confidence_at_low(tmp_path):
    bad = [
        reply(call("write_repro", path="repro.py", content="print('nothing checked')\n")),
        done("it fails", repro_cmd=REPRO_CMD),
        done("it really fails", repro_cmd=REPRO_CMD),
    ]
    run = execute(_no_repro_script(bad), tmp_path, max_steps_per_phase=3)

    assert run.done.data["resolved_confidence"] == pytest.approx(0.3)
    assert "Confidence: **low**" in run.report
    assert "The bug was not reproduced (step_limit" in run.report
    assert "does NOT reproduce the bug" in " ".join(run.prompt_texts())
    assert "No failing repro could be established" in " ".join(run.prompt_texts())
    assert "+    return a + b" in run.patch, "the run still ends with a patch"
    assert run.phases == PHASE_ORDER
    assert run.llm.remaining == 0


def test_giving_up_on_the_repro_also_continues_at_low_confidence(tmp_path):
    run = execute(_no_repro_script([give_up("needs a GPU")]), tmp_path)
    assert run.done.data["resolved_confidence"] == pytest.approx(0.3)
    assert "not reproduced (gave_up: needs a GPU)" in run.report
    assert "Reproduced before patching: no" in run.report


@pytest.mark.parametrize(
    "replies, expected",
    [
        ([reply(call("write_repro", path="repro.py", content=REPRO_SCRIPT)), done("x")], "needs repro_cmd"),
        ([done("x", repro_cmd=REPRO_CMD)], "No repro script exists yet"),
        (
            [reply(call("write_repro", path="repro.py", content=REPRO_SCRIPT)), done("x", repro_cmd="pytest")],
            "must run the script you wrote under .anvil/",
        ),
        (
            [reply(call("write_repro", path="repro.py", content=REPRO_SCRIPT)), done("x", repro_cmd=".anvil/no-such-tool")],
            "could not run repro_cmd (exit 127)",
        ),
        (
            [reply(call("write_repro", path="repro.py", content="print('x')\n")), done("x", repro_cmd=REPRO_CMD)],
            "does NOT reproduce the bug",
        ),
    ],
)
def test_the_reproduce_gate_rejects_unusable_repros_with_a_reason(tmp_path, replies, expected):
    run = execute(understand() + localize() + replies, tmp_path)
    assert expected in run.last_tool_result()
    assert "patch" not in run.phases, "a rejected repro must not let the run move on to PATCH"


def test_a_repro_that_hangs_is_rejected(tmp_path):
    def hanging(cmd, files):
        if cmd.startswith(REPRO_CMD):
            return ExecResult(-1, "", "", True, 120.0)
        return project_exec(cmd, files)

    pipeline = FakePipeline(FakeSandbox(project_files(), on_exec=hanging))
    replies = [reply(call("write_repro", path="repro.py", content=REPRO_SCRIPT)), done("x", repro_cmd=REPRO_CMD)]
    run = execute(understand() + localize() + replies, tmp_path, pipeline=pipeline)
    assert "timed out" in run.last_tool_result()


def test_the_repro_that_failed_is_shown_to_the_patch_phase(tmp_path):
    run = execute(happy(), tmp_path)
    patch_kickoff = next(t for t in run.prompt_texts() if t.startswith("Begin phase PATCH."))
    assert "python .anvil/repro.py" in patch_kickoff and "AssertionError: add(2, 3) == -1" in patch_kickoff


# ---- verify, retry, rollback -----------------------------------------------------------------


def test_failing_verification_retries_then_rolls_back_and_asks_for_a_different_hypothesis(tmp_path):
    script = (
        understand() + localize() + reproduce()
        + wrong_patch("return a - b", "return a * b", "multiplied")
        + wrong_patch("return a * b", "return a ** b", "exponent")
        + good_patch()
        + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, max_patch_attempts=2, max_rollbacks=1)

    sandbox_events = run.pipeline.sandbox.events
    assert [e for e in sandbox_events if e[0] in ("checkpoint", "rollback")] == [
        ("checkpoint", "reproduce-start", "ckpt-1"),
        ("checkpoint", "patch-start", "ckpt-2"),
        ("rollback", "ckpt-2"),
    ]
    assert run.llm.remaining == 0
    assert len(run.llm.calls) == len(script), "failed verifications must not cost an LLM call"
    assert run.phases.count("patch") == 3 and run.phases.count("verify") == 3
    assert "+    return a + b" in run.patch and "a ** b" not in run.patch and "a * b" not in run.patch

    texts = " ".join(run.prompt_texts())
    assert "Your previous patch failed verification" in texts
    assert "The repro still fails after your patch" in texts
    rethink = next(t for t in run.prompt_texts() if "DIFFERENT hypothesis" in t)
    assert "- multiplied" in rethink and "- exponent" in rethink and "rolled back" in rethink
    assert "rollbacks: 1" in run.report and "Patch attempts: 3" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)


def test_the_repro_command_is_rerun_by_the_harness_after_every_patch_attempt(tmp_path):
    script = (
        understand() + localize() + reproduce()
        + wrong_patch("return a - b", "return a * b", "multiplied")
        + good_patch("return a * b") + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, max_patch_attempts=3)
    repro_events = [e for e in run.of("tool_call") if e.data["tool"] == "repro"]
    assert [e.phase.value for e in repro_events] == ["reproduce", "verify", "verify"]


def test_when_recovery_is_exhausted_the_last_attempt_is_kept_unverified_at_low_confidence(tmp_path):
    script = (
        understand() + localize() + reproduce()
        + wrong_patch("return a - b", "return a * b", "multiplied")
        + wrong_patch("return a - b", "return a ** b", "exponent")
        + finalize()
    )
    run = execute(script, tmp_path, max_patch_attempts=1, max_rollbacks=1)

    assert run.phases == PHASE_ORDER[:5] + ["patch", "verify", "patch", "verify", "finalize"], "no REVIEW for an unverified patch"
    assert "a ** b" in run.patch
    assert run.done.data["resolved_confidence"] == pytest.approx(0.3)
    assert "Verification still failed after 2 patch attempts and 1 rollbacks" in run.report
    assert "Verified after patching: no" in run.report
    assert run.llm.remaining == 0


def test_a_model_that_cannot_patch_ends_with_an_empty_patch_and_a_report(tmp_path):
    run = execute(understand() + localize() + reproduce() + [give_up("no idea")], tmp_path,
                  max_patch_attempts=1, max_rollbacks=0)
    assert run.patch == ""
    assert run.done.data["resolved_confidence"] == 0.0
    assert "None: no patch was produced." in run.report
    assert len(run.llm.calls) == 8, "no verification and no closing summary are requested for an empty patch"


def test_phase_done_without_any_edit_is_rejected_by_the_patch_gate(tmp_path):
    script = understand() + localize() + reproduce() + [done("fixed it, honest")]
    run = execute(script, tmp_path)  # the script ends here, so the run stops with an error after the rejection
    assert "No source files have changed yet" in run.last_tool_result()


def test_failing_tests_that_the_model_judges_unrelated_downgrade_confidence_to_medium(tmp_path):
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
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6)
    assert "FAIL: run_tests" in run.report and "Confidence: **medium**" in run.report


# ---- review ---------------------------------------------------------------------------------

DOCSTRING = {"path": "calc.py", "old": "def add(a, b):", "new": 'def add(a, b):\n    """Add two numbers."""'}


def test_a_review_that_requests_changes_triggers_exactly_one_rework_round(tmp_path):
    script = (
        understand() + localize() + reproduce() + good_patch() + verify()
        + [reply(call("git_diff")), give_up("Add a docstring to add().")]
        + [reply(call("edit_file", **DOCSTRING)), done("added a docstring")]
        + verify() + finalize()
    )
    run = execute(script, tmp_path)

    assert run.phases == PHASE_ORDER[:5] + ["patch", "verify", "review", "patch", "verify", "finalize"]
    assert '"""Add two numbers."""' in run.patch and "+    return a + b" in run.patch
    assert "Add a docstring to add()." in " ".join(run.prompt_texts())
    assert "one rework round was made and was not re-reviewed" in run.report
    assert "Review: changes requested" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6)
    assert [e[1] for e in run.pipeline.sandbox.events if e[0] == "checkpoint"] == [
        "reproduce-start", "patch-start", "before-rework"
    ]
    assert run.llm.remaining == 0


def test_a_rework_that_fails_verification_restores_the_reviewed_patch(tmp_path):
    script = (
        understand() + localize() + reproduce() + good_patch() + verify()
        + [reply(call("git_diff")), give_up("Please undo the fix.")]
        + wrong_patch("return a + b", "return a * b", "broke the fix")
        + finalize()
    )
    run = execute(script, tmp_path, max_patch_attempts=1)

    assert "+    return a + b" in run.patch and "a * b" not in run.patch
    assert ("rollback", "ckpt-3") in run.pipeline.sandbox.events, "the checkpoint taken before the rework"
    assert "the reviewed patch was restored" in run.report
    assert "Verified after patching: yes" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6)


def test_a_review_that_never_finishes_leaves_the_patch_unreviewed(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + verify() + [reply(text="hmm")] * 3 + finalize()
    run = execute(script, tmp_path)
    assert "Review: inconclusive (stalled)" in run.report
    assert "the patch is unreviewed" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6)


# ---- sandboxes that are not contract-clean ----------------------------------------------------


def test_the_repro_survives_a_sandbox_whose_checkpoint_wipes_untracked_files(tmp_path):
    pipeline = FakePipeline(StashStyleSandbox(project_files(), on_exec=project_exec))
    script = (
        understand() + localize() + reproduce()
        + wrong_patch("return a - b", "return a * b", "multiplied")
        + wrong_patch("return a - b", "return a ** b", "exponent")
        + good_patch() + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, pipeline=pipeline, max_patch_attempts=1, max_rollbacks=2)
    assert "+    return a + b" in run.patch
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)
    assert ".anvil/repro.py" in pipeline.sandbox.files
    assert "rollbacks: 2" in run.report, "the second rollback used a ref the sandbox had already consumed"


# ---- budgets ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budget, steps, tokens",
    [
        ({"max_total_steps": 5}, 5, 500),
        ({"max_tokens_total": 450}, 5, 500),
        ({"wall_clock_seconds": 0}, 0, 0),
    ],
)
def test_budget_exhaustion_still_writes_outputs_and_emits_done(tmp_path, budget, steps, tokens):
    run = execute(happy(), tmp_path, **budget)

    assert (run.out / "patch.diff").exists() and (run.out / "report.md").exists()
    assert run.done.data["steps"] == steps and run.done.data["tokens"] == tokens
    assert run.done.data["resolved_confidence"] == 0.0
    assert "budget exhausted" in run.report and "Stopped early" in run.report
    assert "None: no patch was produced." in run.report
    assert [e.data["kind"] for e in run.of("error")] == ["budget"]
    assert run.pipeline.sandbox.closed
    assert run.phases[-1] == "finalize"


def test_a_run_that_runs_out_of_budget_late_keeps_the_verified_patch(tmp_path):
    run = execute(happy(), tmp_path, max_total_steps=HAPPY_STEPS - 3)  # REVIEW's first call is refused

    assert "+    return a + b" in run.patch
    assert "Verified after patching: yes" in run.report and "Review: not run" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6)
    assert run.done.data["steps"] == HAPPY_STEPS - 3
    assert "The closing summary" not in run.report, "no LLM call is attempted once the run was halted"


def test_the_step_budget_can_also_cut_the_closing_summary(tmp_path):
    run = execute(happy(), tmp_path, max_total_steps=HAPPY_STEPS - 1)
    assert "+    return a + b" in run.patch
    assert "Review: approved" in run.report
    assert "The closing summary could not be written" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)


# ---- failures anywhere still end with outputs ---------------------------------------------------


def test_a_tool_that_raises_does_not_stop_the_run_or_the_outputs(tmp_path):
    tools = [crashing("read_file", RuntimeError("boom")), *[t for t in default_tools() if t.name != "read_file"]]
    run = execute(happy(), tmp_path, pipeline=FakePipeline(registry=FakeRegistry(tools)))

    (error,) = run.of("error")
    assert error.data["kind"] == "tool" and "boom" in error.data["message"]
    assert "+    return a + b" in run.patch
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)
    assert run.llm.remaining == 0


def test_a_crash_while_patching_still_yields_outputs_and_a_low_confidence_report(tmp_path):
    tools = [crashing("edit_file"), *[t for t in default_tools() if t.name != "edit_file"]]
    script = understand() + localize() + reproduce() + [reply(call("edit_file", **FIX)), give_up("edit tool is broken")]
    run = execute(script, tmp_path, pipeline=FakePipeline(registry=FakeRegistry(tools)),
                  max_patch_attempts=1, max_rollbacks=0)
    assert run.patch == ""
    assert (run.out / "report.md").exists() and run.done.data["resolved_confidence"] == 0.0
    assert [e.data["kind"] for e in run.of("error")] == ["tool"]


def test_the_model_script_running_dry_mid_run_is_an_internal_error_not_a_crash(tmp_path):
    run = execute(understand(), tmp_path)  # LOCALIZE's first call finds an empty script
    (error,) = run.of("error")
    assert error.data["kind"] == "localize" and "script exhausted" in error.data["message"]
    assert "Stopped early by an error (localize)" in run.report
    assert run.done.data["resolved_confidence"] == 0.0 and run.pipeline.sandbox.closed


def test_an_llm_failure_stops_the_run_gracefully(tmp_path):
    class Down:
        def chat(self, messages, tools=None):
            raise LLMError("LLM request failed with HTTP 503", status_code=503, retryable=True, attempts=5)

    run = execute([], tmp_path, llm=Down())
    (error,) = run.of("error")
    assert error.data["kind"] == "llm" and "503" in error.data["message"]
    assert "Stopped early: LLM call failed" in run.report
    assert run.done.data["steps"] == 1


@pytest.mark.parametrize(
    "pipeline_kwargs, phase",
    [
        ({"fail_ingest": RuntimeError("issue not found")}, "ingest"),
        ({"fail_profile": RuntimeError("no sandbox backend")}, "profile"),
    ],
)
def test_setup_failures_still_produce_an_empty_patch_and_a_report(tmp_path, pipeline_kwargs, phase):
    run = execute([], tmp_path, pipeline=FakePipeline(**pipeline_kwargs))
    assert run.patch == ""
    assert f"Stopped early by an error ({phase})" in run.report
    assert [e.data["kind"] for e in run.of("error")] == [phase]
    assert run.done.data["resolved_confidence"] == 0.0 and run.done.data["steps"] == 0
    assert run.phases[-1] == "finalize"


def test_a_missing_api_key_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    out = tmp_path / "out"
    bus = EventBus()
    queue = bus.subscribe()
    config = {"output_dir": str(out), "model": "m", "base_url": "http://localhost:9/v1"}
    run_harness(ISSUE_URL, config, bus, pipeline=FakePipeline())  # no llm: the real client is built
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [e.data["kind"] for e in events if e.type == "error"] == ["llm"]
    assert "AI_API_KEY" in (out / "report.md").read_text()
    assert events[-1].type == "done"


def test_a_diff_that_cannot_be_read_still_ends_with_outputs(tmp_path):
    class BrokenDiff(FakeSandbox):
        def diff(self):
            raise RuntimeError("git exploded")

    pipeline = FakePipeline(BrokenDiff(project_files(), on_exec=project_exec))
    run = execute(understand() + localize() + reproduce() + [give_up("nothing works")], tmp_path,
                  pipeline=pipeline, max_patch_attempts=1, max_rollbacks=0)
    assert run.patch == "" and run.done.data["resolved_confidence"] == 0.0
    assert "git exploded" in " ".join(e.data["message"] for e in run.of("error"))


def test_a_subscriber_that_raises_cannot_break_the_run(tmp_path):
    class ExplodingBus(EventBus):
        def emit(self, event):
            raise RuntimeError("subscriber bug")

    out = tmp_path / "out"
    run_harness(ISSUE_URL, {"output_dir": str(out)}, ExplodingBus(), llm=RecordingLLM(happy()), pipeline=FakePipeline())
    assert "+    return a + b" in (out / "patch.diff").read_text()


def test_an_unwritable_output_directory_is_reported_and_done_is_still_emitted(tmp_path):
    blocker = tmp_path / "out"
    blocker.write_text("i am a file, not a directory")
    run = execute(happy(), tmp_path)
    assert [e.data["kind"] for e in run.of("error")] == ["io"]
    assert run.events[-1].type == "done"


def test_invalid_settings_fall_back_to_defaults_and_are_reported(tmp_path):
    run = execute(happy(), tmp_path, max_total_steps="lots")
    assert [e.data["kind"] for e in run.of("error")] == ["config"]
    assert "max_total_steps" in run.report and "defaults were used" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)


def test_ctrl_c_writes_the_outputs_first_and_then_propagates(tmp_path):
    class Interrupted:
        def chat(self, messages, tools=None):
            raise KeyboardInterrupt

    out = tmp_path / "out"
    bus = EventBus()
    queue = bus.subscribe()
    with pytest.raises(KeyboardInterrupt):
        run_harness(ISSUE_URL, {"output_dir": str(out)}, bus, llm=Interrupted(), pipeline=FakePipeline())
    assert (out / "patch.diff").exists() and (out / "report.md").exists()
    types = []
    while not queue.empty():
        types.append(queue.get_nowait().type)
    assert types[-1] == "done"


def test_run_harness_can_be_called_from_a_worker_thread_while_the_loop_consumes_events(tmp_path):
    import asyncio
    import threading

    out = tmp_path / "out"
    bus = EventBus()

    async def main():
        queue = bus.subscribe()
        worker = threading.Thread(
            target=run_harness,
            args=(ISSUE_URL, {"output_dir": str(out)}, bus),
            kwargs={"llm": RecordingLLM(happy()), "pipeline": FakePipeline()},
        )
        worker.start()
        kinds = []
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=10)
            kinds.append(event.type)
            if event.type == "done":
                break
        worker.join()
        return kinds

    kinds = asyncio.run(main())
    assert kinds[0] == "phase" and kinds[-1] == "done" and "llm_usage" in kinds
