"""REPRODUCE proves the bug; it must not fix it.

Seen with a real model (Qwen3-Coder): it wrote a valid failing repro, then edited the source in the same phase. The harness
re-ran the repro on the fixed tree, got exit 0, said "does NOT reproduce", and the model spent thirteen more calls
trying to reproduce a bug it had just fixed before giving up. The phase offers no edit tool now, and whatever a shell
command changes outside .anvil/ is reverted before the harness runs the repro.
"""

from anvil.agent.prompts import PHASE_SPECS
from anvil.events import Phase
from anvil.sandbox.base import ExecResult
from tests.fakes import (
    FIX,
    REPRO_CMD,
    REPRO_SCRIPT,
    FakePipeline,
    FakeSandbox,
    call,
    done,
    give_up,
    project_exec,
    project_files,
    reply,
)
from tests.test_orchestrator import execute, finalize, good_patch, localize, review_ok, understand, verify

SED_FIX = "sed -i 's/a - b/a + b/' calc.py"


def editing_exec(cmd: str, files: dict[str, str]) -> ExecResult:
    """``project_exec`` plus a shell command that fixes the bug the way a model might: with sed."""
    if cmd == SED_FIX:
        files["calc.py"] = files["calc.py"].replace("return a - b", "return a + b")
        return ExecResult(0, "", "", False, 0.01)
    return project_exec(cmd, files)


def pipeline(sandbox_class=FakeSandbox) -> FakePipeline:
    return FakePipeline(sandbox_class(project_files(), on_exec=editing_exec))


def write_and_run_repro():
    return [
        reply(call("write_repro", path="repro.py", content=REPRO_SCRIPT)),
        reply(call("run_cmd", cmd=REPRO_CMD)),
    ]


def rest():
    return good_patch() + verify() + review_ok() + finalize()


def sandbox_moves(run):
    return [e for e in run.pipeline.sandbox.events if e[0] in ("checkpoint", "rollback")]


# ---- the phase has no way to edit the repository ---------------------------------------------------------


def test_reproduce_offers_no_edit_tool_and_its_prompt_does_not_advertise_one():
    spec = PHASE_SPECS[Phase.REPRODUCE]
    assert "edit_file" not in spec.tools
    assert "edit_file" not in spec.system_prompt.split("\n")[0], "the Tools: list of the prompt"
    assert "reverts every change made outside .anvil/" in spec.system_prompt, "the model is told why not to edit"
    assert "edit_file" in PHASE_SPECS[Phase.PATCH].tools


def test_an_edit_file_call_in_reproduce_is_refused_and_the_file_is_untouched(tmp_path):
    fake = pipeline()
    script = (
        understand() + localize() + write_and_run_repro()
        + [reply(call("edit_file", **FIX)), reply(call("run_cmd", cmd=REPRO_CMD)), done("fails", repro_cmd=REPRO_CMD)]
        + rest()
    )
    run = execute(script, tmp_path, pipeline=fake)

    refused = next(e for e in run.of("tool_result") if e.data["tool"] == "edit_file" and e.phase is Phase.REPRODUCE)
    assert refused.data["ok"] is False and "not available in the reproduce phase" in refused.data["output_preview"]
    assert "Reproduced before patching: yes" in run.report
    assert not any(e[0] == "rollback" for e in fake.sandbox.events), "nothing was changed, so nothing to revert"


# ---- a shell command that edits the source ----------------------------------------------------------------


def test_source_edits_made_through_the_shell_are_reverted_before_the_repro_is_run_and_the_phase_goes_on(tmp_path):
    fake = pipeline()
    script = (
        understand() + localize() + write_and_run_repro()
        + [
            reply(call("run_cmd", cmd=SED_FIX)),
            done("Fixed add() by changing - to +", repro_cmd=REPRO_CMD),  # refused: the tree was reverted
            reply(call("run_cmd", cmd=REPRO_CMD)),  # the repro fails again on the clean tree
            done("add(2, 3) returns -1: the repro fails", repro_cmd=REPRO_CMD),
        ]
        + rest()
    )
    run = execute(script, tmp_path, pipeline=fake)

    assert sandbox_moves(run) == [
        ("checkpoint", "reproduce-start", "ckpt-1"),
        ("rollback", "ckpt-1"),
        ("checkpoint", "patch-start", "ckpt-2"),
    ]
    told = next(t for t in run.prompt_texts() if "has reverted them" in t)
    assert "calc.py" in told and "not of a fix" in told
    assert ".anvil/repro.py" in fake.sandbox.files, "the repro outlives the revert"
    assert "Reproduced before patching: yes" in run.report, "confirmed on the code the issue describes"
    assert "the harness reverted them" in run.report and "calc.py" in run.report.split("Known limitations")[1]
    assert "+    return a + b" in run.patch and run.patch.count("calc.py") >= 1
    assert run.llm.remaining == 0


def test_the_summary_of_a_fix_never_reaches_the_later_phases_as_the_reproduction(tmp_path):
    script = (
        understand() + localize() + write_and_run_repro()
        + [
            reply(call("run_cmd", cmd=SED_FIX)),
            done("Fixed add() by changing - to +", repro_cmd=REPRO_CMD),
            reply(call("run_cmd", cmd=REPRO_CMD)),
            done("add(2, 3) returns -1: the repro fails", repro_cmd=REPRO_CMD),
        ]
        + rest()
    )
    run = execute(script, tmp_path, pipeline=pipeline())
    patch_prompts = " ".join(text for text in run.prompt_texts())
    assert "Fixed add() by changing - to +" not in patch_prompts.split("Phase: PATCH")[-1]
    assert "the repro fails" in patch_prompts


def test_edits_left_behind_when_the_phase_ends_without_phase_done_are_reverted_before_patch(tmp_path):
    fake = pipeline()
    script = (
        understand() + localize() + write_and_run_repro()
        + [reply(call("run_cmd", cmd=SED_FIX)), give_up("it is already fixed, I cannot reproduce it")]
        + rest()
    )
    run = execute(script, tmp_path, pipeline=fake)

    assert sandbox_moves(run) == [
        ("checkpoint", "reproduce-start", "ckpt-1"),
        ("rollback", "ckpt-1"),
        ("checkpoint", "patch-start", "ckpt-2"),
    ], "PATCH starts from the original code, not from the model's own fix"
    assert "+    return a + b" in run.patch, "PATCH could make its edit: the code it expects was there"
    assert "Reproduced before patching: no" in run.report
    assert "The bug was not reproduced" in run.report and "the harness reverted them" in run.report


def test_a_reproduce_that_changes_only_files_under_dot_anvil_reverts_nothing(tmp_path):
    fake = pipeline()
    script = understand() + localize() + write_and_run_repro() + [done("fails", repro_cmd=REPRO_CMD)] + rest()
    run = execute(script, tmp_path, pipeline=fake)
    assert [e for e in sandbox_moves(run) if e[0] == "rollback"] == []
    assert "the harness reverted them" not in run.report


def test_when_no_checkpoint_could_be_taken_the_edits_stay_and_the_report_says_so(tmp_path):
    class NoReproduceCheckpoint(FakeSandbox):
        def checkpoint(self, label):
            if label == "reproduce-start":
                raise RuntimeError("no git")
            return super().checkpoint(label)

    fake = pipeline(NoReproduceCheckpoint)
    script = (
        understand() + localize() + write_and_run_repro()
        + [reply(call("run_cmd", cmd=SED_FIX)), give_up("already fixed")]
        + [reply(call("git_diff")), done("nothing left to change")]  # PATCH: the tree is already fixed
        + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, pipeline=fake)

    assert not any(e[0] == "rollback" for e in fake.sandbox.events)
    assert "could not be reverted" in run.report
    assert run.done.type == "done", "the run still ends with its outputs"


def test_the_checkpoint_and_the_revert_are_announced_in_the_reproduce_phase_and_the_phase_starts_once(tmp_path):
    script = (
        understand() + localize() + write_and_run_repro()
        + [reply(call("run_cmd", cmd=SED_FIX)), give_up("already fixed")] + rest()
    )
    run = execute(script, tmp_path, pipeline=pipeline())

    moves = [e for e in run.of("tool_call") if e.data["tool"] in ("checkpoint", "rollback") and e.phase is Phase.REPRODUCE]
    assert [(e.data["tool"], e.data["args"].get("label") or e.data["args"].get("ref")) for e in moves[:2]] == [
        ("checkpoint", "reproduce-start"),
        ("rollback", "ckpt-1"),
    ]
    starts = [e for e in run.of("phase") if e.data["name"] == "reproduce"]
    assert len(starts) == 1
