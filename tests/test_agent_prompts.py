"""Phase specs, control tools, kickoff messages and the write_repro tool."""

import pytest

from anvil.agent.prompts import (
    GIVE_UP,
    MAX_BODY_CHARS,
    MAX_COMMENTS,
    PHASE_DONE,
    PHASE_SPECS,
    control_tools,
    finalize_kickoff,
    issue_brief,
    patch_kickoff,
    phase_kickoff,
    phase_summary,
    verify_kickoff,
)
from anvil.agent.repro import WriteReproTool, scratch_path
from anvil.events import Phase
from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile
from tests.fakes import FakeSandbox

# ---- phase specs ----------------------------------------------------------------------------

ALLOWLISTS = {
    Phase.UNDERSTAND: set(),
    Phase.LOCALIZE: {"list_dir", "grep", "read_file"},
    Phase.REPRODUCE: {"read_file", "grep", "edit_file", "run_cmd"},
    Phase.PATCH: {"read_file", "grep", "edit_file", "run_cmd", "git_diff"},
    Phase.VERIFY: {"run_tests", "run_cmd", "read_file", "git_diff"},
    Phase.REVIEW: {"git_diff", "read_file"},
    Phase.FINALIZE: set(),
}


def test_every_llm_phase_has_a_spec_and_ingest_profile_do_not():
    assert set(PHASE_SPECS) == set(ALLOWLISTS)
    assert Phase.INGEST not in PHASE_SPECS and Phase.PROFILE not in PHASE_SPECS


@pytest.mark.parametrize("phase, allowed", ALLOWLISTS.items())
def test_allowlists_match_the_design(phase, allowed):
    tools = set(PHASE_SPECS[phase].tools)
    if phase is Phase.REPRODUCE:
        assert tools == allowed | {"write_repro"}
    else:
        assert tools == allowed


def test_only_understand_and_finalize_are_text_only():
    assert {p for p, s in PHASE_SPECS.items() if s.text_only} == {Phase.UNDERSTAND, Phase.FINALIZE}


def test_only_reproduce_can_write_files_from_scratch_and_only_patch_and_reproduce_can_edit():
    editors = {p for p, s in PHASE_SPECS.items() if "edit_file" in s.tools}
    assert editors == {Phase.REPRODUCE, Phase.PATCH}
    assert {p for p, s in PHASE_SPECS.items() if "write_repro" in s.tools} == {Phase.REPRODUCE}


@pytest.mark.parametrize("phase", list(PHASE_SPECS))
def test_every_system_prompt_names_its_phase_and_the_control_tools(phase):
    prompt = PHASE_SPECS[phase].system_prompt
    assert f"Phase: {phase.value.upper()}" in prompt
    assert PHASE_DONE in prompt and GIVE_UP in prompt
    assert "untrusted" in prompt


@pytest.mark.parametrize("phase", list(PHASE_SPECS))
def test_control_tools_are_offered_in_every_phase(phase):
    names = [t["function"]["name"] for t in control_tools(phase)]
    assert names == [PHASE_DONE, GIVE_UP]


def test_reproduce_phase_done_requires_the_repro_command():
    def required(phase):
        done = control_tools(phase)[0]["function"]["parameters"]
        return done["required"]

    assert required(Phase.REPRODUCE) == ["summary", "repro_cmd"]
    assert required(Phase.PATCH) == ["summary"]


# ---- messages -------------------------------------------------------------------------------


def _issue(**overrides) -> IssueRef:
    base = dict(owner="acme", repo="calc", number=7, url="https://github.com/acme/calc/issues/7", title="add is wrong", body="add(2, 3) gives -1")
    return IssueRef(**{**base, **overrides})


_PROFILE = RepoProfile(["python"], "python", "pip install -e .", "pytest", "pytest", notes="src layout")


def test_issue_brief_contains_issue_profile_and_map_and_fences_the_untrusted_text():
    brief = issue_brief(_issue(comments=["me too"]), _PROFILE, "calc.py\ntests/")
    for expected in ("add is wrong", "acme/calc", "add(2, 3) gives -1", "pytest", "src layout", "me too", "calc.py\ntests/"):
        assert expected in brief
    assert "<issue>" in brief and "</issue>" in brief
    assert "untrusted" in brief


def test_issue_brief_clips_long_bodies_and_limits_comments():
    long_issue = _issue(body="x" * (MAX_BODY_CHARS * 3), comments=[f"comment number {i}" for i in range(20)])
    brief = issue_brief(long_issue, _PROFILE, "")
    assert len(brief) < MAX_BODY_CHARS + 3000
    assert f"comment number {MAX_COMMENTS - 1}" in brief
    assert f"comment number {MAX_COMMENTS}" not in brief


def test_issue_brief_survives_a_sparse_issue_and_profile():
    brief = issue_brief(_issue(title="", body=""), RepoProfile([], "", None, None, None), "")
    assert "(no title)" in brief and "(empty)" in brief and "unknown" in brief


def test_phase_kickoff_and_summary_formats():
    assert phase_kickoff(Phase.LOCALIZE) == "Begin phase LOCALIZE."
    assert phase_kickoff(Phase.LOCALIZE, "facts").endswith("\n\nfacts")
    assert phase_summary(Phase.UNDERSTAND, " it subtracts ") == "[UNDERSTAND summary]\nit subtracts"
    assert "(none)" in phase_summary(Phase.UNDERSTAND, "")


def test_first_patch_kickoff_carries_the_repro_and_its_failure():
    text = patch_kickoff(attempt=1, repro_cmd="python .anvil/repro.py", repro_output="AssertionError")
    assert "attempt 1" in text and "python .anvil/repro.py" in text and "AssertionError" in text
    assert "failed verification" not in text


def test_patch_kickoff_without_a_repro_warns_against_speculation():
    text = patch_kickoff(attempt=1, repro_cmd=None, repro_output="")
    assert "No failing repro" in text and "python" not in text


def test_retry_kickoff_includes_the_trimmed_failure():
    text = patch_kickoff(attempt=2, repro_cmd="r", repro_output="o", feedback="FAILED test_add", kind="retry")
    assert "still applied" in text and "FAILED test_add" in text


def test_rethink_kickoff_lists_earlier_approaches_and_demands_a_different_hypothesis():
    text = patch_kickoff(
        attempt=3, repro_cmd="r", repro_output="o", feedback="still failing", kind="rethink",
        earlier_approaches=["changed the operator", "cast to int"],
    )
    assert "rolled back" in text and "DIFFERENT hypothesis" in text
    assert "- changed the operator" in text and "- cast to int" in text


def test_rework_kickoff_carries_the_reviewers_request():
    text = patch_kickoff(attempt=4, repro_cmd="r", repro_output="o", feedback="remove the debug print", kind="rework")
    assert "reviewer" in text and "remove the debug print" in text


def test_verify_and_finalize_kickoffs():
    assert "`python .anvil/repro.py`" in verify_kickoff(repro_cmd="python .anvil/repro.py")
    assert "no confirmed repro" in verify_kickoff(repro_cmd=None)
    assert "Facts about this run:\nreproduced: yes" in finalize_kickoff("reproduced: yes")


# ---- write_repro ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("repro.py", ".anvil/repro.py"),
        (".anvil/repro.py", ".anvil/repro.py"),
        ("./repro.py", ".anvil/repro.py"),
        ("sub/dir/repro.js", ".anvil/sub/dir/repro.js"),
        (" repro.py ", ".anvil/repro.py"),
        ("a\\b.py", ".anvil/a/b.py"),
    ],
)
def test_scratch_path_lands_inside_the_scratch_dir(raw, expected):
    assert scratch_path(raw) == expected


@pytest.mark.parametrize("raw", ["", ".anvil", "/etc/passwd", "../outside.py", ".anvil/../../x", "a/../../x"])
def test_scratch_path_rejects_anything_that_escapes_or_is_not_a_file(raw):
    assert scratch_path(raw) is None


def test_write_repro_writes_inside_the_scratch_dir_and_reports_the_path():
    sandbox = FakeSandbox({})
    result = WriteReproTool().run({"path": "repro.py", "content": "print(1)\n"}, sandbox)
    assert result.ok and result.meta == {"path": ".anvil/repro.py"}
    assert sandbox.files == {".anvil/repro.py": "print(1)\n"}


def test_write_repro_refuses_paths_outside_the_scratch_dir_without_writing():
    sandbox = FakeSandbox({})
    result = WriteReproTool().run({"path": "../evil.py", "content": "x"}, sandbox)
    assert not result.ok and sandbox.files == {}


@pytest.mark.parametrize("args", [{}, {"path": "r.py"}, {"content": "x"}, {"path": 3, "content": "x"}, {"path": "r.py", "content": None}])
def test_write_repro_requires_string_path_and_content(args):
    assert not WriteReproTool().run(args, FakeSandbox({})).ok


def test_write_repro_turns_sandbox_errors_into_failed_results():
    class ReadOnly(FakeSandbox):
        def write_file(self, path, content):
            raise PermissionError("read-only file system")

    result = WriteReproTool().run({"path": "r.py", "content": "x"}, ReadOnly({}))
    assert not result.ok and "read-only" in result.output


def test_write_repro_satisfies_the_tool_protocol_shape():
    tool = WriteReproTool()
    assert tool.name == "write_repro"
    assert tool.parameters["required"] == ["path", "content"]
