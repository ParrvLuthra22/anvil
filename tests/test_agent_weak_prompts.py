"""features.weak_model_prompts: the short phase prompts, and that the old ones come back with the switch off."""

from __future__ import annotations

import re

import pytest

from anvil.agent.prompts import PHASE_SPECS, WEAK_PHASE_SPECS, control_tools, phase_kickoff, phase_spec
from anvil.agent.repro import WriteReproTool
from anvil.events import Phase
from anvil.llm.toolcalls import parse_text_tool_calls
from anvil.repo.profile import RepoProfile
from anvil.tools.registry import make_default_registry
from tests.test_orchestrator import execute, finalize, good_patch, happy, localize, reproduce, review_ok, understand, verify, wrong_patch

PHASES = list(PHASE_SPECS)
WEAK_BASE_START = "You are ANVIL, a software engineer fixing one GitHub issue"
OLD_BASE_START = "You are ANVIL, an autonomous software engineer"


def real_schemas(phase: Phase) -> list[dict]:
    """The schemas the model is actually offered in ``phase``: the registry's tools, write_repro and the control tools."""
    registry = make_default_registry(RepoProfile(["python"], "python", None, "pytest", "pytest"))
    registry.register(WriteReproTool())
    spec = WEAK_PHASE_SPECS[phase]
    allowed = [s for s in registry.schemas() if s.get("function", s)["name"] in spec.tools]
    return allowed + control_tools(phase)


# ---- which prompts, by flag -------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_the_flag_selects_the_table(phase):
    assert phase_spec(phase, True) is WEAK_PHASE_SPECS[phase]
    assert phase_spec(phase, False) is PHASE_SPECS[phase]
    assert WEAK_PHASE_SPECS[phase].system_prompt != PHASE_SPECS[phase].system_prompt


def test_a_whole_run_sends_the_old_prompts_by_default_and_the_short_ones_only_with_the_switch_on(tmp_path):
    short = execute(happy(), tmp_path / "a", features={"weak_model_prompts": True})
    old = execute(happy(), tmp_path / "b")

    assert all(m[0]["content"].startswith(WEAK_BASE_START) for m, _ in short.llm.calls)
    assert all(m[0]["content"].startswith(OLD_BASE_START) for m, _ in old.llm.calls)
    for run in (short, old):
        assert "Confidence: **high** (0.90)" in run.report and "Verified after patching: yes" in run.report
        assert run.llm.remaining == 0


def test_the_other_scenarios_still_pass_with_the_short_prompts_switched_on(tmp_path):
    script = (
        understand() + localize() + reproduce()
        + wrong_patch("return a - b", "return a * b", "multiplied")
        + wrong_patch("return a * b", "return a ** b", "exponent")
        + wrong_patch("return a ** b", "return a // b", "floor")
        + good_patch() + verify() + review_ok() + finalize()
    )
    rollback = execute(script, tmp_path / "r", max_patch_attempts=3, max_rollbacks=1, max_total_patch_attempts=10, features={"weak_model_prompts": True})
    assert "rollbacks: 1" in rollback.report and "Verified after patching: yes" in rollback.report
    assert all(m[0]["content"].startswith(WEAK_BASE_START) for m, _ in rollback.llm.calls)


def test_each_phase_call_carries_its_own_phase_marker(tmp_path):
    run = execute(happy(), tmp_path)
    phases = [re.search(r"Phase: ([A-Z]+)", m[0]["content"]).group(1) for m, _ in run.llm.calls]
    assert phases == ["UNDERSTAND"] + ["LOCALIZE"] * 3 + ["REPRODUCE"] * 3 + ["PATCH"] * 3 + ["VERIFY"] * 2 + ["REVIEW"] * 2 + ["FINALIZE"]


def test_the_kickoff_messages_do_not_depend_on_the_flag():
    assert phase_kickoff(Phase.PATCH, "detail") == "Begin phase PATCH.\n\ndetail"


# ---- the same phases with the same tools -------------------------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_the_short_prompts_offer_the_same_tools_as_the_old_ones(phase):
    assert WEAK_PHASE_SPECS[phase].tools == PHASE_SPECS[phase].tools
    assert WEAK_PHASE_SPECS[phase].phase is phase
    assert WEAK_PHASE_SPECS[phase].text_only == PHASE_SPECS[phase].text_only


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_every_short_prompt_names_its_phase_the_way_the_old_ones_do(phase):
    assert f"Phase: {phase.value.upper()}" in WEAK_PHASE_SPECS[phase].system_prompt


def test_reproduce_still_has_no_edit_tool_and_still_says_the_harness_reverts_edits():
    spec = WEAK_PHASE_SPECS[Phase.REPRODUCE]
    assert "edit_file" not in spec.tools
    assert "reverts every change made outside .anvil/" in spec.system_prompt
    assert "do not fix it" in spec.system_prompt.lower()


# ---- shorter ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_every_short_prompt_is_shorter_than_the_one_it_replaces(phase):
    assert len(WEAK_PHASE_SPECS[phase].system_prompt) < len(PHASE_SPECS[phase].system_prompt)


def test_together_they_are_at_least_a_quarter_shorter():
    old = sum(len(s.system_prompt) for s in PHASE_SPECS.values())
    new = sum(len(s.system_prompt) for s in WEAK_PHASE_SPECS.values())
    assert new <= old * 0.75, f"{new} against {old} characters"


# ---- the rules every phase carries ------------------------------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
@pytest.mark.parametrize(
    "rule",
    [
        "exactly ONE tool call per turn",
        "Read a file before editing it",
        "smallest change that fixes the issue",
        "Never edit tests",
        "call phase_done(summary)",
    ],
)
def test_every_short_prompt_carries_the_common_rules(phase, rule):
    assert rule in WEAK_PHASE_SPECS[phase].system_prompt


# ---- one concrete example call per phase --------------------------------------------------------------------------


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_each_short_prompt_has_exactly_one_example_call_and_it_is_a_valid_one(phase):
    prompt = WEAK_PHASE_SPECS[phase].system_prompt
    assert prompt.count("Example call:") == 1
    example = prompt.split("Example call:")[1].strip().splitlines()[0]
    schemas = real_schemas(phase)
    parsed = parse_text_tool_calls(example, schemas)
    assert len(parsed.calls) == 1, f"the parser reads it as a call: {example!r}"
    call = parsed.calls[0]
    schema = next(s.get("function", s) for s in schemas if s.get("function", s)["name"] == call["tool"])
    properties = schema["parameters"].get("properties", {})
    assert set(call["args"]) <= set(properties), f"{call['tool']} takes {sorted(properties)}"
    assert set(schema["parameters"].get("required", [])) <= set(call["args"]), "every required argument is shown"


@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_the_example_calls_a_tool_the_phase_actually_has(phase):
    example = WEAK_PHASE_SPECS[phase].system_prompt.split("Example call:")[1].strip()
    name = re.match(r"(\w+)\(", example).group(1)
    assert name in WEAK_PHASE_SPECS[phase].tools or name == "phase_done"


# ---- PATCH and REVIEW: how errors are raised ----------------------------------------------------------------------


def test_patch_says_which_exception_to_raise_never_assert_and_to_follow_how_the_module_reports_errors():
    prompt = WEAK_PHASE_SPECS[Phase.PATCH].system_prompt
    assert "raise a specific exception (ValueError, TypeError, or the type the issue names)" in prompt
    assert "with a clear message that names the bad value" in prompt
    assert "Never use assert to validate input" in prompt
    assert "Grep the file, then its package, for how similar errors are reported and follow that" in prompt
    assert "If a neighbouring check of the same input uses assert, raise the proper exception there too" in prompt


def test_review_checks_the_exception_type_and_message_against_the_same_rules():
    prompt = WEAK_PHASE_SPECS[Phase.REVIEW].system_prompt
    assert "Does the exception type and message match what the issue asks for?" in prompt
    assert "never assert for input validation" in prompt and "ValueError, TypeError" in prompt
    assert "names the bad value" in prompt and "same convention as the other errors in that module" in prompt


def test_the_old_prompts_carry_the_short_validation_rule_and_none_of_the_weak_style():
    old = " ".join(spec.system_prompt for spec in PHASE_SPECS.values())
    assert "Do not modify existing tests unless the issue is about them." in old
    assert "edit_file is an exact string replacement" in old
    assert old.count("Never use assert to validate input") == 1, "the one short rule, in PATCH (tests/test_agent_validation_prompts.py)"
    assert "Example call:" not in old and "exactly ONE tool call" not in old
    assert "Grep the file, then its package" not in old and "neighbouring check" not in old, "the longer convention advice is weak-only"
