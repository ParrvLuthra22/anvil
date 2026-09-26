"""features.nav_tools: outline and find_symbol in LOCALIZE and PATCH only; find_references never."""

from __future__ import annotations

import pytest

from anvil.agent.prompts import NAV_TOOLS, PHASE_SPECS, WEAK_PHASE_SPECS, phase_spec
from anvil.agent.settings import AgentSettings
from anvil.events import Phase
from anvil.tools.base import ToolResult
from tests.fakes import FakePipeline, FakeRegistry, FakeSandbox, FakeTool, call, default_tools, done, project_exec, project_files, reply
from tests.test_orchestrator import execute, finalize, good_patch, happy, reproduce, review_ok, understand, verify

PHASES = list(PHASE_SPECS)
NAV_PHASES = {Phase.LOCALIZE, Phase.PATCH}
ON = {"nav_tools": True}  # the feature is off by default: the tests that exercise it turn it on
NOTE = "Also available: outline(path) lists the classes and functions of a file with their line numbers"


# ---- which phases get them ----------------------------------------------------------------------------------------


def test_the_two_tools_are_outline_and_find_symbol_and_not_find_references():
    assert NAV_TOOLS == ("outline", "find_symbol")


@pytest.mark.parametrize("weak", [True, False], ids=["short prompts", "original prompts"])
@pytest.mark.parametrize("phase", PHASES, ids=lambda p: p.value)
def test_only_localize_and_patch_get_the_tools_and_only_with_the_switch_on(phase, weak):
    plain = (WEAK_PHASE_SPECS if weak else PHASE_SPECS)[phase]
    assert phase_spec(phase, weak, nav_tools=False) is plain and phase_spec(phase, weak) is plain, "off is exactly the old spec"
    with_nav = phase_spec(phase, weak, nav_tools=True)
    if phase in NAV_PHASES:
        assert with_nav.tools == plain.tools + ("outline", "find_symbol")
        assert with_nav.system_prompt == plain.system_prompt + "\n" + NOTE + ", and find_symbol(name) finds where one is defined. Use them to pick a line range before you read."
        assert "find_references" not in with_nav.tools and "find_references" not in with_nav.system_prompt
    else:
        assert with_nav is plain, "no other phase is touched"


@pytest.mark.parametrize("phase", sorted(NAV_PHASES, key=lambda p: p.value), ids=lambda p: p.value)
def test_the_tools_are_added_after_the_ones_the_phase_already_had(phase):
    plain = phase_spec(phase, True)
    assert phase_spec(phase, True, True).tools[: len(plain.tools)] == plain.tools


def test_the_switch_is_read_from_the_config_and_independent_of_the_others():
    on = AgentSettings.from_mapping({"features": {"nav_tools": True}})
    assert on.nav_tools is True and AgentSettings.from_mapping({"features": {"nav_tools": False}}).nav_tools is False
    assert on.token_budgets and on.patch_sanity and not on.weak_model_prompts, "turning it on changes nothing else"
    off_others = AgentSettings.from_mapping({"features": {"nav_tools": True, "token_budgets": False}})
    assert off_others.nav_tools is True and off_others.token_budgets is False


def test_it_is_off_by_default_at_every_level(tmp_path):
    """Pinned: nav_tools stays off until a real run shows it lowering tokens per call."""
    from anvil.agent.orchestrator import load_config

    assert AgentSettings().nav_tools is False, "the dataclass default"
    assert AgentSettings.from_mapping({}).nav_tools is False, "a config that does not mention it"
    assert AgentSettings.from_mapping({"features": {}}).nav_tools is False
    assert AgentSettings.from_mapping({"features": None}).nav_tools is False
    assert load_config()["features"]["nav_tools"] is False, "the shipped config.yaml says so explicitly"
    assert AgentSettings.from_mapping(load_config()).nav_tools is False
    run = execute(happy(), tmp_path, pipeline=nav_pipeline())  # default settings, registry that has all three tools
    for phase in ("localize", "patch"):
        assert not ({"outline", "find_symbol", "find_references"} & offered(run, phase)), "so no phase is offered them"
    assert all(NOTE not in m[0]["content"] for m, _ in run.llm.calls)


# ---- through the registry and a whole run --------------------------------------------------------------------------


def nav_registry() -> FakeRegistry:
    """The default fake tools plus the three navigation tools as Akshat's registry has them."""
    def outline(args, sb):
        return ToolResult(True, f"outline of {args.get('path')}: 2: def add(a, b)")

    def find_symbol(args, sb):
        return ToolResult(True, f"{args.get('name')} is defined at calc.py:1")

    def find_references(args, sb):
        return ToolResult(True, "calc.py:1\ntests/test_calc.py:1")

    string = lambda name: {"type": "object", "properties": {name: {"type": "string"}}, "required": [name]}  # noqa: E731
    return FakeRegistry(default_tools() + [
        FakeTool("outline", outline, parameters=string("path")),
        FakeTool("find_symbol", find_symbol, parameters=string("name")),
        FakeTool("find_references", find_references, parameters=string("name")),
    ])


def nav_pipeline(registry=None) -> FakePipeline:
    return FakePipeline(FakeSandbox(project_files(), on_exec=project_exec), registry or nav_registry())


def offered(run, phase: str) -> set[str]:
    """The tool names the model was offered in the first call of ``phase``."""
    for messages, tools in run.llm.calls:
        if f"Phase: {phase.upper()}" in messages[0]["content"]:
            return {t["function"]["name"] for t in tools}
    raise AssertionError(f"no {phase} call")


def test_a_whole_run_offers_the_tools_in_localize_and_patch_and_nowhere_else(tmp_path):
    run = execute(happy(), tmp_path, pipeline=nav_pipeline(), features=ON)
    for phase in ("localize", "patch"):
        assert {"outline", "find_symbol"} <= offered(run, phase) and "find_references" not in offered(run, phase)
    for phase in ("reproduce", "verify", "review"):
        assert not ({"outline", "find_symbol", "find_references"} & offered(run, phase))


def test_with_the_switch_off_no_phase_is_offered_any_of_them(tmp_path):
    run = execute(happy(), tmp_path, pipeline=nav_pipeline(), features={"nav_tools": False})
    for phase in ("localize", "reproduce", "patch", "verify", "review"):
        assert not ({"outline", "find_symbol", "find_references"} & offered(run, phase))
    assert all("Also available: outline" not in m[0]["content"] for m, _ in run.llm.calls)


def test_the_prompt_mentions_them_only_in_the_phases_that_have_them(tmp_path):
    run = execute(happy(), tmp_path, pipeline=nav_pipeline(), features=ON)
    with_note = {m[0]["content"].split("Phase: ")[1].split(" ")[0].split(".")[0] for m, _ in run.llm.calls if NOTE in m[0]["content"]}
    assert with_note == {"LOCALIZE", "PATCH"}


def test_a_registry_without_the_tools_just_does_not_offer_them_and_the_run_is_unaffected(tmp_path):
    run = execute(happy(), tmp_path, features=ON)  # the default fake registry has none of the three
    assert not ({"outline", "find_symbol"} & offered(run, "localize"))
    assert "Confidence: **high**" in run.report


def test_localize_can_use_the_tools(tmp_path):
    script = (
        understand()
        + [reply(call("outline", path="calc.py")), reply(call("find_symbol", name="add")), done("calc.py:2 uses a - b")]
        + reproduce() + good_patch() + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, pipeline=nav_pipeline(), features=ON)
    results = [e for e in run.of("tool_result") if e.data["tool"] in ("outline", "find_symbol")]
    assert [(e.data["tool"], e.data["ok"]) for e in results] == [("outline", True), ("find_symbol", True)]
    assert "outline of calc.py" in results[0].data["output_preview"] and "add is defined at calc.py:1" in results[1].data["output_preview"]


def test_find_references_is_refused_even_though_the_registry_has_it(tmp_path):
    script = understand() + [reply(call("find_references", name="add")), done("calc.py:2 uses a - b")] + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, pipeline=nav_pipeline(), features=ON)
    refused = next(e for e in run.of("tool_result") if e.data["tool"] == "find_references")
    assert refused.data["ok"] is False and "not available in the localize phase" in refused.data["output_preview"]


def test_the_tools_are_refused_in_a_phase_that_does_not_have_them(tmp_path):
    script = (
        understand()
        + [reply(call("grep", pattern="def add")), done("calc.py:2")]
        + [reply(call("outline", path="calc.py"))]  # REPRODUCE: not allowed
        + reproduce() + good_patch() + verify() + review_ok() + finalize()
    )
    run = execute(script, tmp_path, pipeline=nav_pipeline(), features=ON)
    refused = next(e for e in run.of("tool_result") if e.data["tool"] == "outline")
    assert refused.data["ok"] is False and "not available in the reproduce phase" in refused.data["output_preview"]
