"""Tokens by phase (features.token_budgets): the Budget's per-phase counters and the table in report.md."""

from __future__ import annotations

from anvil.agent.budget import Budget, PhaseUsage
from anvil.agent.emitter import Emitter
from anvil.agent.outputs import render_report
from anvil.agent.settings import AgentSettings
from anvil.agent.state import RunState
from anvil.agent.usage import record_usage
from anvil.events import EventBus, Phase
from anvil.llm.client import LLMResponse
from tests.fakes import call, reply
from tests.test_orchestrator import execute, happy


def response(prompt: int, completion: int) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[], usage={"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion})


# ---- the counters ----------------------------------------------------------------------------------------------


def test_the_budget_adds_up_calls_and_tokens_per_phase_in_the_order_the_phases_were_first_charged():
    budget = Budget(AgentSettings())
    budget.add_phase_usage("localize", 1000, 50)
    budget.add_phase_usage("reproduce", 2000, 80)
    budget.add_phase_usage("localize", 1500, 40)
    assert list(budget.by_phase) == ["localize", "reproduce"]
    assert budget.by_phase["localize"] == PhaseUsage(calls=2, prompt_tokens=2500, completion_tokens=90)
    assert budget.by_phase["reproduce"] == PhaseUsage(calls=1, prompt_tokens=2000, completion_tokens=80)


def test_a_negative_count_is_never_subtracted():
    budget = Budget(AgentSettings())
    budget.add_phase_usage("patch", -5, -5)
    assert budget.by_phase["patch"] == PhaseUsage(calls=1, prompt_tokens=0, completion_tokens=0)


def test_record_usage_attributes_a_call_to_the_phase_the_emitter_is_in():
    budget, emitter = Budget(AgentSettings()), Emitter(EventBus())
    emitter.set_phase(Phase.PATCH)
    record_usage(response(700, 30), budget, emitter, AgentSettings())
    emitter.set_phase(Phase.VERIFY)
    record_usage(response(900, 20), budget, emitter, AgentSettings())
    assert budget.by_phase == {"patch": PhaseUsage(1, 700, 30), "verify": PhaseUsage(1, 900, 20)}
    assert budget.tokens == 1650, "the global count is unchanged by the attribution"


def test_a_call_made_outside_any_phase_is_filed_under_none_not_lost():
    budget = Budget(AgentSettings())
    record_usage(response(10, 5), budget, Emitter(EventBus()), AgentSettings())
    assert budget.by_phase == {"none": PhaseUsage(1, 10, 5)}


# ---- the table -------------------------------------------------------------------------------------------------


def render(usage) -> str:
    return render_report(RunState("https://github.com/acme/calc/issues/7"), "", steps=3, tokens=9999, seconds=1.0, phase_usage=usage)


def test_the_table_lists_each_phase_with_calls_tokens_and_the_prompt_size_per_call_and_a_total():
    report = render({"localize": PhaseUsage(3, 30_000, 900), "patch": PhaseUsage(2, 15_001, 400)})
    assert "## Tokens by phase" in report
    assert "| Phase | Calls | Prompt tokens | Completion tokens | Prompt tokens per call |" in report
    assert "|---|---:|---:|---:|---:|" in report
    assert "| localize | 3 | 30,000 | 900 | 10,000 |" in report
    assert "| patch | 2 | 15,001 | 400 | 7,500 |" in report, "7,500.5 per call, rounded"
    assert "| **total** | 5 | 45,001 | 1,300 | 9,000 |" in report


def test_the_table_sits_between_the_budget_and_the_known_limitations():
    report = render({"localize": PhaseUsage(1, 100, 10)})
    assert report.index("## Budget") < report.index("## Tokens by phase") < report.index("## Known limitations")


def test_no_usage_means_no_section():
    assert "Tokens by phase" not in render(None)
    assert "Tokens by phase" not in render({})


def test_the_report_without_the_argument_is_the_report_it_always_was():
    plain = render_report(RunState("u"), "", steps=1, tokens=1, seconds=1.0)
    assert "Tokens by phase" not in plain and "## Budget" in plain


# ---- through the orchestrator ----------------------------------------------------------------------------------


def test_a_whole_run_reports_every_phase_and_the_totals_match_the_calls_made(tmp_path):
    run = execute(happy(), tmp_path)
    table = run.report.split("## Tokens by phase")[1].split("## Known limitations")[0]
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]] for line in table.splitlines() if line.startswith("| ") and "Phase" not in line}
    assert list(rows) == ["understand", "localize", "reproduce", "patch", "verify", "review", "finalize", "**total**"]
    assert rows["understand"] == ["1", "80", "20", "80"]
    assert rows["localize"] == ["3", "240", "60", "80"]
    assert rows["**total**"] == ["15", "1,200", "300", "80"], "15 calls of 80 prompt and 20 completion tokens"
    assert run.done.data["steps"] == 15


def test_with_token_budgets_off_the_report_has_no_table(tmp_path):
    run = execute(happy(), tmp_path, features={"token_budgets": False})
    assert "Tokens by phase" not in run.report and "## Budget" in run.report
