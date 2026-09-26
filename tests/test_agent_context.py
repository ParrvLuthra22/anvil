"""The orchestrator's use of the ContextManager: pinned brief, phase compression, budget, summariser."""

from __future__ import annotations

import pytest

from anvil.agent.prompts import SUMMARIZER_PROMPT
from anvil.agent.settings import AgentSettings
from anvil.context.tokens import estimate_messages_tokens
from anvil.llm.client import LLMResponse
from anvil.llm.errors import LLMError
from tests.fakes import FakePipeline, FakeSandbox, RecordingLLM, call, done, project_exec, project_files, reply
from tests.test_orchestrator import (
    execute,
    finalize,
    good_patch,
    happy,
    localize,
    reproduce,
    review_ok,
    understand,
    verify,
    wrong_patch,
)

BIG_FILE = "\n".join(f"def helper_{i}(x):\n    return x + {i}  # padding to make the line longer" for i in range(150))


def system_of(messages: list[dict]) -> str:
    return messages[0]["content"]


def calls_in_phase(run, phase: str) -> list[list[dict]]:
    return [messages for messages, _ in run.llm.calls if f"Phase: {phase.upper()}" in system_of(messages)]


class SummarisingLLM(RecordingLLM):
    """A RecordingLLM that answers the context summariser's calls itself, so scripts only hold the agent's replies."""

    def __init__(self, script, *, fail: bool = False) -> None:
        super().__init__(script)
        self.fail = fail
        self.summary_transcripts: list[str] = []

    def chat(self, messages, tools=None):
        if system_of(messages) == SUMMARIZER_PROMPT:
            self.summary_transcripts.append(messages[1]["content"])
            if self.fail:
                raise LLMError("provider exploded", retryable=True, attempts=5)
            return LLMResponse("Read helper functions in big.py; no bug there.", [], {"total_tokens": 40})
        return super().chat(messages, tools)

    @property
    def remaining(self) -> int:
        return super().remaining + len(self.summary_transcripts)


def wandering_localize(reads: int) -> list:
    """A LOCALIZE phase that reads the big file over and over before finding the bug."""
    steps = [reply(call("read_file", path="big.py", start=1, end=60 + i)) for i in range(reads)]
    return steps + [reply(call("grep", pattern="def add")), reply(call("read_file", path="calc.py"))] + [
        done("calc.py:2 uses a - b")
    ]


def big_project_pipeline() -> FakePipeline:
    sandbox = FakeSandbox({**project_files(), "big.py": BIG_FILE}, on_exec=project_exec)
    return FakePipeline(sandbox)


def wandering_script(reads: int) -> list:
    return understand() + wandering_localize(reads) + reproduce() + good_patch() + verify() + review_ok() + finalize()


# ---- what the model is shown ----------------------------------------------------------------


def test_the_repo_map_is_pinned_as_its_own_message_after_the_issue_brief(tmp_path):
    run = execute(happy(), tmp_path)
    messages = run.llm.calls[0][0]
    assert [m["role"] for m in messages] == ["system", "user", "user", "user"]
    brief, repo_map, kickoff = (m["content"] for m in messages[1:])
    assert "add() returns the wrong sum" in brief and "Repository overview" not in brief
    assert repo_map == "## Repository overview\ncalc.py\ntests/test_calc.py"
    assert kickoff.startswith("Begin phase UNDERSTAND")


def test_a_finished_phase_reaches_later_phases_only_as_its_summary(tmp_path):
    run = execute(happy(), tmp_path)
    (first_patch_call, *_) = calls_in_phase(run, "patch")
    contents = [m["content"] for m in first_patch_call]
    assert [m["role"] for m in first_patch_call if m["role"] in ("assistant", "tool")] == []
    assert "[UNDERSTAND summary]\nadd() subtracts instead of adding" in contents
    assert "[LOCALIZE summary]\ncalc.py:2 uses a - b" in contents
    assert any(c.startswith("[REPRODUCE summary]") for c in contents)
    assert not any("calc.py:1:def add" in c for c in contents)  # LOCALIZE's raw grep output is gone
    assert not any("write_repro" in str(m.get("tool_calls", "")) for m in first_patch_call)


def test_the_phase_in_progress_keeps_its_own_full_transcript(tmp_path):
    run = execute(happy(), tmp_path)
    last_patch_call = calls_in_phase(run, "patch")[-1]
    assert [m["role"] for m in last_patch_call[-4:]] == ["assistant", "tool", "assistant", "tool"]
    assert "Replaced 1 occurrence" in last_patch_call[-3]["content"]


def test_the_diff_is_pinned_from_the_first_phase_after_the_patch_exists(tmp_path):
    run = execute(happy(), tmp_path)

    def diff_messages(messages):
        return [m["content"] for m in messages if m["content"].startswith("[Diff of the working tree")]

    assert all(not diff_messages(m) for m in calls_in_phase(run, "patch")[:1])
    for phase in ("verify", "review"):
        (diff,) = diff_messages(calls_in_phase(run, phase)[0])
        assert "+    return a + b" in diff and ".anvil" not in diff


def test_no_phase_after_a_rollback_is_shown_the_discarded_patch(tmp_path):
    wrong = wrong_patch("return a - b", "return a * b", "used multiplication")
    script = understand() + localize() + reproduce() + (wrong + verify()) * 6 + finalize()
    run = execute(script, tmp_path)
    patch_calls = calls_in_phase(run, "patch")
    rethink = [m for m in patch_calls if any("DIFFERENT hypothesis" in x["content"] for x in m)]
    assert rethink
    assert not any(x["content"].startswith("[Diff of the working tree") for x in rethink[0])


# ---- the budget -----------------------------------------------------------------------------


def test_every_prompt_of_a_wandering_run_fits_max_context_tokens(tmp_path):
    budget = 2500
    llm = SummarisingLLM(wandering_script(20))
    run = execute(
        None, tmp_path, llm=llm, pipeline=big_project_pipeline(),
        max_context_tokens=budget, context_keep_steps=4, tool_output_char_cap=1500,
    )
    assert run.report and "+    return a + b" in run.patch
    assert len(run.llm.calls) > 30
    sizes = [estimate_messages_tokens(messages) for messages, _ in run.llm.calls]
    assert max(sizes) <= budget, f"largest prompt ~{max(sizes)} tokens"
    assert max(sizes) > budget * 0.6, "the test never came close to the budget"


def test_old_tool_output_is_shown_as_one_line_summaries(tmp_path):
    run = execute(
        None, tmp_path, llm=RecordingLLM(wandering_script(12)), pipeline=big_project_pipeline(), context_keep_steps=3,
        features={"token_budgets": False},  # this test scripts a 12-call LOCALIZE: the classic phase limits
    )
    last_localize = [m for m in calls_in_phase(run, "localize")][-1]
    results = [m["content"] for m in last_localize if m["role"] == "tool"]
    pruned = [r for r in results if r.startswith("[output pruned] read_file(path='big.py'")]
    assert len(pruned) >= 8 and all("\n" not in r for r in pruned)
    assert all(not r.startswith("[output pruned]") for r in results[-3:])


def test_the_model_never_sees_more_than_the_output_cap_of_one_tool_result(tmp_path):
    run = execute(None, tmp_path, llm=RecordingLLM(wandering_script(3)), pipeline=big_project_pipeline(), tool_output_char_cap=600)
    biggest = max(len(m["content"]) for messages, _ in run.llm.calls for m in messages if m["role"] == "tool")
    assert biggest <= 600
    assert any("lines omitted]" in m["content"] for messages, _ in run.llm.calls for m in messages if m["role"] == "tool")


# ---- the summariser -------------------------------------------------------------------------


def test_the_summariser_is_a_real_llm_call_charged_to_the_run(tmp_path):
    llm = SummarisingLLM(wandering_script(20))
    run = execute(
        None, tmp_path, llm=llm, pipeline=big_project_pipeline(),
        max_context_tokens=2500, context_keep_steps=8, tool_output_char_cap=1500, features={"token_budgets": False},
    )
    assert llm.summary_transcripts, "the context never filled up"
    assert "big.py" in llm.summary_transcripts[0]
    shown = [c for _, c in [(m, x["content"]) for m, _ in run.llm.calls for x in m] if c.startswith("[Summary of earlier work]")]
    assert shown and "no bug there" in shown[-1]
    agent_calls, summary_calls = len(run.llm.calls), len(llm.summary_transcripts)
    assert run.done.data["steps"] == agent_calls + summary_calls
    assert len(run.of("llm_usage")) == agent_calls + summary_calls
    assert run.done.data["tokens"] == 100 * agent_calls + 40 * summary_calls
    assert any(e.data["role"] == "system" and "summarising" in e.data["text"] for e in run.of("message"))
    assert not run.of("error")
    assert "+    return a + b" in run.patch


def test_a_failing_summariser_costs_nothing_but_a_warning_and_the_run_still_succeeds(tmp_path):
    llm = SummarisingLLM(wandering_script(20), fail=True)
    run = execute(
        None, tmp_path, llm=llm, pipeline=big_project_pipeline(),
        max_context_tokens=2500, context_keep_steps=8, tool_output_char_cap=1500, features={"token_budgets": False},
    )
    assert llm.summary_transcripts
    errors = run.of("error")
    assert errors and all(e.data["kind"] == "context" for e in errors)
    assert "could not summarise old history" in errors[0].data["message"]
    assert "+    return a + b" in run.patch
    assert "Confidence: **high**" in run.report
    sizes = [estimate_messages_tokens(messages) for messages, _ in run.llm.calls]
    assert max(sizes) <= 2500
    assert any(
        m["content"].startswith("[Earlier work was dropped") for messages, _ in run.llm.calls for m in messages
    )


def test_a_run_that_never_fills_the_context_never_calls_the_summariser(tmp_path):
    llm = SummarisingLLM(happy())
    run = execute(None, tmp_path, llm=llm)
    assert llm.summary_transcripts == []
    assert run.done.data["steps"] == len(run.llm.calls)


# ---- settings -------------------------------------------------------------------------------


def test_context_settings_have_sensible_defaults_and_come_from_config_yaml():
    from anvil.agent.orchestrator import load_config

    settings = AgentSettings.from_mapping(load_config())
    assert settings.max_context_tokens == 32_000
    assert settings.context_keep_steps == 3, "the token_saving value: features.token_budgets is on by default"
    assert settings.context_summarize_threshold == 0.75
    assert AgentSettings.from_mapping({}) == AgentSettings()


@pytest.mark.parametrize(
    "bad",
    [
        {"max_context_tokens": 10},
        {"max_context_tokens": "big"},
        {"max_context_tokens": 4000.5},
        {"context_keep_steps": 0},
        {"context_summarize_threshold": 0},
        {"context_summarize_threshold": 1.5},
        {"context_summarize_threshold": True},
    ],
)
def test_invalid_context_settings_raise_naming_the_key(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        AgentSettings.from_mapping(bad)
