"""PATCH and REVIEW say how validation is done: a specific exception, never assert; REVIEW asks whether the exception fits the issue.

Seen with a real model on pallets__flask-4045 ("Raise error when blueprint name contains a dot"): it used assert, which is
stripped under python -O and raises AssertionError, not the ValueError the issue asks for. weak_model_prompts is off by
default, so the rule is in the original prompts as well as in the short ones.
"""

from __future__ import annotations

import re

import pytest

from anvil.agent.prompts import PHASE_SPECS, WEAK_PHASE_SPECS
from anvil.events import Phase
from tests.test_orchestrator import execute, happy

TABLES = pytest.param(PHASE_SPECS, id="original prompts"), pytest.param(WEAK_PHASE_SPECS, id="short prompts")
QUESTION = "does the exception type and message match what the issue asks for?"
RULE = re.compile(r"If the issue asks for validation,.*?assert to validate input\.", re.DOTALL)


@pytest.mark.parametrize("table", TABLES)
def test_patch_tells_the_model_to_raise_a_specific_exception_and_never_to_assert(table):
    prompt = table[Phase.PATCH].system_prompt
    rule = RULE.search(prompt)
    assert rule, "the rule is in the PATCH prompt"
    text = rule.group(0)
    assert "raise a specific exception" in text and "ValueError" in text and "TypeError" in text
    assert "the type the issue names" in text
    assert "clear message" in text
    assert "Never use assert to validate input" in text
    assert len(text) < 260, "short: one sentence pair, not a paragraph"


@pytest.mark.parametrize("table", TABLES)
def test_review_asks_whether_the_exception_type_and_message_match_the_issue(table):
    prompt = " ".join(table[Phase.REVIEW].system_prompt.lower().split())
    assert QUESTION in prompt
    assert "never assert" in prompt, "and what a fitting exception is: a specific one, not assert"


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("phase", [Phase.UNDERSTAND, Phase.LOCALIZE, Phase.REPRODUCE, Phase.VERIFY, Phase.FINALIZE], ids=lambda p: p.value)
def test_the_rule_is_only_in_patch_and_review(table, phase):
    prompt = " ".join(table[phase].system_prompt.lower().split())
    assert "to validate input" not in prompt and QUESTION not in prompt


@pytest.mark.parametrize("weak", [False, True], ids=["default prompts", "weak_model_prompts on"])
def test_the_model_is_sent_the_rule_in_a_whole_run_whichever_table_is_in_use(tmp_path, weak):
    run = execute(happy(), tmp_path, features={"weak_model_prompts": weak})
    by_phase = {}
    for messages, _ in run.llm.calls:
        phase = re.search(r"Phase: ([A-Z]+)", messages[0]["content"]).group(1)
        by_phase.setdefault(phase, " ".join(messages[0]["content"].lower().split()))
    assert "never use assert to validate input" in by_phase["PATCH"]
    assert QUESTION in by_phase["REVIEW"]
    assert "never use assert to validate input" not in by_phase["LOCALIZE"]
