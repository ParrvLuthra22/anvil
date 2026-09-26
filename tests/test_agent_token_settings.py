"""The ``features`` switches and the ``token_saving`` numbers in config.yaml, and how they change the effective settings."""

import pytest

from anvil.agent.orchestrator import load_config
from anvil.agent.settings import DEFAULT_PHASE_CALLS, AgentSettings

PHASES = ["understand", "localize", "reproduce", "patch", "verify", "review"]


def settings(**config) -> AgentSettings:
    return AgentSettings.from_mapping(config)


def test_both_features_are_on_by_default_and_in_the_shipped_config():
    assert all(getattr(settings(), flag) is True for flag in ("token_budgets", "weak_model_prompts", "patch_sanity", "nav_tools"))
    shipped = AgentSettings.from_mapping(load_config())
    assert all(getattr(shipped, flag) is True for flag in ("token_budgets", "weak_model_prompts", "patch_sanity", "nav_tools"))
    assert load_config()["features"] == {"token_budgets": True, "weak_model_prompts": True, "patch_sanity": True, "nav_tools": True}


def test_the_shipped_phase_caps_are_the_agreed_ones():
    assert dict(AgentSettings.from_mapping(load_config()).phase_call_caps) == {
        "understand": 1, "localize": 8, "reproduce": 10, "patch": 15, "verify": 8, "review": 3,
    }
    assert DEFAULT_PHASE_CALLS == dict(settings().phase_call_caps)


def test_with_token_budgets_on_the_context_settings_are_tightened_never_loosened():
    on = settings(tool_output_char_cap=8000, context_keep_steps=6)
    assert (on.tool_output_char_cap, on.context_keep_steps) == (4000, 3)
    already_tighter = settings(tool_output_char_cap=1000, context_keep_steps=2)
    assert (already_tighter.tool_output_char_cap, already_tighter.context_keep_steps) == (1000, 2)


def test_with_token_budgets_off_the_context_settings_are_exactly_the_configured_ones_and_no_cap_applies():
    off = settings(features={"token_budgets": False}, tool_output_char_cap=8000, context_keep_steps=6)
    assert (off.tool_output_char_cap, off.context_keep_steps) == (8000, 6)
    assert off.token_budgets is False
    assert [off.call_cap(phase) for phase in PHASES] == [None] * 6


def test_call_cap_by_phase_name():
    on = settings()
    assert [on.call_cap(p) for p in PHASES] == [1, 8, 10, 15, 8, 3]
    assert on.call_cap("finalize") is None and on.call_cap("ingest") is None


def test_the_numbers_can_be_changed_in_the_config():
    custom = settings(token_saving={
        "read_file_max_lines": 80, "read_file_head_lines": 20, "context_keep_steps": 2,
        "tool_output_char_cap": 2500, "repo_map_chars": 1500, "phase_calls": {"localize": 4, "patch": 6},
    })
    assert (custom.read_file_max_lines, custom.read_file_head_lines, custom.repo_map_chars) == (80, 20, 1500)
    assert (custom.context_keep_steps, custom.tool_output_char_cap) == (2, 2500)
    assert custom.call_cap("localize") == 4 and custom.call_cap("patch") == 6
    assert custom.call_cap("review") == 3, "a phase left out keeps its default"


def test_features_can_be_switched_off_one_at_a_time():
    assert settings(features={"weak_model_prompts": False}).token_budgets is True
    assert settings(features={"weak_model_prompts": False}).weak_model_prompts is False
    assert settings(features={"token_budgets": False}).weak_model_prompts is True
    assert settings(features=None).token_budgets is True, "an empty features: key in YAML is not an error"


@pytest.mark.parametrize(
    "config, message",
    [
        ({"features": {"token_budgets": "yes"}}, "features.token_budgets"),
        ({"features": {"weak_model_prompts": 1}}, "features.weak_model_prompts"),
        ({"features": ["token_budgets"]}, "'features' must be a mapping"),
        ({"token_saving": {"read_file_max_lines": 5}}, "token_saving.read_file_max_lines"),
        ({"token_saving": {"read_file_max_lines": 100, "read_file_head_lines": 101}}, "must not exceed"),
        ({"token_saving": {"tool_output_char_cap": 50}}, "token_saving.tool_output_char_cap"),
        ({"token_saving": {"phase_calls": {"localise": 4}}}, "unknown phases"),
        ({"token_saving": {"phase_calls": {"localize": 0}}}, "token_saving.phase_calls.localize"),
        ({"token_saving": {"phase_calls": {"localize": 2.5}}}, "whole number"),
        ({"token_saving": {"phase_calls": [1, 2]}}, "phase_calls"),
        ({"token_saving": "lots"}, "'token_saving' must be a mapping"),
    ],
)
def test_bad_values_are_refused_with_the_key_named(config, message):
    with pytest.raises(ValueError, match=message):
        AgentSettings.from_mapping(config)


def test_a_broken_config_still_gives_usable_defaults_through_the_fallback():
    fallback = AgentSettings.fallback({"output_dir": "out", "features": "nonsense"})
    assert fallback.output_dir == "out" and fallback.token_budgets is True


def test_with_token_budgets_off_and_the_keys_absent_the_old_defaults_apply():
    off = settings(features={"token_budgets": False})
    assert (off.tool_output_char_cap, off.context_keep_steps) == (8000, 6)
