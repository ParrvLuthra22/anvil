"""Model profiles: defaults by model family, and how config.yaml overrides them."""

import json
from dataclasses import replace

import httpx
import pytest

from anvil.agent.orchestrator import load_config
from anvil.agent.settings import AgentSettings
from anvil.llm import LLMConfigError, ModelProfile, profile_for
from anvil.llm.config import LLMConfig
from anvil.llm.profiles import DEFAULT_MAX_CONTEXT_TOKENS, PROFILES, resolve_profile
from tests.test_llm_client import CONFIG, USER, Server, _env, build, ok  # noqa: F401 (_env: autouse)

BASE = {"model": "m", "base_url": "https://llm.example.com/v1"}


# ---- which profile a model name gets ---------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("deepseek-chat", "deepseek"),
        ("deepseek-coder", "deepseek"),
        ("deepseek-ai/DeepSeek-V3", "deepseek"),
        ("accounts/fireworks/models/deepseek-v3", "deepseek"),
        ("DEEPSEEK-CHAT", "deepseek"),
        ("deepseek-reasoner", "deepseek-reasoning"),
        ("deepseek/deepseek-r1", "deepseek-reasoning"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-reasoning"),
        ("Qwen/Qwen2.5-Coder-32B-Instruct", "qwen"),
        ("qwen-plus", "qwen"),
        ("qwen-max", "qwen"),
        ("qwen3-coder-plus", "qwen"),
        ("accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", "qwen"),
        ("qwq-32b", "qwen-reasoning"),
        ("Qwen/Qwen3-32B", "qwen-reasoning"),
        ("qwen3-235b-a22b", "qwen-reasoning"),
        ("gemini-2.0-flash", "default"),
        ("gpt-4o", "default"),
        ("llama-3.3-70b-versatile", "default"),
        ("kimi-k2-thinking", "default"),
        ("", "default"),
    ],
)
def test_the_profile_is_chosen_from_the_model_name(model, expected):
    assert profile_for(model).name == expected


def test_a_profile_can_be_forced_by_name_whatever_the_model_is_called():
    assert profile_for("my-finetune", "qwen").name == "qwen"
    assert profile_for("deepseek-chat", "default").name == "default"
    assert profile_for("deepseek-chat", "AUTO").name == "deepseek"
    assert profile_for("gpt-4o", None).name == "default"


def test_an_unknown_profile_name_is_an_error_that_lists_the_choices():
    with pytest.raises(ValueError, match="model_profile must be one of auto, default, deepseek"):
        profile_for("gpt-4o", "gpt")


def test_every_profile_is_conservative_about_context():
    assert DEFAULT_MAX_CONTEXT_TOKENS == 32_000
    assert {p.max_context_tokens for p in PROFILES.values()} == {32_000}
    assert all(p.tool_mode == "auto" and p.strip_reasoning for p in PROFILES.values())


def test_reasoning_models_get_no_output_cap_and_the_recommended_temperature():
    for name in ("deepseek-reasoning", "qwen-reasoning"):
        assert PROFILES[name].max_output_tokens is None and PROFILES[name].temperature == 0.6
    assert PROFILES["deepseek"].max_output_tokens == 8192 and PROFILES["deepseek"].temperature == 0.0
    assert PROFILES["qwen"].max_output_tokens == 4096 and PROFILES["qwen"].temperature == 0.0, "the largest reply seen was 541 tokens"


# ---- config.yaml wins over the profile ---------------------------------------------------------------


def config_for(model, **extra):
    return LLMConfig.from_mapping({**BASE, "model": model, **extra}, env={})


def test_a_profile_fills_in_what_the_config_leaves_out():
    cfg = config_for("deepseek-reasoner")
    assert (cfg.profile, cfg.temperature, cfg.max_output_tokens, cfg.tool_mode, cfg.strip_reasoning) == (
        "deepseek-reasoning", 0.6, None, "auto", True
    )
    cfg = config_for("qwen-plus")
    assert (cfg.profile, cfg.temperature, cfg.max_output_tokens) == ("qwen", 0.0, 4096)


def test_explicit_config_values_beat_the_profile():
    cfg = config_for("deepseek-reasoner", temperature=0.2, tool_mode="text", max_output_tokens=1000, strip_reasoning=False)
    assert (cfg.temperature, cfg.tool_mode, cfg.max_output_tokens, cfg.strip_reasoning) == (0.2, "text", 1000, False)


def test_a_temperature_of_zero_in_the_config_is_an_explicit_choice_not_unset():
    assert config_for("deepseek-reasoner", temperature=0).temperature == 0.0
    assert config_for("deepseek-reasoner").temperature == 0.6


def test_max_output_tokens_null_means_the_providers_default_not_the_profiles():
    assert config_for("qwen-plus", max_output_tokens=None).max_output_tokens is None
    assert config_for("qwen-plus").max_output_tokens == 4096


def test_model_profile_in_the_config_forces_a_profile():
    cfg = config_for("my-finetune", model_profile="qwen")
    assert (cfg.profile, cfg.max_output_tokens) == ("qwen", 4096)


def test_an_unknown_model_profile_in_the_config_is_a_config_error():
    with pytest.raises(LLMConfigError, match="model_profile"):
        config_for("gpt-4o", model_profile="gpt")


def test_the_env_model_override_picks_the_profile_too():
    cfg = LLMConfig.from_mapping({**BASE, "model": "gpt-4o"}, env={"AI_MODEL": "deepseek-chat"})
    assert cfg.profile == "deepseek" and cfg.model == "deepseek-chat"
    assert resolve_profile({"model": "gpt-4o"}, {"AI_MODEL": "qwen-max"}).name == "qwen"
    assert resolve_profile({"model": "gpt-4o"}, {"AI_MODEL": "  "}).name == "default"


# ---- it reaches the request ------------------------------------------------------------------------------


def body_for(model, **overrides):
    server = Server(ok("hi"))
    client, _ = build(server, model=model, **overrides)
    client.chat(USER)
    return server.body()


def test_a_chat_model_gets_an_output_cap_in_its_requests():
    assert body_for("deepseek-chat")["max_tokens"] == 8192
    assert body_for("Qwen/Qwen2.5-Coder-32B-Instruct")["max_tokens"] == 4096


def test_a_reasoning_model_is_not_capped_and_an_unknown_model_gets_nothing_new():
    assert "max_tokens" not in body_for("deepseek-reasoner")
    assert set(body_for("gemini-2.0-flash")) == {"model", "messages", "temperature"}


def test_the_shipped_temperature_pin_is_sent_even_to_a_reasoning_model():
    """config.yaml ships `temperature: 0`; that is an explicit setting, so it wins (the probe warns about it)."""
    body = body_for("deepseek-reasoner")
    assert body["temperature"] == 0.0


def test_without_a_temperature_in_the_config_a_reasoning_model_runs_at_the_profile_temperature():
    server = Server(ok("hi"))
    config = {k: v for k, v in CONFIG.items() if k != "temperature"}
    from anvil.llm import make_client

    client = make_client({**config, "model": "qwq-32b"}, transport=httpx.MockTransport(server), sleep=lambda s: None)
    client.chat(USER)
    assert json.loads(server.requests[0].content)["temperature"] == 0.6


# ---- the agent's context budget follows the profile too -------------------------------------------------------


def test_the_context_budget_defaults_to_the_profile_and_config_overrides_it(monkeypatch):
    monkeypatch.setitem(PROFILES, "deepseek", replace(PROFILES["deepseek"], max_context_tokens=64_000))
    assert AgentSettings.from_mapping({"model": "deepseek-chat"}).max_context_tokens == 64_000
    assert AgentSettings.from_mapping({"model": "deepseek-chat", "max_context_tokens": 20_000}).max_context_tokens == 20_000
    assert AgentSettings.from_mapping({"model": "gpt-4o"}).max_context_tokens == 32_000


def test_a_bad_model_profile_is_a_settings_error_the_orchestrator_already_handles():
    with pytest.raises(ValueError, match="model_profile"):
        AgentSettings.from_mapping({"model": "gpt-4o", "model_profile": "gpt"})


def test_settings_without_a_model_still_build():
    assert AgentSettings.from_mapping({}) == AgentSettings()


def test_the_shipped_config_resolves_cleanly():
    config = load_config()
    assert config["model_profile"] == "auto"
    llm = LLMConfig.from_mapping(config, env={})
    assert llm.profile == "default" and llm.temperature == 0.0 and llm.strip_reasoning is True
    assert AgentSettings.from_mapping(config).max_context_tokens == 32_000


def test_the_model_profile_type_is_exported():
    assert isinstance(profile_for("x"), ModelProfile)


def test_the_qwen_cap_can_be_raised_or_removed_in_the_config_without_a_code_edit():
    assert config_for("qwen3-coder", max_output_tokens=8192).max_output_tokens == 8192
    assert config_for("qwen3-coder", max_output_tokens=None).max_output_tokens is None
    assert body_for("qwen3-coder", max_output_tokens=8192)["max_tokens"] == 8192


def test_the_shipped_qwen_profile_stays_below_what_a_free_tier_refused():
    """OpenRouter's free tier answered HTTP 402 to max_tokens=8192 and accepted 4096."""
    assert PROFILES["qwen"].max_output_tokens < 8192

