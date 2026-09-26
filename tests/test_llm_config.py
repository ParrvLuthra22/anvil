from pathlib import Path

import pytest
import yaml

from anvil.llm.config import LLMConfig
from anvil.llm.errors import LLMConfigError

BASE = {"model": "m1", "base_url": "https://api.example.com/v1/"}


def test_defaults_and_trailing_slash_stripped():
    cfg = LLMConfig.from_mapping(BASE, env={})
    assert cfg.model == "m1"
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.temperature == 0.0
    assert cfg.tool_mode == "auto"
    assert cfg.max_attempts == 5


def test_env_overrides_model_and_base_url():
    env = {"AI_MODEL": "other", "AI_BASE_URL": "https://alt.example.com/"}
    cfg = LLMConfig.from_mapping(BASE, env=env)
    assert (cfg.model, cfg.base_url) == ("other", "https://alt.example.com")


def test_empty_env_override_is_ignored():
    cfg = LLMConfig.from_mapping(BASE, env={"AI_MODEL": "", "AI_BASE_URL": "   "})
    assert cfg.model == "m1"
    assert cfg.base_url == "https://api.example.com/v1"


def test_uses_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "from-env")
    assert LLMConfig.from_mapping(BASE).model == "from-env"


def test_overrides_are_read_from_mapping():
    cfg = LLMConfig.from_mapping(
        {**BASE, "temperature": 0.3, "tool_mode": "TEXT", "llm_max_attempts": 2,
         "llm_timeout_seconds": 30, "llm_backoff_base_seconds": 0.5, "llm_backoff_max_seconds": 4},
        env={},
    )
    assert cfg.temperature == 0.3
    assert cfg.tool_mode == "text"
    assert cfg.max_attempts == 2
    assert cfg.timeout_seconds == 30.0
    assert (cfg.backoff_base_seconds, cfg.backoff_max_seconds) == (0.5, 4.0)


def test_unknown_keys_are_ignored():
    assert LLMConfig.from_mapping({**BASE, "sandbox": "auto", "max_total_steps": 120}, env={})


@pytest.mark.parametrize(
    "bad",
    [
        {"model": "", "base_url": "https://x"},
        {"base_url": "https://x"},
        {"model": "m"},
        {"model": "m", "base_url": "ftp://x"},
        {**BASE, "tool_mode": "magic"},
        {**BASE, "llm_max_attempts": 0},
        {**BASE, "llm_max_attempts": 2.5},
        {**BASE, "llm_max_attempts": True},
        {**BASE, "llm_timeout_seconds": "fast"},
        {**BASE, "temperature": -1},
    ],
)
def test_invalid_config_raises_typed_error(bad):
    with pytest.raises(LLMConfigError):
        LLMConfig.from_mapping(bad, env={})


def test_missing_model_error_names_the_env_override():
    with pytest.raises(LLMConfigError, match="AI_MODEL"):
        LLMConfig.from_mapping({"base_url": "https://x"}, env={})


def test_repository_config_yaml_is_valid():
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = LLMConfig.from_mapping(yaml.safe_load(path.read_text()), env={})
    assert cfg.tool_mode == "auto"
    assert cfg.temperature == 0.0
