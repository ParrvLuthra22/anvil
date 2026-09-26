from pathlib import Path

import pytest

from anvil.agent.orchestrator import load_config

REQUIRED_KEYS = {
    "model", "base_url", "temperature", "max_steps_per_phase", "max_total_steps",
    "max_tokens_total", "wall_clock_seconds", "tool_output_char_cap", "sandbox",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)


def test_repo_config_has_all_contract_keys():
    config = load_config()
    assert REQUIRED_KEYS <= config.keys()
    assert config["temperature"] == 0
    assert config["sandbox"] == "auto"


def test_env_overrides_model_and_base_url(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "other-model")
    monkeypatch.setenv("AI_BASE_URL", "http://localhost:1234/v1")
    config = load_config()
    assert config["model"] == "other-model"
    assert config["base_url"] == "http://localhost:1234/v1"


def test_empty_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "")
    assert load_config()["model"] != ""


def test_load_config_never_exposes_api_key(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "secret")
    assert "secret" not in str(load_config().values())


def test_load_config_from_explicit_path(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model: m\nbase_url: u\n", encoding="utf-8")
    assert load_config(cfg) == {"model": "m", "base_url": "u"}
