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
    assert config["sandbox"] == "worktree", "docker must stay opt-in: `auto` would pick an untested, network-less backend"


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


def test_the_shipped_default_never_selects_docker_by_itself(tmp_path):
    """Regression: with `sandbox: auto` a running Docker daemon silently switched the run to DockerSandbox."""
    from anvil.sandbox import make_sandbox

    config = load_config()
    assert config["sandbox"] == "worktree"
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    import os

    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "i"], cwd=tmp_path, check=True, env={**os.environ, **env})
    sandbox = make_sandbox(config, tmp_path.resolve(), None)
    try:
        assert type(sandbox).__name__ == "WorktreeSandbox"
    finally:
        sandbox.close()
