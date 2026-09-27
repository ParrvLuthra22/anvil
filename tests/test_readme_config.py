"""Keep the README's documented feature defaults and phase caps in sync."""

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _readme_value(readme: str, key: str) -> str:
    """Return the Default cell for a configuration key in the README table."""
    row = next(
        (line for line in readme.splitlines() if line.startswith(f"| `{key}` |")),
        None,
    )
    assert row is not None, f"README configuration table is missing {key}"
    return row.split("|", 3)[2].strip().strip("`")


def test_readme_feature_defaults_match_config_yaml():
    """The listed feature defaults must be the shipped YAML values."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert _readme_value(readme, "model") == config["model"]
    assert _readme_value(readme, "base_url") == config["base_url"]

    for feature in ("weak_model_prompts", "nav_tools", "token_budgets", "patch_sanity"):
        configured = str(config["features"][feature]).lower()
        assert _readme_value(readme, f"features.{feature}") == configured

    assert config["features"] == {
        "token_budgets": True,
        "weak_model_prompts": False,
        "patch_sanity": True,
        "nav_tools": False,
    }


def test_readme_phase_call_caps_match_config_yaml():
    """The README must show every configured phase cap and its value."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documented = _readme_value(readme, "token_saving.phase_calls")
    parsed = dict(
        (phase, int(calls))
        for phase, calls in re.findall(r"([a-z_]+)\s*=\s*(\d+)", documented)
    )
    assert parsed == config["token_saving"]["phase_calls"]
    assert config["token_saving"]["phase_calls"]["reproduce"] == 10
    assert config["token_saving"]["phase_calls"]["review"] == 6


def test_readme_hard_patch_attempt_limit_matches_config_yaml():
    """The README distinguishes the run-wide PATCH limit from rollback retries."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert _readme_value(readme, "max_total_patch_attempts") == str(config["max_total_patch_attempts"])
    assert config["max_total_patch_attempts"] == 3
    assert "Hard limit of 3 PATCH attempts across the entire run" in readme


def test_readme_documents_qwen_output_and_http_402_retry_policy():
    """Document the current model-profile default and split 402 behavior."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Qwen defaults to 4096 tokens" in readme
    assert "up to 5 attempts with jittered waits from 2 to 30 seconds" in readme
    assert "Other HTTP 402" in readme and "fail without retrying" in readme
