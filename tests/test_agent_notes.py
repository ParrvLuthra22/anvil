"""src/anvil/agent/NOTES.md feeds the project docs: it must keep matching the code it describes."""

from pathlib import Path

import pytest

from anvil.agent.orchestrator import load_config
from anvil.agent.recovery import LOOP_REPEATS, LOOP_STRIKE_LIMIT, MAX_SILENT_REPLIES, ErrorClass
from anvil.agent.settings import AgentSettings

NOTES = (Path(__file__).resolve().parents[1] / "src" / "anvil" / "agent" / "NOTES.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("error_class", list(ErrorClass), ids=lambda c: c.value)
def test_every_error_class_is_described(error_class):
    assert f"`{error_class.value}`" in NOTES


def test_the_shipped_context_defaults_are_the_ones_the_notes_state():
    settings = AgentSettings.from_mapping(load_config())
    assert f"`max_context_tokens` {settings.max_context_tokens}" in NOTES
    assert f"`context_keep_steps` {settings.context_keep_steps}" in NOTES
    assert f"`context_summarize_threshold` {settings.context_summarize_threshold}" in NOTES
    assert f"`tool_output_char_cap` {settings.tool_output_char_cap}" in NOTES


def test_the_recovery_limits_are_the_ones_the_notes_state():
    settings = AgentSettings.from_mapping(load_config())
    assert LOOP_REPEATS == 3 and LOOP_STRIKE_LIMIT == 3 and MAX_SILENT_REPLIES == 1
    assert "3 times in a row" in NOTES and "Strike n of 3" in NOTES and "nudged once" in NOTES
    assert f"`max_patch_attempts`, {settings.max_patch_attempts}" in NOTES
    assert f"`max_rollbacks`, {settings.max_rollbacks}" in NOTES
    assert f"`max_total_patch_attempts`, {settings.max_total_patch_attempts}" in NOTES and "`max_total_patch_attempts` (3" in NOTES


def test_the_config_keys_the_pipeline_reads_are_in_config_yaml():
    config = load_config()
    assert config["install_dependencies"] is True
    assert config["sandbox"] == "worktree"
