"""Typed agent settings drawn from the parsed ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from anvil.context.manager import (
    DEFAULT_KEEP_STEPS,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_SUMMARIZE_THRESHOLD,
)
from anvil.llm.profiles import resolve_profile

# Model calls a phase may make before the harness forces it to close (``features.token_budgets``).
DEFAULT_PHASE_CALLS: dict[str, int] = {"understand": 1, "localize": 8, "reproduce": 10, "patch": 15, "verify": 8, "review": 3}
DEFAULT_READ_FILE_MAX_LINES = 150
DEFAULT_READ_FILE_HEAD_LINES = 60
DEFAULT_TOKEN_SAVING_KEEP_STEPS = 3
DEFAULT_TOKEN_SAVING_OUTPUT_CAP = 4000
DEFAULT_REPO_MAP_CHARS = 3000
_FEATURES = ("token_budgets", "weak_model_prompts", "patch_sanity")
_LEGACY_OUTPUT_CAP = 8000  # the value used for tool_output_char_cap when it is absent and token_budgets is off


@dataclass(frozen=True)
class AgentSettings:
    """Budgets, recovery limits and output locations for one run.

    The LLM's own settings (model, base URL, retries) live in ``LLMConfig``;
    unknown keys in the mapping are ignored so both can share ``config.yaml``.
    """

    max_steps_per_phase: int = 25
    max_total_steps: int = 120
    max_tokens_total: int = 1_500_000
    wall_clock_seconds: float = 1800.0
    tool_output_char_cap: int = DEFAULT_TOKEN_SAVING_OUTPUT_CAP  # what the default (token_budgets on) resolves to
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    context_keep_steps: int = DEFAULT_TOKEN_SAVING_KEEP_STEPS
    context_summarize_threshold: float = DEFAULT_SUMMARIZE_THRESHOLD
    max_patch_attempts: int = 3
    max_rollbacks: int = 2
    command_timeout_seconds: int = 120
    output_dir: str = "output"
    cost_per_million_prompt_tokens: float = 0.0
    cost_per_million_completion_tokens: float = 0.0
    token_budgets: bool = True
    weak_model_prompts: bool = True
    patch_sanity: bool = True
    read_file_max_lines: int = DEFAULT_READ_FILE_MAX_LINES
    read_file_head_lines: int = DEFAULT_READ_FILE_HEAD_LINES
    repo_map_chars: int = DEFAULT_REPO_MAP_CHARS
    phase_call_caps: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_PHASE_CALLS))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> AgentSettings:
        """Build validated settings, using the defaults above for absent keys.

        Raises ``ValueError`` naming the key when a value has the wrong type or
        is out of range.
        """
        output_dir = config.get("output_dir", cls.output_dir)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError(f"'output_dir' must be a non-empty string, got {output_dir!r}")
        features = _features(config)
        saving = _token_saving(config)
        output_cap = _whole(config, "tool_output_char_cap", _LEGACY_OUTPUT_CAP, 200)
        keep_steps = _whole(config, "context_keep_steps", DEFAULT_KEEP_STEPS, 1)
        if features["token_budgets"]:  # never looser than what was configured, only tighter
            output_cap = min(output_cap, saving["tool_output_char_cap"])
            keep_steps = min(keep_steps, saving["context_keep_steps"])
        return cls(
            token_budgets=features["token_budgets"],
            weak_model_prompts=features["weak_model_prompts"],
            patch_sanity=features["patch_sanity"],
            read_file_max_lines=saving["read_file_max_lines"],
            read_file_head_lines=saving["read_file_head_lines"],
            repo_map_chars=saving["repo_map_chars"],
            phase_call_caps=saving["phase_calls"],
            max_steps_per_phase=_whole(config, "max_steps_per_phase", cls.max_steps_per_phase, 1),
            max_total_steps=_whole(config, "max_total_steps", cls.max_total_steps, 1),
            max_tokens_total=_whole(config, "max_tokens_total", cls.max_tokens_total, 1),
            wall_clock_seconds=_number(config, "wall_clock_seconds", cls.wall_clock_seconds, 0),
            tool_output_char_cap=output_cap,
            max_context_tokens=_whole(
                config, "max_context_tokens", resolve_profile(config).max_context_tokens, 1000
            ),
            context_keep_steps=keep_steps,
            context_summarize_threshold=_number(
                config, "context_summarize_threshold", cls.context_summarize_threshold, 0.1, 1.0
            ),
            max_patch_attempts=_whole(config, "max_patch_attempts", cls.max_patch_attempts, 1),
            max_rollbacks=_whole(config, "max_rollbacks", cls.max_rollbacks, 0),
            command_timeout_seconds=_whole(config, "command_timeout_seconds", cls.command_timeout_seconds, 1),
            output_dir=output_dir.strip(),
            cost_per_million_prompt_tokens=_number(
                config, "cost_per_million_prompt_tokens", cls.cost_per_million_prompt_tokens, 0
            ),
            cost_per_million_completion_tokens=_number(
                config, "cost_per_million_completion_tokens", cls.cost_per_million_completion_tokens, 0
            ),
        )

    @classmethod
    def fallback(cls, config: Mapping[str, Any]) -> AgentSettings:
        """Defaults for a config that failed validation, keeping ``output_dir`` if that one is usable.

        Where the results go matters even when some other key is broken: the run still owes
        the caller its patch and report.
        """
        output_dir = config.get("output_dir")
        if isinstance(output_dir, str) and output_dir.strip():
            return cls(output_dir=output_dir.strip())
        return cls()

    def call_cap(self, phase: str) -> int | None:
        """Model calls ``phase`` may make before the harness forces it to close, or ``None`` when the cap is off."""
        return self.phase_call_caps.get(phase) if self.token_budgets else None

    def cost_estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimated USD cost of one model call (0.0 unless prices are configured)."""
        return (
            prompt_tokens * self.cost_per_million_prompt_tokens
            + completion_tokens * self.cost_per_million_completion_tokens
        ) / 1_000_000


def _features(config: Mapping[str, Any]) -> dict[str, bool]:
    """The ``features`` switches: each is on unless set to false; anything but a true/false value is an error."""
    section = config.get("features")
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"'features' must be a mapping of switches, got {section!r}")
    flags = {}
    for name in _FEATURES:
        value = section.get(name, True)
        if not isinstance(value, bool):
            raise ValueError(f"'features.{name}' must be true or false, got {value!r}")
        flags[name] = value
    return flags


def _token_saving(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``token_saving`` numbers (used when ``features.token_budgets`` is on), validated, with defaults."""
    section = config.get("token_saving")
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"'token_saving' must be a mapping, got {section!r}")
    values = {
        "read_file_max_lines": _whole(section, "read_file_max_lines", DEFAULT_READ_FILE_MAX_LINES, 20, "token_saving."),
        "read_file_head_lines": _whole(section, "read_file_head_lines", DEFAULT_READ_FILE_HEAD_LINES, 0, "token_saving."),
        "context_keep_steps": _whole(section, "context_keep_steps", DEFAULT_TOKEN_SAVING_KEEP_STEPS, 1, "token_saving."),
        "tool_output_char_cap": _whole(section, "tool_output_char_cap", DEFAULT_TOKEN_SAVING_OUTPUT_CAP, 200, "token_saving."),
        "repo_map_chars": _whole(section, "repo_map_chars", DEFAULT_REPO_MAP_CHARS, 800, "token_saving."),
    }
    if values["read_file_head_lines"] > values["read_file_max_lines"]:
        raise ValueError("'token_saving.read_file_head_lines' must not exceed 'token_saving.read_file_max_lines'")
    calls = section.get("phase_calls", {})
    if not isinstance(calls, Mapping):
        raise ValueError(f"'token_saving.phase_calls' must be a mapping of phase to call count, got {calls!r}")
    unknown = sorted(set(calls) - set(DEFAULT_PHASE_CALLS))
    if unknown:
        raise ValueError(f"'token_saving.phase_calls' names unknown phases {unknown}; known: {sorted(DEFAULT_PHASE_CALLS)}")
    values["phase_calls"] = {
        phase: _whole(calls, phase, default, 1, "token_saving.phase_calls.") for phase, default in DEFAULT_PHASE_CALLS.items()
    }
    return values


def _number(
    config: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float | None = None, prefix: str = ""
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{prefix}{key}' must be a number, got {value!r}")
    if value < minimum:
        raise ValueError(f"'{prefix}{key}' must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"'{prefix}{key}' must be <= {maximum}, got {value!r}")
    return float(value)


def _whole(config: Mapping[str, Any], key: str, default: int, minimum: int, prefix: str = "") -> int:
    value = _number(config, key, default, minimum, prefix=prefix)
    if value != int(value):
        raise ValueError(f"'{prefix}{key}' must be a whole number, got {config.get(key)!r}")
    return int(value)
