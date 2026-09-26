"""Typed agent settings drawn from the parsed ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
    tool_output_char_cap: int = 8000
    max_patch_attempts: int = 3
    max_rollbacks: int = 2
    command_timeout_seconds: int = 120
    output_dir: str = "output"
    cost_per_million_prompt_tokens: float = 0.0
    cost_per_million_completion_tokens: float = 0.0

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> AgentSettings:
        """Build validated settings, using the defaults above for absent keys.

        Raises ``ValueError`` naming the key when a value has the wrong type or
        is out of range.
        """
        output_dir = config.get("output_dir", cls.output_dir)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError(f"'output_dir' must be a non-empty string, got {output_dir!r}")
        return cls(
            max_steps_per_phase=_whole(config, "max_steps_per_phase", cls.max_steps_per_phase, 1),
            max_total_steps=_whole(config, "max_total_steps", cls.max_total_steps, 1),
            max_tokens_total=_whole(config, "max_tokens_total", cls.max_tokens_total, 1),
            wall_clock_seconds=_number(config, "wall_clock_seconds", cls.wall_clock_seconds, 0),
            tool_output_char_cap=_whole(config, "tool_output_char_cap", cls.tool_output_char_cap, 200),
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

    def cost_estimate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimated USD cost of one model call (0.0 unless prices are configured)."""
        return (
            prompt_tokens * self.cost_per_million_prompt_tokens
            + completion_tokens * self.cost_per_million_completion_tokens
        ) / 1_000_000


def _number(config: Mapping[str, Any], key: str, default: float, minimum: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number, got {value!r}")
    if value < minimum:
        raise ValueError(f"'{key}' must be >= {minimum}, got {value!r}")
    return float(value)


def _whole(config: Mapping[str, Any], key: str, default: int, minimum: int) -> int:
    value = _number(config, key, default, minimum)
    if value != int(value):
        raise ValueError(f"'{key}' must be a whole number, got {config.get(key)!r}")
    return int(value)
