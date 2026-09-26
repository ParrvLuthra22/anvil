"""LLM settings drawn from the parsed ``config.yaml`` plus environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from anvil.llm.errors import LLMConfigError

TOOL_MODES = ("auto", "native", "text")


@dataclass(frozen=True)
class LLMConfig:
    """Everything the client needs except the API key (which is never stored in config).

    ``tool_mode``: ``native`` uses the provider's tool-calling API, ``text``
    describes the tools in the prompt and parses a JSON block from the reply,
    ``auto`` tries native first and falls back to text permanently.
    """

    model: str
    base_url: str
    temperature: float = 0.0
    tool_mode: str = "auto"
    max_attempts: int = 5
    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 10.0
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any], env: Mapping[str, str] | None = None
    ) -> LLMConfig:
        """Build a validated config from the parsed ``config.yaml`` mapping.

        ``AI_MODEL`` and ``AI_BASE_URL`` in ``env`` (default ``os.environ``)
        override ``model`` and ``base_url``; empty values are ignored. Unknown
        keys (agent budgets, sandbox, ...) are ignored. Raises
        ``LLMConfigError`` on anything missing or invalid.
        """
        env = os.environ if env is None else env
        model = _text(_override(env, "AI_MODEL") or config.get("model"), "model", "AI_MODEL")
        base_url = _text(
            _override(env, "AI_BASE_URL") or config.get("base_url"), "base_url", "AI_BASE_URL"
        )
        if not base_url.startswith(("http://", "https://")):
            raise LLMConfigError(f"base_url must start with http:// or https://, got {base_url!r}")

        tool_mode = str(config.get("tool_mode", "auto")).strip().lower()
        if tool_mode not in TOOL_MODES:
            raise LLMConfigError(f"tool_mode must be one of {', '.join(TOOL_MODES)}; got {tool_mode!r}")

        return cls(
            model=model,
            base_url=base_url.rstrip("/"),
            temperature=_number(config, "temperature", 0.0, minimum=0.0),
            tool_mode=tool_mode,
            max_attempts=int(_number(config, "llm_max_attempts", 5, minimum=1, integer=True)),
            timeout_seconds=_number(config, "llm_timeout_seconds", 120.0, minimum=0.001),
            connect_timeout_seconds=_number(config, "llm_connect_timeout_seconds", 10.0, minimum=0.001),
            backoff_base_seconds=_number(config, "llm_backoff_base_seconds", 1.0, minimum=0.0),
            backoff_max_seconds=_number(config, "llm_backoff_max_seconds", 60.0, minimum=0.0),
        )


def _override(env: Mapping[str, str], name: str) -> str:
    """Return the stripped env override ``name`` ('' when unset or blank)."""
    return env.get(name, "").strip()


def _text(value: Any, key: str, env_var: str) -> str:
    """Return ``value`` as a stripped non-empty string or raise naming the key and env override."""
    text = str(value).strip() if value is not None else ""
    if not text:
        raise LLMConfigError(f"'{key}' is not set: define it in config.yaml or export {env_var}")
    return text


def _number(
    config: Mapping[str, Any], key: str, default: float, *, minimum: float, integer: bool = False
) -> float:
    """Read a numeric setting, rejecting booleans, non-numbers and values below ``minimum``."""
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMConfigError(f"'{key}' must be a number, got {value!r}")
    if integer and value != int(value):
        raise LLMConfigError(f"'{key}' must be a whole number, got {value!r}")
    if value < minimum:
        raise LLMConfigError(f"'{key}' must be >= {minimum}, got {value!r}")
    return float(value)
