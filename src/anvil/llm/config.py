"""LLM settings drawn from the parsed ``config.yaml`` plus environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

from anvil.llm.errors import LLMConfigError
from anvil.llm.profiles import ModelProfile, profile_for

TOOL_MODES = ("auto", "native", "text")
# Request fields the client owns; ``llm_extra_params`` may not override them.
RESERVED_PARAMS = frozenset({"model", "messages", "tools", "stream"})


@dataclass(frozen=True)
class LLMConfig:
    """Everything the client needs except the API key (which is never stored in config).

    ``tool_mode``: ``native`` uses the provider's tool-calling API, ``text``
    describes the tools in the prompt and parses a JSON block from the reply,
    ``auto`` tries native first and falls back to text permanently.
    ``strip_reasoning`` removes ``<think>`` blocks and reasoning fields from replies.
    ``max_output_tokens`` is sent as ``max_tokens`` (``None``: the provider's default). ``extra_params`` are added to
    every request body as they are (provider-specific switches such as ``top_p`` or ``enable_thinking``).
    """

    model: str
    base_url: str
    profile: str = "default"
    temperature: float = 0.0
    tool_mode: str = "auto"
    strip_reasoning: bool = True
    max_output_tokens: int | None = None
    extra_params: Mapping[str, Any] = field(default_factory=dict)
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

        try:
            profile = profile_for(model, _profile_name(config))
        except ValueError as exc:
            raise LLMConfigError(str(exc)) from None

        tool_mode = str(config.get("tool_mode", profile.tool_mode)).strip().lower()
        if tool_mode not in TOOL_MODES:
            raise LLMConfigError(f"tool_mode must be one of {', '.join(TOOL_MODES)}; got {tool_mode!r}")

        return cls(
            model=model,
            base_url=base_url.rstrip("/"),
            profile=profile.name,
            temperature=_number(config, "temperature", profile.temperature, minimum=0.0),
            tool_mode=tool_mode,
            strip_reasoning=_flag(config, "strip_reasoning", profile.strip_reasoning),
            max_output_tokens=_optional_count(config, "max_output_tokens", profile),
            extra_params=_extra_params(config),
            max_attempts=int(_number(config, "llm_max_attempts", 5, minimum=1, integer=True)),
            timeout_seconds=_number(config, "llm_timeout_seconds", 120.0, minimum=0.001),
            connect_timeout_seconds=_number(config, "llm_connect_timeout_seconds", 10.0, minimum=0.001),
            backoff_base_seconds=_number(config, "llm_backoff_base_seconds", 1.0, minimum=0.0),
            backoff_max_seconds=_number(config, "llm_backoff_max_seconds", 60.0, minimum=0.0),
        )


def _profile_name(config: Mapping[str, Any]) -> str | None:
    value = config.get("model_profile")
    return None if value is None else str(value)


def _optional_count(config: Mapping[str, Any], key: str, profile: ModelProfile) -> int | None:
    """A positive whole number. Absent: the profile's default. Present but null: unset (the provider's default)."""
    if key not in config:
        return profile.max_output_tokens
    if config[key] is None:
        return None
    return int(_number(config, key, 1, minimum=1, integer=True))


def _extra_params(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("llm_extra_params")
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise LLMConfigError(f"'llm_extra_params' must be a mapping of request fields, got {value!r}")
    reserved = sorted(RESERVED_PARAMS & set(value))
    if reserved:
        raise LLMConfigError(f"'llm_extra_params' may not set {', '.join(reserved)}: the client owns those fields")
    return dict(value)


def _flag(config: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read a boolean setting, rejecting anything that is not a real boolean (``"false"`` would be truthy)."""
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise LLMConfigError(f"'{key}' must be true or false, got {value!r}")
    return value


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
