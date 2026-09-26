"""Per-model-family defaults: how to talk to DeepSeek, Qwen, or anything else.

A ``ModelProfile`` is a bundle of defaults chosen by the model's name. Anything set in ``config.yaml`` wins over the
profile; the profile only fills in what the config leaves out. ``model_profile`` in the config forces one by name
(``deepseek``, ``deepseek-reasoning``, ``qwen``, ``qwen-reasoning``, ``default``) instead of guessing from the model name.

Matching is by substring anywhere in the name, because hosted models carry prefixes:
``deepseek-chat``, ``deepseek-ai/DeepSeek-V3``, ``deepseek/deepseek-r1`` (OpenRouter),
``accounts/fireworks/models/qwen3-coder``, ``Qwen/Qwen2.5-Coder-32B-Instruct``, ``qwq-32b``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_MAX_CONTEXT_TOKENS = 32_000  # conservative on purpose: prompts above the real window fail outright


@dataclass(frozen=True)
class ModelProfile:
    """Defaults for a family of models."""

    name: str
    tool_mode: str = "auto"
    """``native``, ``text`` or ``auto`` (try native, fall back to text for good)."""
    temperature: float = 0.0
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    """Budget for the prompt; the context manager keeps the history under it."""
    max_output_tokens: int | None = None
    """Sent as ``max_tokens``; ``None`` leaves the provider's default (often small: 1.5k to 4k)."""
    strip_reasoning: bool = True
    """Remove ``<think>`` blocks and reasoning fields from replies."""


PROFILES: dict[str, ModelProfile] = {
    "default": ModelProfile("default"),
    # Chat models: their default reply limit is small (DeepSeek 4k, DashScope Qwen about 2k), and a whole-file edit needs more.
    "deepseek": ModelProfile("deepseek", max_output_tokens=8192),
    "qwen": ModelProfile("qwen", max_output_tokens=8192),
    # Reasoning models think before answering, sometimes for tens of thousands of tokens: a small cap would cut the thinking
    # off and leave no answer, so the provider's own limit applies. Vendors recommend 0.6; greedy decoding makes them loop.
    "deepseek-reasoning": ModelProfile("deepseek-reasoning", temperature=0.6),
    "qwen-reasoning": ModelProfile("qwen-reasoning", temperature=0.6),
}

# Checked in order; the first that matches the model name wins (so the reasoning variants come before their families).
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"deepseek.*(?:r1|reasoner|reasoning)", re.IGNORECASE), "deepseek-reasoning"),
    (re.compile(r"deepseek", re.IGNORECASE), "deepseek"),
    (re.compile(r"qwq|qwen.*thinking|qwen[-_]?3(?!.*coder)", re.IGNORECASE), "qwen-reasoning"),
    (re.compile(r"qwen", re.IGNORECASE), "qwen"),
)

PROFILE_CHOICES = ("auto", *PROFILES)


def profile_for(model: str, override: str | None = None) -> ModelProfile:
    """The profile for ``model``, or the one named by ``override`` (``None`` or ``"auto"``: guess from the name).

    Raises ``ValueError`` for an unknown profile name.
    """
    name = (override or "auto").strip().lower()
    if name == "auto":
        for pattern, profile_name in _PATTERNS:
            if pattern.search(model or ""):
                return PROFILES[profile_name]
        return PROFILES["default"]
    if name not in PROFILES:
        raise ValueError(f"model_profile must be one of {', '.join(PROFILE_CHOICES)}; got {override!r}")
    return PROFILES[name]


def resolve_profile(config: Mapping[str, Any], env: Mapping[str, str] | None = None) -> ModelProfile:
    """The profile that applies to a parsed ``config.yaml``: its ``model`` (``AI_MODEL`` overrides it) and ``model_profile``."""
    env = os.environ if env is None else env
    model = (env.get("AI_MODEL", "").strip() or str(config.get("model") or "")).strip()
    override = config.get("model_profile")
    return profile_for(model, None if override is None else str(override))
