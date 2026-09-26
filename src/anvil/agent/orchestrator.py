"""Entry point that drives one full run from issue URL to patch and report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from anvil.events import EventBus

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"

_ENV_OVERRIDES = {"AI_BASE_URL": "base_url", "AI_MODEL": "model"}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (or ``path``) and apply the AI_BASE_URL / AI_MODEL env overrides.

    The API key is deliberately not part of the returned dict; the LLM client
    reads it straight from ``AI_API_KEY``.
    """
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}
    for env_var, key in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            config[key] = value
    return config


def run_harness(issue_url: str, config: dict, bus: EventBus) -> None:
    """Run the full phase cycle for ``issue_url``, emitting events on ``bus``."""
    raise NotImplementedError
