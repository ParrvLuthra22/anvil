"""Textual terminal UI for ANVIL.

The :class:`~anvil.tui.app.AnvilApp` is the root Textual application.
It consumes :class:`~anvil.events.AgentEvent` objects from an
:class:`~anvil.events.EventBus` and works identically for live runs and
trace replays.
"""

from anvil.tui.app import AnvilApp

__all__ = ["AnvilApp"]
