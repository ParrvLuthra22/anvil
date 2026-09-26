"""Offline tests for src/anvil/tui/."""

from __future__ import annotations

import anvil.tui
from anvil.tui.app import AnvilApp


def test_tui_package_has_docstring():
    """The TUI package must have a module docstring (contract from test_tui.py)."""
    assert anvil.tui.__doc__, "anvil.tui must have a module docstring"


def test_tui_exports_app():
    """AnvilApp must be importable from the tui package."""
    assert AnvilApp is not None


def test_anvil_app_is_class():
    """AnvilApp must be a class (not an instance)."""
    import inspect
    assert inspect.isclass(AnvilApp)


def test_anvil_app_has_required_bindings():
    """AnvilApp must declare q/r/d key bindings."""
    keys = {b.key for b in AnvilApp.BINDINGS}
    assert "q" in keys, "q (quit) binding missing"
    assert "r" in keys, "r (restart) binding missing"
    assert "d" in keys, "d (toggle diff) binding missing"
