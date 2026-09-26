"""Tests for src/anvil/tui/ — covering:

* Module / contract checks (no Textual runtime needed)
* Unit tests for event → UI-state logic (using AnvilApp.pilot())
* Unit tests for _clip() helper
* Replay speed / pause state
* PhaseStepper state machine
* Error banner visibility
* Result banner population
* Unknown / malformed events don't crash the app
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

import anvil.tui
from anvil.events import AgentEvent, Phase
from anvil.tui.app import (
    AnvilApp,
    DiffPane,
    ErrorBanner,
    MetricsPanel,
    PhaseStepper,
    ResultBanner,
    _clip,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_bus():
    """Return a minimal concrete EventBus for testing."""
    from anvil.events import EventBus

    class _TestBus(EventBus):
        def __init__(self):
            self._queues = []

        def emit(self, event):
            for q in self._queues:
                q.put_nowait(event)

        def subscribe(self):
            q = asyncio.Queue()
            self._queues.append(q)
            return q

    return _TestBus()


def _ev(
    ev_type: str,
    phase: Phase | None = None,
    data: dict | None = None,
    ts: float | None = None,
) -> AgentEvent:
    return AgentEvent(
        ts=ts or time.time(),
        type=ev_type,
        phase=phase,
        data=data or {},
    )


def _make_app(**kwargs) -> AnvilApp:
    bus = kwargs.pop("bus", _make_bus())
    return AnvilApp(bus=bus, config={}, **kwargs)


# ===========================================================================
# 1. Module / contract (no Textual runtime)
# ===========================================================================

def test_tui_package_has_docstring():
    assert anvil.tui.__doc__, "anvil.tui must have a module docstring"


def test_tui_exports_app():
    from anvil.tui import AnvilApp as A
    assert A is AnvilApp


def test_anvil_app_is_class():
    import inspect
    assert inspect.isclass(AnvilApp)


def test_anvil_app_has_required_bindings():
    keys = {b.key for b in AnvilApp.BINDINGS}
    assert "q"     in keys, "q (quit) binding missing"
    assert "r"     in keys, "r (restart) binding missing"
    assert "d"     in keys, "d (toggle diff) binding missing"
    assert "space" in keys, "space (pause) binding missing"


# ===========================================================================
# 2. _clip() helper
# ===========================================================================

def test_clip_short_text():
    assert _clip("hello") == "hello"


def test_clip_long_text():
    long = "x" * 500
    result = _clip(long, 100)
    # Should be clipped + ellipsis markup
    assert len(result) < 500
    assert "…" in result


def test_clip_exact_limit():
    text = "a" * 400
    result = _clip(text, 400)
    assert result == text  # exactly at limit → no clip


# ===========================================================================
# 3. PhaseStepper state machine (widget unit tests, no full app needed)
# ===========================================================================

@pytest.mark.asyncio
async def test_phase_stepper_marks_active():
    """PhaseStepper.mark_active sets correct state."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        stepper = app.query_one("#phase-stepper", PhaseStepper)
        stepper.mark_active(Phase.PATCH)
        assert stepper._states[Phase.PATCH] == "active"


@pytest.mark.asyncio
async def test_phase_stepper_marks_all_done():
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        stepper = app.query_one("#phase-stepper", PhaseStepper)
        stepper.mark_active(Phase.VERIFY)
        stepper.mark_all_done()
        for phase in [Phase.INGEST, Phase.PROFILE, Phase.VERIFY, Phase.FINALIZE]:
            assert stepper._states[phase] == "done"


@pytest.mark.asyncio
async def test_phase_stepper_reset():
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        stepper = app.query_one("#phase-stepper", PhaseStepper)
        stepper.mark_active(Phase.PATCH)
        stepper.reset()
        for phase in stepper._states.values():
            assert phase == "pending"


# ===========================================================================
# 4. Event → UI-state mapping (via _handle_event)
# ===========================================================================

@pytest.mark.asyncio
async def test_phase_event_updates_stepper():
    """A 'phase' event sets the stepper to active."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("phase", Phase.LOCALIZE, {"name": "localize"})
        app._handle_event(ev)
        stepper = app.query_one("#phase-stepper", PhaseStepper)
        assert stepper._states[Phase.LOCALIZE] == "active"
        assert app._current_phase == Phase.LOCALIZE


@pytest.mark.asyncio
async def test_error_event_shows_banner():
    """An 'error' event makes the ErrorBanner visible."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.PATCH, {"kind": "llm", "message": "LLM gave up"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" in banner.classes


@pytest.mark.asyncio
async def test_new_phase_clears_error_banner():
    """A 'phase' event clears the error banner (run recovered)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        app._handle_event(_ev("error", Phase.PATCH, {"kind": "x", "message": "y"}))
        app._handle_event(_ev("phase", Phase.VERIFY, {"name": "verify"}))
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" not in banner.classes


# ===========================================================================
# 4b. Severity mapping (Item 4 — kinds per src/anvil/agent/NOTES.md)
# ===========================================================================

def test_is_fatal_for_truly_fatal_kinds():
    """llm, budget, internal are the only fatal kinds."""
    from anvil.tui.app import _is_fatal
    for kind in ("llm", "budget", "internal"):
        assert _is_fatal(kind) is True, f"Expected {kind!r} to be fatal"


def test_is_fatal_false_for_all_benign_kinds():
    """All real benign/recovery kinds (per NOTES.md) must NOT be fatal."""
    from anvil.tui.app import _is_fatal
    benign = [
        "loop", "edit", "invalid_call", "no_tool_call", "test_failure",
        "timeout", "tool", "deps", "sandbox", "context", "rollback",
        "config", "io", "finalize",
        # phase names used as kind
        "ingest", "profile", "understand", "localize", "reproduce",
        "patch", "verify", "review",
    ]
    for kind in benign:
        assert _is_fatal(kind) is False, f"Expected {kind!r} to be non-fatal"


def test_is_fatal_false_is_case_insensitive():
    """Kind matching must be case-insensitive."""
    from anvil.tui.app import _is_fatal
    assert _is_fatal("LOOP") is False
    assert _is_fatal("LLM") is True


@pytest.mark.asyncio
async def test_benign_loop_does_not_show_banner():
    """A 'loop' error (benign) must NOT show the ErrorBanner."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.PATCH, {"kind": "loop", "message": "loop detected"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" not in banner.classes, "Benign 'loop' error should not show red banner"


@pytest.mark.asyncio
async def test_benign_invalid_call_does_not_show_banner():
    """'invalid_call' is benign — no red banner."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.LOCALIZE, {"kind": "invalid_call", "message": "unknown tool"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" not in banner.classes


@pytest.mark.asyncio
async def test_benign_no_tool_call_does_not_show_banner():
    """'no_tool_call' is benign — no red banner."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.UNDERSTAND, {"kind": "no_tool_call", "message": "no tool"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" not in banner.classes


@pytest.mark.asyncio
async def test_fatal_llm_shows_banner():
    """'llm' is fatal — must show the red ErrorBanner."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.PATCH, {"kind": "llm", "message": "LLM gave up"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" in banner.classes, "'llm' error must show red banner"


@pytest.mark.asyncio
async def test_fatal_budget_shows_banner():
    """'budget' is fatal — must show the red ErrorBanner."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.VERIFY, {"kind": "budget", "message": "token budget exceeded"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" in banner.classes, "'budget' error must show red banner"


@pytest.mark.asyncio
async def test_fatal_internal_shows_banner():
    """'internal' is fatal — must show the red ErrorBanner."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("error", Phase.PATCH, {"kind": "internal", "message": "unexpected exception"})
        app._handle_event(ev)
        banner = app.query_one("#error-banner", ErrorBanner)
        assert "visible" in banner.classes


@pytest.mark.asyncio
async def test_done_event_shows_result_banner():
    """A 'done' event makes the ResultBanner visible with correct data."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev(
            "done",
            Phase.FINALIZE,
            {
                "resolved_confidence": 0.88,
                "patch_path": "output/patch.diff",
                "report_path": "output/report.md",
                "steps": 42,
                "tokens": 12345,
                "seconds": 90.5,
            },
        )
        app._handle_event(ev)
        banner = app.query_one("#result-banner", ResultBanner)
        assert "visible" in banner.classes
        # Check content — use .content (Textual 8 API)
        line1_text = str(app.query_one("#result-line1", Label).content)
        assert "88%" in line1_text
        assert "42" in line1_text


@pytest.mark.asyncio
async def test_llm_usage_updates_metrics():
    """An 'llm_usage' event increments the metrics panel."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        metrics = app.query_one("#metrics-panel", MetricsPanel)
        before = metrics._steps
        ev = _ev("llm_usage", Phase.UNDERSTAND, {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_estimate": 0.0001,
        })
        app._handle_event(ev)
        assert metrics._steps == before + 1
        assert metrics._prompt_tok == 100
        assert metrics._completion_tok == 50


@pytest.mark.asyncio
async def test_unknown_event_type_does_not_crash():
    """Unknown event types are silently ignored — the app must not raise."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = _ev("UNKNOWN_FUTURE_EVENT_TYPE_XYZ", None, {"foo": "bar"})
        # Must not raise
        app._handle_event(ev)


@pytest.mark.asyncio
async def test_event_with_none_data_does_not_crash():
    """An event with data=None must not crash the handler."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = AgentEvent(ts=time.time(), type="message", phase=None, data=None)
        app._handle_event(ev)  # must not raise


@pytest.mark.asyncio
async def test_event_with_malformed_data_does_not_crash():
    """An event with data as a string (not dict) must not crash."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        ev = AgentEvent(ts=time.time(), type="tool_result", phase=Phase.PATCH,
                        data="not a dict at all")  # type: ignore
        app._handle_event(ev)


@pytest.mark.asyncio
async def test_very_long_line_does_not_crash():
    """A message with 10 000 chars is clipped and rendered without error."""
    app = _make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        ev = _ev("message", Phase.PATCH, {"role": "assistant", "text": "A" * 10_000})
        app._handle_event(ev)


# ===========================================================================
# 5. Replay state
# ===========================================================================

@pytest.mark.asyncio
async def test_replay_starts_paused_when_space_pressed():
    """action_toggle_pause toggles replay pause state."""
    from tests.mock_llm import fake_event_stream
    events = fake_event_stream()
    bus = _make_bus()
    app = AnvilApp(
        bus=bus,
        replay=True,
        replay_events=events,
        replay_speed=1.0,
        config={},
    )
    # Call action methods directly — pilot.press("space") goes to the Input
    # widget which has focus and consumes the keystroke before the app sees it.
    async with app.run_test(size=(120, 40)) as pilot:
        assert app._replay_paused is False
        app.action_toggle_pause()
        assert app._replay_paused is True
        app.action_toggle_pause()
        assert app._replay_paused is False


@pytest.mark.asyncio
async def test_replay_speed_keys():
    """Speed actions change replay speed."""
    from tests.mock_llm import fake_event_stream
    events = fake_event_stream()
    bus = _make_bus()
    app = AnvilApp(
        bus=bus,
        replay=True,
        replay_events=events,
        replay_speed=1.0,
        config={},
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_speed_4()
        assert app._replay_speed == 4.0
        app.action_speed_instant()
        assert app._replay_speed == 0.0
        app.action_speed_1()
        assert app._replay_speed == 1.0


@pytest.mark.asyncio
async def test_step_forward_in_replay():
    """action_step_forward advances by one event in paused replay mode."""
    from tests.mock_llm import fake_event_stream
    events = fake_event_stream()
    bus = _make_bus()
    app = AnvilApp(
        bus=bus,
        replay=True,
        replay_events=events,
        replay_speed=1.0,
        config={},
    )
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_toggle_pause()   # pause the replay
        idx_before = app._replay_idx
        app.action_step_forward()
        assert app._replay_idx == idx_before + 1


# ===========================================================================
# 6. MockLLM contract
# ===========================================================================

def test_mock_llm_replays_in_order():
    from anvil.llm.client import LLMResponse
    from tests.mock_llm import MockLLM

    responses = [
        LLMResponse(text="first",  tool_calls=[], usage={"total_tokens": 10}),
        LLMResponse(text="second", tool_calls=[], usage={"total_tokens": 20}),
    ]
    mock = MockLLM(responses)
    assert mock.chat([]).text == "first"
    assert mock.chat([]).text == "second"


def test_mock_llm_raises_when_exhausted():
    from anvil.llm.client import LLMResponse
    from tests.mock_llm import MockLLM

    mock = MockLLM([LLMResponse(text="x", tool_calls=[], usage={})])
    mock.chat([])
    with pytest.raises(RuntimeError, match="exhausted"):
        mock.chat([])


def test_fake_event_stream_covers_all_phases():
    from tests.mock_llm import fake_event_stream

    events = fake_event_stream()
    phase_events = [e for e in events if e.type == "phase"]
    phases_seen = {e.phase for e in phase_events}
    expected = set(Phase)
    assert expected == phases_seen, f"Missing phases: {expected - phases_seen}"


def test_fake_event_stream_ends_with_done():
    from tests.mock_llm import fake_event_stream

    events = fake_event_stream()
    assert events[-1].type == "done"


def test_fake_event_stream_has_tool_calls():
    from tests.mock_llm import fake_event_stream

    events = fake_event_stream()
    tool_calls = [e for e in events if e.type == "tool_call"]
    assert len(tool_calls) >= 5, "Expected at least one tool call per phase"


# ===========================================================================
# Import Label for type annotation in test
# ===========================================================================
from textual.widgets import Label  # noqa: E402 — must be after pytest imports
