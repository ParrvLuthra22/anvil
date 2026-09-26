"""Offline tests for src/anvil/trace/recorder.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from anvil.events import AgentEvent, Phase
from anvil.trace.recorder import TraceRecorder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    ev_type: str = "message",
    phase: Phase | None = Phase.PATCH,
    data: dict | None = None,
) -> AgentEvent:
    return AgentEvent(
        ts=time.time(),
        type=ev_type,
        phase=phase,
        data=data or {"role": "assistant", "text": "hello"},
    )


# ---------------------------------------------------------------------------
# Contract: methods exist and are callable
# ---------------------------------------------------------------------------

def test_recorder_exposes_contract_methods():
    assert callable(TraceRecorder.record)
    assert callable(TraceRecorder.load)


# ---------------------------------------------------------------------------
# Round-trip: record then load
# ---------------------------------------------------------------------------

def test_roundtrip(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    events = [
        _make_event("phase", Phase.INGEST, {"name": "ingest"}),
        _make_event("message", Phase.UNDERSTAND, {"role": "user", "text": "fix it"}),
        _make_event("tool_call", Phase.PATCH, {"tool": "edit_file", "args": {}}),
        _make_event("done", None, {"resolved_confidence": 0.9, "steps": 12}),
    ]

    rec = TraceRecorder(trace)
    for ev in events:
        rec.record(ev)
    rec.close()

    loaded = list(TraceRecorder.load(trace))
    assert len(loaded) == len(events)
    for orig, got in zip(events, loaded):
        assert got.type == orig.type
        assert got.phase == orig.phase
        assert got.data == orig.data


# ---------------------------------------------------------------------------
# Each line is flushed (file readable between writes)
# ---------------------------------------------------------------------------

def test_flush_between_writes(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    rec = TraceRecorder(trace)
    ev = _make_event()
    rec.record(ev)
    # Before close, file should already have content
    content = trace.read_text()
    assert content.strip(), "File should be flushed immediately after record()"
    rec.close()


# ---------------------------------------------------------------------------
# Parent directory is created automatically
# ---------------------------------------------------------------------------

def test_creates_parent_dirs(tmp_path: Path):
    trace = tmp_path / "deep" / "nested" / "trace.jsonl"
    rec = TraceRecorder(trace)
    rec.record(_make_event())
    rec.close()
    assert trace.exists()


# ---------------------------------------------------------------------------
# Corrupted last line is silently skipped
# ---------------------------------------------------------------------------

def test_tolerates_corrupted_last_line(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    events = [_make_event("phase", Phase.INGEST), _make_event("message", Phase.PATCH)]
    rec = TraceRecorder(trace)
    for ev in events:
        rec.record(ev)
    rec.close()

    # Append a truncated/corrupt line
    with trace.open("a") as fh:
        fh.write('{"ts": 1.0, "type": "bad", "phase": null, ')  # no closing brace

    loaded = list(TraceRecorder.load(trace))
    assert len(loaded) == 2, "Corrupt last line must be skipped silently"


# ---------------------------------------------------------------------------
# Mid-file corruption: skips bad line, loads rest
# ---------------------------------------------------------------------------

def test_tolerates_mid_file_corruption(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    good_before = _make_event("phase", Phase.INGEST)
    good_after = _make_event("done", None, {"steps": 5})

    # Write manually: good, corrupt, good
    lines = [
        json.dumps({"ts": 1.0, "type": "phase", "phase": "ingest", "data": {"name": "ingest"}}),
        "NOT_JSON_AT_ALL",
        json.dumps({"ts": 2.0, "type": "done", "phase": None, "data": {"steps": 5}}),
    ]
    trace.write_text("\n".join(lines) + "\n")

    loaded = list(TraceRecorder.load(trace))
    assert len(loaded) == 2
    assert loaded[0].type == "phase"
    assert loaded[1].type == "done"


# ---------------------------------------------------------------------------
# Phase None is preserved
# ---------------------------------------------------------------------------

def test_none_phase_roundtrip(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    ev = _make_event("done", None, {"resolved_confidence": 0.8})
    rec = TraceRecorder(trace)
    rec.record(ev)
    rec.close()

    loaded = list(TraceRecorder.load(trace))
    assert loaded[0].phase is None


# ---------------------------------------------------------------------------
# Unknown future phase name doesn't crash the loader
# ---------------------------------------------------------------------------

def test_unknown_phase_graceful(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    line = json.dumps({"ts": 1.0, "type": "phase", "phase": "future_phase", "data": {}})
    trace.write_text(line + "\n")

    loaded = list(TraceRecorder.load(trace))
    assert len(loaded) == 1
    assert loaded[0].phase is None  # unknown → None


# ---------------------------------------------------------------------------
# Context-manager interface
# ---------------------------------------------------------------------------

def test_context_manager(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    with TraceRecorder(trace) as rec:
        rec.record(_make_event())
    assert trace.exists()
    assert trace.stat().st_size > 0


# ---------------------------------------------------------------------------
# Empty file yields no events
# ---------------------------------------------------------------------------

def test_empty_file(tmp_path: Path):
    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    loaded = list(TraceRecorder.load(trace))
    assert loaded == []


# ---------------------------------------------------------------------------
# Live trace: queue drain flushes to disk mid-run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_trace_readable_mid_run(tmp_path: Path):
    import asyncio
    from anvil.events import EventBus

    bus = EventBus()
    trace = tmp_path / "live_trace.jsonl"
    rec = TraceRecorder(trace)
    q = bus.subscribe()

    async def _drain():
        while True:
            ev = await q.get()
            rec.record(ev)
            if ev.type == "done":
                break
        rec.close()

    drain_task = asyncio.create_task(_drain())

    # Emit first event
    bus.emit(_make_event("phase", Phase.INGEST))
    await asyncio.sleep(0.02)

    # Read mid-run: first line must be present and valid
    lines_mid = trace.read_text().strip().splitlines()
    assert len(lines_mid) == 1, "Trace file should have 1 line mid-run"

    # Emit second event
    bus.emit(_make_event("message", Phase.INGEST))
    await asyncio.sleep(0.02)

    lines_mid2 = trace.read_text().strip().splitlines()
    assert len(lines_mid2) == 2, "Trace file should have 2 lines mid-run"

    # Finish run
    bus.emit(_make_event("done", None, {"resolved_confidence": 1.0}))
    await drain_task

    loaded = list(TraceRecorder.load(trace))
    assert len(loaded) == 3


@pytest.mark.asyncio
async def test_entrypoint_recorder_writes_trace_before_run_finishes(tmp_path: Path):
    """The production recorder wiring flushes events while a run is active."""
    import asyncio

    from anvil.__main__ import _make_event_bus, _wire_recorder

    bus = _make_event_bus()
    trace = tmp_path / "output" / "trace.jsonl"
    _wire_recorder(bus, trace)

    bus.emit(_make_event("phase", Phase.INGEST, {"name": "ingest"}))
    # Let the actual _wire_recorder drain task consume and flush the event.
    for _ in range(20):
        await asyncio.sleep(0.005)
        if trace.exists() and trace.stat().st_size:
            break
    assert len(trace.read_text().splitlines()) == 1

    bus.emit(_make_event("done", None, {"steps": 1}))
    for _ in range(20):
        await asyncio.sleep(0.005)
        if len(trace.read_text().splitlines()) == 2:
            break
    assert len(list(TraceRecorder.load(trace))) == 2


def test_main_make_event_bus_returns_events_event_bus():
    """_make_event_bus() in __main__.py returns anvil.events.EventBus."""
    from anvil.__main__ import _make_event_bus
    from anvil.events import EventBus

    bus = _make_event_bus()
    assert isinstance(bus, EventBus)
