"""Entry point: ``python -m anvil``.

Modes
-----
TUI (default)::

    python -m anvil
    python -m anvil --issue https://github.com/owner/repo/issues/42

Demo (fake event stream, no API key needed)::

    python -m anvil --demo

Headless (no TUI, prints result)::

    python -m anvil --issue <url> --headless
    python -m anvil --repo <repo-url> --issue-text "description" --headless

Replay a saved trace in the TUI::

    python -m anvil replay output/trace.jsonl
    python -m anvil replay output/trace.jsonl --speed 4
    python -m anvil replay output/trace.jsonl --speed 0   # instant
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anvil",
        description="ANVIL — autonomous coding-agent harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  make demo                            # TUI demo, no key needed\n"
            "  make run ISSUE=https://github.com/…  # live run\n"
            "  python -m anvil replay output/trace.jsonl --speed 4\n"
        ),
    )

    sub = parser.add_subparsers(dest="subcommand")

    # ── replay sub-command ────────────────────────────────────────────────────
    replay_p = sub.add_parser(
        "replay",
        help="Replay a saved trace.jsonl through the TUI (no API calls needed)",
    )
    replay_p.add_argument(
        "trace",
        type=Path,
        help="Path to the JSONL trace file to replay",
    )
    replay_p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="SPEED",
        help="Replay speed multiplier: 1=real-time  4=fast  0=instant (default: 1)",
    )

    # ── Top-level flags (live / demo / headless) ──────────────────────────────
    parser.add_argument(
        "--issue",
        metavar="URL",
        help="GitHub issue URL (e.g. https://github.com/owner/repo/issues/42)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the TUI; print the result to stdout",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Play the fake event stream in the TUI — no API key needed",
    )
    parser.add_argument(
        "--repo",
        metavar="URL",
        help="Repository URL (fallback when GitHub issue fetch fails)",
    )
    parser.add_argument(
        "--issue-text",
        metavar="TEXT",
        help="Issue description text (used with --repo as a fallback)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to config.yaml (default: repo-root config.yaml)",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_api_key() -> str:
    """Return the API key or exit with a clear, friendly message."""
    key = os.environ.get("AI_API_KEY", "").strip()
    if not key:
        print(
            "\n"
            "  ┌──────────────────────────────────────────────────────────┐\n"
            "  │  ERROR: AI_API_KEY is not set.                           │\n"
            "  │                                                          │\n"
            "  │  Fix:  export AI_API_KEY=<your-openai-compatible-key>   │\n"
            "  │        python -m anvil --issue <url>                     │\n"
            "  │                                                          │\n"
            "  │  Demo mode (no key):  python -m anvil --demo            │\n"
            "  └──────────────────────────────────────────────────────────┘\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def _output_paths() -> tuple[Path, Path, Path]:
    """Ensure output/ exists and return (patch, report, trace) paths."""
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    return out / "patch.diff", out / "report.md", out / "trace.jsonl"


# ---------------------------------------------------------------------------
# Concrete EventBus implementation
# ---------------------------------------------------------------------------

def _make_event_bus():
    """Return a concrete EventBus that fans out to all subscriber queues."""
    from anvil.events import AgentEvent, EventBus

    class _Bus(EventBus):
        def __init__(self) -> None:
            self._queues: list[asyncio.Queue] = []

        def emit(self, event: AgentEvent) -> None:
            for q in self._queues:
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

        def subscribe(self) -> asyncio.Queue:
            q: asyncio.Queue = asyncio.Queue()
            self._queues.append(q)
            return q

    return _Bus()


# ---------------------------------------------------------------------------
# Fake / stub harness (used for --demo and until Parrv's orchestrator is ready)
# ---------------------------------------------------------------------------

async def _emit_fake_stream(issue_url: str, bus, delay: float = 0.12) -> None:
    """Drive the bus with the fake event stream for demo / offline use."""
    from tests.mock_llm import fake_event_stream

    events = fake_event_stream(issue_url=issue_url, base_ts=time.time())
    for ev in events:
        bus.emit(ev)
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Trace recorder wiring helper
# ---------------------------------------------------------------------------

def _wire_recorder(bus, trace_path: Path):
    """Subscribe a TraceRecorder to *bus*; returns the recorder."""
    from anvil.trace.recorder import TraceRecorder

    recorder = TraceRecorder(trace_path)
    q = bus.subscribe()

    async def _loop() -> None:
        while True:
            ev = await q.get()
            recorder.record(ev)
            if ev.type == "done":
                recorder.close()
                break

    asyncio.get_event_loop().create_task(_loop())
    return recorder


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def _run_demo(config: dict) -> None:
    """Play the fake event stream in the TUI — no API key required."""
    from anvil.tui.app import AnvilApp

    bus = _make_event_bus()
    _, _, trace_path = _output_paths()

    issue_url = "https://github.com/example/repo/issues/1  [DEMO]"

    def _on_start(url: str) -> None:
        _wire_recorder(bus, trace_path)
        asyncio.get_event_loop().create_task(_emit_fake_stream(url, bus))

    app = AnvilApp(
        bus=bus,
        issue_url=issue_url,
        on_start=_on_start,
        config=config,
    )
    app.run()


def _run_tui(args: argparse.Namespace, config: dict) -> None:
    """Launch the TUI for a live run."""
    from anvil.tui.app import AnvilApp

    issue_url = args.issue or ""
    bus = _make_event_bus()
    _, _, trace_path = _output_paths()

    def _on_start(url: str) -> None:
        _wire_recorder(bus, trace_path)
        # Try Parrv's real orchestrator first; fall back to fake stream
        try:
            from anvil.agent.orchestrator import run_harness

            async def _run() -> None:
                await asyncio.get_event_loop().run_in_executor(
                    None, run_harness, url, config, bus
                )

            asyncio.get_event_loop().create_task(_run())
        except NotImplementedError:
            asyncio.get_event_loop().create_task(_emit_fake_stream(url, bus))

    app = AnvilApp(
        bus=bus,
        issue_url=issue_url,
        on_start=_on_start,
        config=config,
    )
    app.run()


def _run_headless(args: argparse.Namespace, config: dict) -> None:
    """Run the agent pipeline without the TUI and print results."""
    _ensure_api_key()

    issue_url: str = args.issue or ""
    if not issue_url and not args.repo:
        print(
            "ERROR: --issue or (--repo + --issue-text) is required in --headless mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not issue_url and args.repo:
        issue_url = args.repo

    bus = _make_event_bus()
    patch_path, report_path, trace_path = _output_paths()
    q = bus.subscribe()

    from anvil.trace.recorder import TraceRecorder

    recorder = TraceRecorder(trace_path)
    print(f"ANVIL — headless run for: {issue_url}")

    async def _record_all() -> None:
        while True:
            ev = await q.get()
            recorder.record(ev)
            if ev.type == "done":
                recorder.close()
                data = ev.data or {}
                conf = data.get("resolved_confidence", 0)
                try:
                    conf_str = f"{conf:.0%}"
                except (TypeError, ValueError):
                    conf_str = str(conf)
                print(
                    f"\n✓ Done — confidence={conf_str}"
                    f"  steps={data.get('steps', '?')}"
                    f"  tokens={data.get('tokens', '?')}"
                    f"  patch={data.get('patch_path', patch_path)}"
                )
                break

    async def _main() -> None:
        task = asyncio.create_task(_record_all())
        await _emit_fake_stream(issue_url, bus)
        await task

    asyncio.run(_main())


def _run_replay(args: argparse.Namespace) -> None:
    """Replay a saved trace.jsonl through the TUI with speed / pause / step."""
    trace_path: Path = args.trace
    speed: float = max(0.0, float(args.speed))

    if not trace_path.exists():
        print(f"ERROR: trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    from anvil.trace.recorder import TraceRecorder
    from anvil.tui.app import AnvilApp

    events = list(TraceRecorder.load(trace_path))
    if not events:
        print("ERROR: trace file is empty or unreadable.", file=sys.stderr)
        sys.exit(1)

    bus = _make_event_bus()

    app = AnvilApp(
        bus=bus,
        issue_url=f"[REPLAY] {trace_path}",
        replay=True,
        replay_events=events,
        replay_speed=speed,
        config={},
    )
    app.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse args and dispatch to the correct run mode."""
    parser = _build_parser()
    args = parser.parse_args()

    # Ensure Ctrl+C exits cleanly without a traceback
    def _sigint_handler(sig: int, frame: object) -> None:
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Load config (best-effort)
    config: dict = {}
    try:
        from anvil.agent.orchestrator import load_config

        config_path = getattr(args, "config", None)
        config = load_config(config_path)
    except Exception:
        config = {}

    if args.subcommand == "replay":
        _run_replay(args)
    elif getattr(args, "demo", False):
        _run_demo(config)
    elif getattr(args, "headless", False):
        _run_headless(args, config)
    else:
        _run_tui(args, config)


if __name__ == "__main__":
    main()
