"""Entry point: ``python -m anvil``.

Modes
-----
TUI (default)::

    python -m anvil
    python -m anvil --issue https://github.com/owner/repo/issues/42

Headless (no TUI, prints result)::

    python -m anvil --issue <url> --headless
    python -m anvil --repo <repo-url> --issue-text "description" --headless

Replay a saved trace in the TUI::

    python -m anvil replay output/trace.jsonl [--speed 4]
"""

from __future__ import annotations

import argparse
import asyncio
import os
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
    )

    # ── Sub-commands ─────────────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="subcommand")

    # replay sub-command
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
        choices=[1.0, 4.0],
        metavar="SPEED",
        help="Replay speed multiplier: 1 (real-time) | 4 (fast) | 0 (instant). Default: 1",
    )

    # ── Top-level flags (main run mode) ──────────────────────────────────────
    parser.add_argument(
        "--issue",
        metavar="URL",
        help="GitHub issue URL to resolve (e.g. https://github.com/owner/repo/issues/42)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the TUI; print the result to stdout",
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
        help="Path to config.yaml (default: repo root config.yaml)",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_api_key() -> str:
    """Return the API key or exit with a clear message."""
    key = os.environ.get("AI_API_KEY", "").strip()
    if not key:
        print(
            "\nERROR: AI_API_KEY is not set.\n"
            "  export AI_API_KEY=<your-api-key>  then re-run.\n",
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
# Real-time event bus implementation
# ---------------------------------------------------------------------------

def _make_event_bus():
    """Return a concrete EventBus that fans out to all subscriber queues."""
    from anvil.events import AgentEvent, EventBus

    class _Bus(EventBus):
        def __init__(self) -> None:
            self._queues: list[asyncio.Queue] = []

        def emit(self, event: AgentEvent) -> None:
            for q in self._queues:
                q.put_nowait(event)

        def subscribe(self) -> asyncio.Queue:
            q: asyncio.Queue = asyncio.Queue()
            self._queues.append(q)
            return q

    return _Bus()


# ---------------------------------------------------------------------------
# Stub harness (used until Parrv's orchestrator is ready)
# ---------------------------------------------------------------------------

async def _stub_harness(issue_url: str, bus) -> None:
    """Drive the bus with the fake event stream so the TUI works standalone."""
    from tests.mock_llm import fake_event_stream

    events = fake_event_stream(issue_url=issue_url, base_ts=time.time())
    for ev in events:
        bus.emit(ev)
        await asyncio.sleep(0.05)  # slight delay so TUI can render


def _run_harness_sync(issue_url: str, config: dict, bus) -> None:
    """Call Parrv's run_harness if it exists; fall back to the stub."""
    try:
        from anvil.agent.orchestrator import run_harness
        run_harness(issue_url, config, bus)
    except NotImplementedError:
        # Orchestrator not yet implemented — use fake stream for demo.
        asyncio.get_event_loop().run_until_complete(_stub_harness(issue_url, bus))


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def _run_tui(args: argparse.Namespace, config: dict) -> None:
    """Launch the Textual TUI for a live run or demo."""
    from anvil.tui.app import AnvilApp

    issue_url = args.issue or ""
    bus = _make_event_bus()
    patch_path, report_path, trace_path = _output_paths()

    from anvil.trace.recorder import TraceRecorder

    recorder = TraceRecorder(trace_path)

    def _on_start(url: str) -> None:
        """Called by the TUI when the user submits an issue URL."""
        nonlocal issue_url
        issue_url = url
        # Subscribe recorder to bus
        q = bus.subscribe()

        async def _record_loop():
            while True:
                ev = await q.get()
                recorder.record(ev)
                if ev.type == "done":
                    recorder.close()
                    break

        asyncio.get_event_loop().create_task(_record_loop())
        asyncio.get_event_loop().create_task(_stub_harness(issue_url, bus))

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
        print("ERROR: --issue or (--repo + --issue-text) is required in --headless mode.", file=sys.stderr)
        sys.exit(1)

    if not issue_url and args.repo:
        # Synthesise a pseudo-URL for the bus; real ingestion uses --repo + --issue-text
        issue_url = args.repo

    bus = _make_event_bus()
    patch_path, report_path, trace_path = _output_paths()

    from anvil.trace.recorder import TraceRecorder

    recorder = TraceRecorder(trace_path)
    q = bus.subscribe()

    print(f"ANVIL — running headless for: {issue_url}")

    # Record all events
    async def _record_all():
        while True:
            ev = await q.get()
            recorder.record(ev)
            if ev.type == "done":
                recorder.close()
                data = ev.data
                print(
                    f"\n✓ Done — confidence={data.get('resolved_confidence', '?'):.0%}  "
                    f"steps={data.get('steps', '?')}  tokens={data.get('tokens', '?')}  "
                    f"patch={data.get('patch_path', patch_path)}"
                )
                break

    async def _main():
        task = asyncio.create_task(_record_all())
        await _stub_harness(issue_url, bus)
        await task

    asyncio.run(_main())


def _run_replay(args: argparse.Namespace) -> None:
    """Replay a saved trace.jsonl through the TUI."""
    trace_path: Path = args.trace
    speed: float = args.speed

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

    async def _replay_loop():
        """Emit events with timing controlled by speed multiplier."""
        prev_ts = events[0].ts
        for ev in events:
            if speed > 0:
                delay = (ev.ts - prev_ts) / speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 2.0))  # cap individual gap at 2 s
            prev_ts = ev.ts
            bus.emit(ev)

    def _on_start(_url: str) -> None:
        asyncio.get_event_loop().create_task(_replay_loop())

    app = AnvilApp(
        bus=bus,
        issue_url="[REPLAY] " + str(trace_path),
        on_start=_on_start,
        replay=True,
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

    # Load config (best-effort; headless / TUI modes both need it)
    try:
        from anvil.agent.orchestrator import load_config
        config = load_config(args.config if hasattr(args, "config") else None)
    except Exception:
        config = {}

    if args.subcommand == "replay":
        _run_replay(args)
    elif args.headless:
        _run_headless(args, config)
    else:
        _run_tui(args, config)


if __name__ == "__main__":
    main()
