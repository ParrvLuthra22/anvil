"""Entry point: ``python -m anvil``.

Modes
-----
TUI (default)::

    python -m anvil
    python -m anvil --issue https://github.com/owner/repo/issues/42

Demo (fake event stream, no API key needed)::

    python -m anvil --demo

Headless (CI/scripting — runs the REAL pipeline)::

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
from urllib.parse import unquote, urlparse


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
    parser.add_argument(
        "--ref",
        metavar="REF",
        help="Git commit or branch to check out (default: repo's default branch)",
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
    out = Path(os.environ.get("ANVIL_OUTPUT_DIR", "output"))
    out.mkdir(parents=True, exist_ok=True)
    return out / "patch.diff", out / "report.md", out / "trace.jsonl"


def _local_repo_git_redirect(repo_url: str) -> tuple[str, dict[str, str]] | None:
    """Map a local fixture path onto the real clone_repo GitHub URL contract."""
    if repo_url.startswith("file://"):
        local_repo = Path(unquote(urlparse(repo_url).path)).resolve()
    else:
        candidate = Path(repo_url)
        if not candidate.exists():
            return None
        local_repo = candidate.resolve()

    synthetic_url = f"https://github.com/anvil-local/{local_repo.name}.git"
    rewrite_index = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    git_env = {
        "GIT_CONFIG_COUNT": str(rewrite_index + 1),
        f"GIT_CONFIG_KEY_{rewrite_index}": f"url.{local_repo.as_uri()}.insteadOf",
        f"GIT_CONFIG_VALUE_{rewrite_index}": synthetic_url,
    }
    return synthetic_url, git_env


# ---------------------------------------------------------------------------
# Shared EventBus — concrete implementation of the abstract base in events.py
#
# Uses threading.Lock for thread-safety so the orchestrator (which runs in a
# thread pool via run_in_executor) can call emit() safely.
# ---------------------------------------------------------------------------

def _make_event_bus():
    """Return a thread-safe concrete EventBus."""
    from anvil.events import EventBus
    return EventBus()


# ---------------------------------------------------------------------------
# Fake event stream — ONLY used by --demo; never imported at real-run time
# ---------------------------------------------------------------------------

async def _emit_fake_stream(issue_url: str, bus, delay: float = 0.12) -> None:
    """Drive the bus with a scripted demo event stream.  Not for production."""
    # Late import: tests/mock_llm only available in dev, not installed package.
    # This function is only called from _run_demo(), never from the real pipeline.
    try:
        from tests.mock_llm import fake_event_stream  # type: ignore[import]
    except ImportError:
        # Installed (non-editable) or missing tests dir — emit a minimal stream
        from anvil.events import AgentEvent, Phase

        def fake_event_stream(issue_url: str = "", base_ts: float = 0.0):  # type: ignore[misc]
            ts = base_ts or time.time()
            events = []
            for i, phase in enumerate(Phase):
                events.append(AgentEvent(ts=ts + i * 0.5, type="phase", phase=phase, data={"name": phase.value}))
            events.append(AgentEvent(
                ts=ts + 10,
                type="done",
                phase=Phase.FINALIZE,
                data={"resolved_confidence": 0.5, "patch_path": "output/patch.diff",
                      "report_path": "output/report.md", "steps": 10, "tokens": 1000, "seconds": 5},
            ))
            return events

    events = fake_event_stream(issue_url=issue_url, base_ts=time.time())
    for ev in events:
        bus.emit(ev)
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Trace recorder wiring helper
# ---------------------------------------------------------------------------

def _wire_recorder(bus, trace_path: Path):
    """Subscribe *bus* and start a recorder task in the CURRENT running loop.

    Must be called from within a coroutine or from code that already has a
    running asyncio event loop (e.g. inside an ``on_start`` callback that the
    Textual app fires, or inside ``asyncio.run()``).  The subscribe() call
    happens here so the queue is registered before the orchestrator starts
    emitting events — ensuring every event is captured and trace.jsonl grows
    during the run, not only after it finishes.
    """
    from anvil.trace.recorder import TraceRecorder

    recorder = TraceRecorder(trace_path)
    # subscribe() must be called inside the running loop so that put_nowait()
    # from any thread delivers to the correct queue.
    q = bus.subscribe()

    async def _drain() -> None:
        try:
            while True:
                ev = await q.get()
                recorder.record(ev)  # flushes immediately — trace grows live
                if ev.type == "done":
                    break
        finally:
            recorder.close()  # always close, even if we exit via cancellation

    asyncio.get_event_loop().create_task(_drain())
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

    def _on_start(
        url: str,
        ref: str | None = None,
        manual_issue_text: str | None = None,
    ) -> None:
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
    """Launch the TUI for a live run using the real orchestrator."""
    from anvil.tui.app import AnvilApp

    issue_url = args.issue or ""
    bus = _make_event_bus()
    _, _, trace_path = _output_paths()

    def _on_start(url: str, ref: str | None = None, manual_issue_text: str | None = None) -> None:
        _wire_recorder(bus, trace_path)
        try:
            from anvil.agent.orchestrator import run_harness

            repo_url = getattr(args, "repo", None)
            issue_text = manual_issue_text or getattr(args, "issue_text", None)
            git_ref = ref or getattr(args, "ref", None)

            async def _run() -> None:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: run_harness(
                        url,
                        config,
                        bus,
                        repo_url=repo_url,
                        issue_text=issue_text,
                        git_ref=git_ref,
                    ),
                )

            asyncio.get_event_loop().create_task(_run())
        except Exception as e:
            print(f"Error starting harness: {e}", file=sys.stderr)

    app = AnvilApp(
        bus=bus,
        issue_url=issue_url,
        on_start=_on_start,
        config=config,
    )
    app.run()


def _run_headless(args: argparse.Namespace, config: dict) -> None:
    """Run the REAL agent pipeline without the TUI and print results.

    Calls ``run_harness(issue_url, config, bus, repo_url=..., issue_text=...)``
    from the orchestrator.  Falls back to a clear error if the orchestrator
    raises NotImplementedError (stub not yet replaced).

    Writes output/patch.diff, output/report.md, and output/trace.jsonl.
    Works from any working directory.
    """
    _ensure_api_key()

    issue_url: str = args.issue or ""
    repo_url: str = getattr(args, "repo", None) or ""
    issue_text: str = getattr(args, "issue_text", None) or ""

    if not issue_url and not repo_url:
        print(
            "ERROR: --issue or (--repo + --issue-text) is required in --headless mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Local fixtures use Git's normal URL rewrite mechanism, so the real
    # RepoPipeline and clone_repo hooks still run without replacing internals.
    display_url = issue_url or repo_url
    git_env: dict[str, str] = {}
    if repo_url:
        redirect = _local_repo_git_redirect(repo_url)
        if redirect is not None:
            repo_url, git_env = redirect

    bus = _make_event_bus()
    patch_path, report_path, trace_path = _output_paths()
    print(f"ANVIL — headless run for: {display_url}")

    async def _record_all(q: asyncio.Queue) -> None:
        """Drain the bus queue, write each event immediately so trace grows live."""
        from anvil.trace.recorder import TraceRecorder

        recorder = TraceRecorder(trace_path)
        try:
            while True:
                ev = await q.get()
                recorder.record(ev)  # flush-on-write — trace.jsonl grows during run
                if ev.type == "done":
                    data = ev.data or {}
                    conf = data.get("resolved_confidence", 0)
                    try:
                        conf_str = f"{conf:.0%}"
                    except (TypeError, ValueError):
                        conf_str = str(conf)
                    actual_patch = data.get("patch_path", str(patch_path))
                    actual_report = data.get("report_path", str(report_path))
                    print(
                        f"\n✓ Done"
                        f"  confidence={conf_str}"
                        f"  steps={data.get('steps', '?')}"
                        f"  tokens={data.get('tokens', '?')}"
                        f"  time={data.get('seconds', '?')}s"
                        f"\n  patch  → {actual_patch}"
                        f"\n  report → {actual_report}"
                        f"\n  trace  → {trace_path}"
                    )
                    break
        finally:
            recorder.close()  # always close — guarantees trace even on timeout/cancel

    async def _main() -> None:
        # subscribe() inside asyncio.run() so the queue belongs to this loop
        q = bus.subscribe()
        record_task = asyncio.create_task(_record_all(q))
        try:
            from anvil.agent.orchestrator import run_harness

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: run_harness(
                    issue_url or repo_url,
                    config,
                    bus,
                    repo_url=repo_url or None,
                    issue_text=issue_text or None,
                    git_ref=getattr(args, "ref", None),
                ),
            )
        except Exception as e:
            print(f"Error starting harness: {e}", file=sys.stderr)
            sys.exit(1)
        await record_task

    previous_env = {key: os.environ.get(key) for key in git_env}
    os.environ.update(git_env)
    try:
        asyncio.run(_main())
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


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

    # Load config (best-effort; falls back to empty dict so the app still starts)
    config: dict = {}
    try:
        from anvil.agent.orchestrator import load_config

        config_path = getattr(args, "config", None)
        config = load_config(config_path)
    except Exception:
        config = {}

    if "ANVIL_OUTPUT_DIR" in os.environ:
        config["output_dir"] = os.environ["ANVIL_OUTPUT_DIR"]

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
