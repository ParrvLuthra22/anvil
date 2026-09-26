"""ANVIL Textual TUI — main application.

Layout (min 80×24)
------------------
┌────────────────────────────────────────────────────────────────────────────┐
│  Header: ANVIL | Issue URL input | [Start]                                 │
├──────────────┬───────────────────────────────────────┬─────────────────────┤
│  Phase       │  Scrolling event log                  │  Metrics panel      │
│  stepper     │  messages / tool calls / results      │  steps/tokens/cost  │
│  (9 phases)  │  colour-coded, never crashes          │  elapsed / budget   │
├──────────────┴───────────────────────────────────────┴─────────────────────┤
│  ERROR banner (red, visible on error events; run continues)                │
├─────────────────────────────────────────────────────────────────────────────│
│  Diff preview pane  (toggle: d)                                            │
│  RESULT panel (appears on done event)                                      │
└────────────────────────────────────────────────────────────────────────────┘

Keyboard:
  q         quit (Ctrl+C also exits cleanly)
  r         restart (clear log + reset phases)
  d         toggle diff pane
  space     pause / resume replay
  → / l     step forward  (replay step-by-step)
  ← / h     step backward (replay step-by-step, re-emits from buffer)
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)

from anvil.events import AgentEvent, EventBus, Phase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PHASE_ORDER: list[Phase] = [
    Phase.INGEST,
    Phase.PROFILE,
    Phase.UNDERSTAND,
    Phase.LOCALIZE,
    Phase.REPRODUCE,
    Phase.PATCH,
    Phase.VERIFY,
    Phase.REVIEW,
    Phase.FINALIZE,
]

_PHASE_LABEL: dict[Phase, str] = {
    Phase.INGEST:     "① INGEST",
    Phase.PROFILE:    "② PROFILE",
    Phase.UNDERSTAND: "③ UNDERSTAND",
    Phase.LOCALIZE:   "④ LOCALIZE",
    Phase.REPRODUCE:  "⑤ REPRODUCE",
    Phase.PATCH:      "⑥ PATCH",
    Phase.VERIFY:     "⑦ VERIFY",
    Phase.REVIEW:     "⑧ REVIEW",
    Phase.FINALIZE:   "⑨ FINALIZE",
}

_PHASE_CSS: dict[str, str] = {
    "pending": "phase-pending",
    "active":  "phase-active",
    "done":    "phase-done",
    "failed":  "phase-failed",
}

# Max chars per log line — prevents the RichLog from choking on huge tool output
_MAX_LINE = 400

# ---------------------------------------------------------------------------
# Error severity  (source: src/anvil/agent/NOTES.md)
# ---------------------------------------------------------------------------
# Benign/recovery kinds: the run continues; show as a yellow notice in the log.
# Fatal kinds (llm, budget, internal): also raise the sticky red ErrorBanner.
#
# Benign kinds emitted by the agent:
#   loop         — same tool+args 3 times in a row, or A/B/A/B; phase advances
#   edit         — edit_file failed; model gets closest lines hint
#   invalid_call — unknown tool, wrong phase, bad JSON args, missing arg
#   no_tool_call — plain reply in a tool-expecting phase; nudge or stall
#   test_failure — run_tests failed (non-timeout); failure lines shown to model
#   timeout      — any tool result reporting a timeout
#   tool         — tool other than edit/run_tests/run_cmd returned ok=False
#   deps         — dependency install warning; run continues
#   sandbox      — checkpoint/rollback/diff/close problem; run continues
#   context      — context summariser failed; run continues
#   rollback     — sandbox rolled back to checkpoint; model retries
#   config       — configuration problem; run continues
#   io           — file I/O warning; run continues
#   finalize     — problem during FINALIZE; outputs still written
#   <phase name> — unexpected exception in a phase (e.g. "ingest", "patch")
_BENIGN_ERROR_KINDS: frozenset[str] = frozenset({
    "loop", "edit", "invalid_call", "no_tool_call", "test_failure",
    "timeout", "tool", "deps", "sandbox", "context", "rollback",
    "config", "io", "finalize",
    # Phase names used as kind when an unexpected exception occurs mid-phase:
    "ingest", "profile", "understand", "localize", "reproduce",
    "patch", "verify", "review",
})


def _clip(text: str, limit: int = _MAX_LINE) -> str:
    """Truncate *text* to *limit* chars with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[:limit] + " [dim]…[/dim]"


def _is_fatal(kind: str) -> bool:
    """Return True if *kind* should trigger the red ErrorBanner.

    Fatal kinds: ``llm``, ``budget``, ``internal``.
    Everything else is benign and shown only as a yellow notice.
    """
    return kind.lower() not in _BENIGN_ERROR_KINDS

# ---------------------------------------------------------------------------
# Phase stepper widget
# ---------------------------------------------------------------------------

class PhaseStepper(Widget):
    """Left column: shows each phase with pending / active / done / failed state."""

    DEFAULT_CSS = """
    PhaseStepper {
        width: 20;
        min-width: 18;
        height: 100%;
        border-right: solid $primary-darken-2;
        padding: 1;
        background: $surface;
    }
    .phase-pending  { color: $text-muted; }
    .phase-active   { color: $warning; text-style: bold; }
    .phase-done     { color: $success; }
    .phase-failed   { color: $error; text-style: bold; }
    #phase-header   { color: $accent; text-style: bold; padding-bottom: 1; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._states: dict[Phase, str] = {p: "pending" for p in _PHASE_ORDER}
        self._labels: dict[Phase, Label] = {}

    def compose(self) -> ComposeResult:
        yield Label("PHASES", id="phase-header")
        yield Static("─" * 16)
        for phase in _PHASE_ORDER:
            lbl = Label(_PHASE_LABEL[phase], classes="phase-pending", id=f"phase-{phase.value}")
            self._labels[phase] = lbl
            yield lbl

    def set_phase_state(self, phase: Phase, state: str) -> None:
        """Update display state: pending | active | done | failed."""
        self._states[phase] = state
        lbl = self._labels.get(phase)
        if lbl is None:
            return
        for css_class in _PHASE_CSS.values():
            lbl.remove_class(css_class)
        lbl.add_class(_PHASE_CSS.get(state, "phase-pending"))
        prefix = {"pending": "  ", "active": "▶ ", "done": "✓ ", "failed": "✗ "}.get(state, "  ")
        lbl.update(prefix + _PHASE_LABEL[phase])

    def mark_active(self, phase: Phase) -> None:
        for p in _PHASE_ORDER:
            if p == phase:
                break
            if self._states[p] == "pending":
                self.set_phase_state(p, "done")
        self.set_phase_state(phase, "active")

    def mark_done(self, phase: Phase) -> None:
        self.set_phase_state(phase, "done")

    def mark_failed(self, phase: Phase) -> None:
        self.set_phase_state(phase, "failed")

    def mark_all_done(self) -> None:
        for p in _PHASE_ORDER:
            if self._states[p] in ("pending", "active"):
                self.set_phase_state(p, "done")

    def reset(self) -> None:
        for p in _PHASE_ORDER:
            self.set_phase_state(p, "pending")


# ---------------------------------------------------------------------------
# Metrics panel widget
# ---------------------------------------------------------------------------

class MetricsPanel(Widget):
    """Right column: live step / token / cost / time / budget metrics."""

    DEFAULT_CSS = """
    MetricsPanel {
        width: 22;
        min-width: 20;
        height: 100%;
        border-left: solid $primary-darken-2;
        padding: 1;
        background: $surface;
    }
    .metric-label { color: $text-muted; }
    .metric-value { color: $accent; text-style: bold; }
    #metrics-header { color: $accent; text-style: bold; padding-bottom: 1; }
    """

    def __init__(self, config: dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._steps = 0
        self._prompt_tok = 0
        self._completion_tok = 0
        self._cost = 0.0
        self._start_ts: float = time.time()
        self._max_steps: int = config.get("max_total_steps", 120)
        self._max_tokens: int = config.get("max_tokens_total", 1_500_000)

    def compose(self) -> ComposeResult:
        yield Label("METRICS", id="metrics-header")
        yield Static("─" * 18)
        yield Label("Steps",          classes="metric-label")
        yield Label("0",              id="m-steps",      classes="metric-value")
        yield Label("Prompt tokens",  classes="metric-label")
        yield Label("0",              id="m-prompt",     classes="metric-value")
        yield Label("Completion tok", classes="metric-label")
        yield Label("0",              id="m-completion", classes="metric-value")
        yield Label("Total tokens",   classes="metric-label")
        yield Label("0",              id="m-total-tok",  classes="metric-value")
        yield Label("Est. cost ($)",  classes="metric-label")
        yield Label("0.000000",       id="m-cost",       classes="metric-value")
        yield Label("Elapsed",        classes="metric-label")
        yield Label("0s",             id="m-elapsed",    classes="metric-value")
        yield Static("─" * 18)
        yield Label("Budget left",    classes="metric-label")
        yield Label("",               id="m-budget",     classes="metric-value")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        elapsed = int(time.time() - self._start_ts)
        try:
            self.query_one("#m-elapsed", Label).update(f"{elapsed}s")
            tok_used = self._prompt_tok + self._completion_tok
            tok_remain = max(0, self._max_tokens - tok_used)
            step_remain = max(0, self._max_steps - self._steps)
            self.query_one("#m-budget", Label).update(
                f"steps:{step_remain}\ntok:{tok_remain:,}"
            )
        except Exception:
            pass

    def update_usage(self, prompt: int, completion: int, cost: float) -> None:
        """Accumulate one LLM call's usage."""
        self._prompt_tok += prompt
        self._completion_tok += completion
        self._cost += cost
        self._steps += 1
        total = self._prompt_tok + self._completion_tok
        try:
            self.query_one("#m-steps",      Label).update(str(self._steps))
            self.query_one("#m-prompt",     Label).update(f"{self._prompt_tok:,}")
            self.query_one("#m-completion", Label).update(f"{self._completion_tok:,}")
            self.query_one("#m-total-tok",  Label).update(f"{total:,}")
            self.query_one("#m-cost",       Label).update(f"{self._cost:.6f}")
        except Exception:
            pass

    def reset(self) -> None:
        """Reset all counters (used by restart)."""
        self._steps = 0
        self._prompt_tok = 0
        self._completion_tok = 0
        self._cost = 0.0
        self._start_ts = time.time()


# ---------------------------------------------------------------------------
# Error banner widget (visible when an error event arrives)
# ---------------------------------------------------------------------------

class ErrorBanner(Widget):
    """Sticky red banner — appears on error events, run continues."""

    DEFAULT_CSS = """
    ErrorBanner {
        height: 2;
        padding: 0 2;
        background: $error-darken-2;
        display: none;
    }
    ErrorBanner.visible { display: block; }
    #error-text { color: $text; }
    """

    def compose(self) -> ComposeResult:
        yield Label("", id="error-text")

    def show_error(self, kind: str, message: str) -> None:
        """Display *message*; makes the banner visible."""
        text = f"[bold red]⚠ {kind.upper()}[/bold red]  {_clip(message, 200)}"
        try:
            self.query_one("#error-text", Label).update(text)
            self.add_class("visible")
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self.query_one("#error-text", Label).update("")
            self.remove_class("visible")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Diff pane widget
# ---------------------------------------------------------------------------

class DiffPane(Widget):
    """Bottom pane: shows the current unified diff, syntax-coloured."""

    DEFAULT_CSS = """
    DiffPane {
        height: 10;
        border-top: solid $primary-darken-2;
        padding: 0 1;
        background: $surface-darken-1;
        display: block;
    }
    DiffPane.hidden { display: none; }
    #diff-header { color: $text-muted; }
    """

    def compose(self) -> ComposeResult:
        yield Label("── DIFF PREVIEW ──────────────────────────────", id="diff-header")
        yield RichLog(id="diff-log", highlight=True, markup=True, wrap=False)

    def update_diff(self, diff_text: str) -> None:
        """Replace the diff log content with syntax-coloured diff."""
        try:
            log = self.query_one("#diff-log", RichLog)
            log.clear()
            for line in diff_text.splitlines()[:200]:  # cap at 200 lines
                line = line.rstrip()
                if line.startswith("+") and not line.startswith("+++"):
                    log.write(f"[green]{_clip(line)}[/green]")
                elif line.startswith("-") and not line.startswith("---"):
                    log.write(f"[red]{_clip(line)}[/red]")
                elif line.startswith("@@"):
                    log.write(f"[cyan]{_clip(line)}[/cyan]")
                else:
                    log.write(_clip(line))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Result banner widget
# ---------------------------------------------------------------------------

class ResultBanner(Widget):
    """Shows patch path, confidence, steps, tokens and time when done."""

    DEFAULT_CSS = """
    ResultBanner {
        height: 5;
        border: double $success;
        padding: 0 2;
        background: $success-darken-3;
        display: none;
    }
    ResultBanner.visible { display: block; }
    """

    def compose(self) -> ComposeResult:
        yield Label("", id="result-line1")
        yield Label("", id="result-line2")

    def show(self, data: dict) -> None:
        """Populate and reveal the result banner from *data*."""
        try:
            conf   = data.get("resolved_confidence", 0)
            patch  = data.get("patch_path",  "output/patch.diff")
            report = data.get("report_path", "output/report.md")
            steps  = data.get("steps",   "?")
            tokens = data.get("tokens",  0)
            secs   = data.get("seconds", "?")

            line1 = (
                f"[bold green]✓ DONE[/bold green]  "
                f"confidence=[bold]{conf:.0%}[/bold]  "
                f"steps={steps}  tokens={tokens:,}  time={secs}s"
            )
            line2 = (
                f"patch  → {patch}\n"
                f"report → {report}"
            )
            self.query_one("#result-line1", Label).update(line1)
            self.query_one("#result-line2", Label).update(line2)
            self.add_class("visible")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Replay status bar
# ---------------------------------------------------------------------------

class ReplayBar(Widget):
    """Thin bar showing replay speed and pause/step status."""

    DEFAULT_CSS = """
    ReplayBar {
        height: 1;
        background: $primary-darken-3;
        padding: 0 2;
        display: none;
    }
    ReplayBar.visible { display: block; }
    #replay-status { color: $accent; }
    """

    def compose(self) -> ComposeResult:
        yield Label("", id="replay-status")

    def update_status(self, speed: float, paused: bool, idx: int, total: int) -> None:
        speed_str = "instant" if speed == 0 else f"{speed}×"
        state = "PAUSED" if paused else f"playing {speed_str}"
        try:
            self.query_one("#replay-status", Label).update(
                f"[bold]REPLAY[/bold]  {state}  event {idx}/{total}  "
                "│ space=pause  →/←=step  1/4/0=speed"
            )
        except Exception:
            pass

    def show(self) -> None:
        self.add_class("visible")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AnvilApp(App):
    """ANVIL Textual application.

    Consumes :class:`~anvil.events.AgentEvent` objects from *bus* and renders
    them. Works identically for live runs, ``--demo``, and trace replays.
    Never crashes on an unknown event type or a missing data field.
    """

    TITLE = "ANVIL — Autonomous Coding Agent"
    CSS = """
    Screen { background: $background; }

    /* ── Top input bar ── */
    #top-bar {
        height: 3;
        background: $primary-darken-3;
        padding: 0 1;
        border-bottom: solid $primary-darken-1;
    }
    #app-title {
        width: 26;
        color: $accent;
        text-style: bold;
        padding: 0 1;
        content-align: left middle;
    }
    #issue-input {
        width: 1fr;
        border: none;
    }
    #start-btn { width: 10; margin: 0 1; }

    /* ── Body ── */
    #body { height: 1fr; }

    /* ── Event log ── */
    #log-area {
        width: 1fr;
        height: 100%;
        border-right: solid $primary-darken-2;
    }
    #event-log {
        width: 100%;
        height: 100%;
        scrollbar-color: $primary;
    }
    """

    BINDINGS = [
        Binding("q",          "quit",         "Quit",          priority=True),
        Binding("ctrl+c",     "quit",         "Quit",          show=False, priority=True),
        Binding("r",          "restart",      "Restart"),
        Binding("d",          "toggle_diff",  "Toggle diff"),
        Binding("space",      "toggle_pause", "Pause/Resume",  show=False),
        Binding("right",      "step_forward", "Step →",        show=False),
        Binding("l",          "step_forward", "Step →",        show=False),
        Binding("left",       "step_back",    "Step ←",        show=False),
        Binding("h",          "step_back",    "Step ←",        show=False),
        Binding("1",          "speed_1",      "Speed 1×",      show=False),
        Binding("4",          "speed_4",      "Speed 4×",      show=False),
        Binding("0",          "speed_instant","Speed instant",  show=False),
    ]

    def __init__(
        self,
        bus: EventBus,
        issue_url: str = "",
        on_start: Callable[[str], None] | None = None,
        replay: bool = False,
        replay_events: list[AgentEvent] | None = None,
        replay_speed: float = 1.0,
        config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bus = bus
        self._issue_url = issue_url
        self._on_start = on_start
        self._replay = replay
        self._replay_events: list[AgentEvent] = replay_events or []
        self._replay_speed: float = replay_speed
        self._config = config or {}

        # Runtime state
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self._started = False
        self._diff_visible = True
        self._current_phase: Phase | None = None

        # Replay state
        self._replay_paused = False
        self._replay_idx = 0          # index of next event to emit
        self._replay_task: asyncio.Task | None = None
        self._replay_history: list[AgentEvent] = []  # all events emitted so far

    # ── Layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Top input bar
        with Horizontal(id="top-bar"):
            yield Label("⚒  ANVIL", id="app-title")
            yield Input(
                value=self._issue_url,
                placeholder="GitHub issue URL…",
                id="issue-input",
            )
            yield Button("▶  Start", id="start-btn", variant="primary")

        # Replay status bar (hidden unless in replay mode)
        yield ReplayBar(id="replay-bar")

        # Body: phase stepper | event log | metrics
        with Horizontal(id="body"):
            yield PhaseStepper(id="phase-stepper")
            with ScrollableContainer(id="log-area"):
                yield RichLog(id="event-log", highlight=True, markup=True, wrap=True)
            yield MetricsPanel(config=self._config, id="metrics-panel")

        # Error banner (appears on error events)
        yield ErrorBanner(id="error-banner")

        # Diff preview pane
        yield DiffPane(id="diff-pane")

        # Result banner (appears on done event)
        yield ResultBanner(id="result-banner")

        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and optionally auto-start."""
        self._queue = self._bus.subscribe()
        self.set_interval(0.05, self._drain_queue)

        if self._replay and self._replay_events:
            replay_bar = self.query_one("#replay-bar", ReplayBar)
            replay_bar.show()
            self._do_start(self._issue_url)
        elif self._issue_url:
            self._do_start(self._issue_url)

    # ── Queue drainer ────────────────────────────────────────────────────────

    def _drain_queue(self) -> None:
        """Poll the event queue and process all available events (non-blocking)."""
        if self._queue is None:
            return
        try:
            while True:
                ev = self._queue.get_nowait()
                self._handle_event(ev)
        except asyncio.QueueEmpty:
            pass
        except Exception:
            pass

    # ── Event → UI dispatch ──────────────────────────────────────────────────

    def _handle_event(self, ev: AgentEvent) -> None:
        """Dispatch an AgentEvent to the appropriate UI widgets.

        Wraps every operation in try/except so a bad event never crashes the TUI.
        Missing fields are defaulted, unknown event types are silently skipped.
        """
        try:
            ev_type = str(ev.type) if ev.type is not None else "unknown"
            data = ev.data if isinstance(ev.data, dict) else {}

            log: RichLog         = self.query_one("#event-log",    RichLog)
            stepper: PhaseStepper = self.query_one("#phase-stepper", PhaseStepper)
            metrics: MetricsPanel = self.query_one("#metrics-panel", MetricsPanel)

            if ev_type == "phase":
                self._handle_phase(ev.phase, data, stepper, log)

            elif ev_type == "message":
                role  = str(data.get("role", "?"))
                text  = str(data.get("text", ""))
                colour = "yellow" if role == "assistant" else "white"
                icon   = "🤖" if role == "assistant" else "👤"
                log.write(f"[{colour}]{icon} [{role}][/{colour}] {_clip(text)}")

            elif ev_type == "tool_call":
                tool = str(data.get("tool", "?"))
                args = data.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                args_str = ", ".join(
                    f"{k}={_clip(repr(v), 60)}" for k, v in list(args.items())[:8]
                )
                log.write(f"[bold blue]⚙ CALL[/bold blue] [cyan]{tool}[/cyan]({args_str})")

            elif ev_type == "tool_result":
                tool    = str(data.get("tool", "?"))
                ok      = bool(data.get("ok", True))
                preview = str(data.get("output_preview", ""))
                icon    = "[green]✓[/green]" if ok else "[red]✗[/red]"
                log.write(f"  {icon} [dim]{tool}[/dim]: {_clip(preview)}")
                if tool == "git_diff" and ok and preview:
                    self.query_one("#diff-pane", DiffPane).update_diff(preview)

            elif ev_type == "llm_usage":
                prompt = int(data.get("prompt_tokens", 0))
                comp   = int(data.get("completion_tokens", 0))
                cost   = float(data.get("cost_estimate", 0.0))
                metrics.update_usage(prompt, comp, cost)
                log.write(f"  [dim]tokens: +{prompt}p +{comp}c  Δcost=${cost:.6f}[/dim]")

            elif ev_type == "error":
                kind = str(data.get("kind", "error"))
                msg  = str(data.get("message", ""))
                if _is_fatal(kind):
                    # Fatal: red log entry + sticky red banner
                    log.write(f"[bold red]⚠ ERROR[/bold red] [{kind}] {_clip(msg)}")
                    self.query_one("#error-banner", ErrorBanner).show_error(kind, msg)
                else:
                    # Benign/recovery: yellow notice in log only, no banner
                    log.write(f"[yellow]⚠ notice[/yellow] [{kind}] {_clip(msg)}")

            elif ev_type == "done":
                stepper.mark_all_done()
                self.query_one("#result-banner", ResultBanner).show(data)
                # Load patch into diff pane if it exists on disk
                patch_path = str(data.get("patch_path", ""))
                if patch_path and os.path.exists(patch_path):
                    try:
                        with open(patch_path) as fh:
                            self.query_one("#diff-pane", DiffPane).update_diff(fh.read())
                    except Exception:
                        pass

            # Unknown types: silently skipped (future-proof).

        except Exception:
            # The TUI must never crash on a bad event under any circumstances.
            pass

    def _handle_phase(
        self,
        phase: Any,
        data: dict,
        stepper: PhaseStepper,
        log: RichLog,
    ) -> None:
        """Handle a phase transition event."""
        try:
            if not isinstance(phase, Phase):
                return
            if self._current_phase and self._current_phase != phase:
                stepper.mark_done(self._current_phase)
            self._current_phase = phase
            stepper.mark_active(phase)
            log.write(f"\n[bold cyan]━━ {phase.value.upper()} ━━[/bold cyan]")
            # Clear the error banner when a new phase starts (run recovered)
            self.query_one("#error-banner", ErrorBanner).clear()
        except Exception:
            pass

    # ── Button / Input handlers ───────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            url = self.query_one("#issue-input", Input).value.strip()
            if url:
                self._do_start(url)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "issue-input":
            url = event.value.strip()
            if url:
                self._do_start(url)

    def _do_start(self, url: str) -> None:
        """Trigger the harness for *url* (idempotent)."""
        if self._started:
            return
        self._started = True
        try:
            self.query_one("#issue-input", Input).value = url
        except Exception:
            pass
        try:
            log: RichLog = self.query_one("#event-log", RichLog)
            log.write(f"[bold green]▶ Starting:[/bold green] {_clip(url, 200)}\n")
        except Exception:
            pass

        if self._replay and self._replay_events:
            self._start_replay_task()
        elif self._on_start:
            self._on_start(url)

    # ── Replay engine ─────────────────────────────────────────────────────────

    def _start_replay_task(self) -> None:
        """Launch the async replay emitter."""
        if self._replay_task and not self._replay_task.done():
            self._replay_task.cancel()
        self._replay_task = asyncio.get_event_loop().create_task(self._replay_loop())

    async def _replay_loop(self) -> None:
        """Emit replay events respecting speed, pause and step controls."""
        events = self._replay_events
        total  = len(events)
        if not events:
            return

        prev_ts = events[0].ts
        self._replay_idx = 0

        while self._replay_idx < total:
            # Update status bar
            try:
                self.query_one("#replay-bar", ReplayBar).update_status(
                    self._replay_speed,
                    self._replay_paused,
                    self._replay_idx,
                    total,
                )
            except Exception:
                pass

            if self._replay_paused:
                await asyncio.sleep(0.1)
                continue

            ev = events[self._replay_idx]

            # Timing delay
            if self._replay_speed > 0:
                delay = (ev.ts - prev_ts) / self._replay_speed
                if delay > 0:
                    elapsed = 0.0
                    step_size = 0.05
                    while elapsed < delay:
                        if self._replay_paused:
                            break
                        await asyncio.sleep(step_size)
                        elapsed += step_size
            prev_ts = ev.ts

            self._bus.emit(ev)
            self._replay_history.append(ev)
            self._replay_idx += 1

        # Final status update
        try:
            self.query_one("#replay-bar", ReplayBar).update_status(
                self._replay_speed, False, total, total
            )
        except Exception:
            pass

    # ── Key-binding actions ───────────────────────────────────────────────────

    def action_quit(self) -> None:
        """Exit cleanly — output/ files are preserved."""
        self.exit()

    def action_restart(self) -> None:
        """Reset the UI to its initial state."""
        self._started = False
        self._current_phase = None
        try:
            self.query_one("#event-log",    RichLog).clear()
            self.query_one("#phase-stepper", PhaseStepper).reset()
            self.query_one("#metrics-panel", MetricsPanel).reset()
            self.query_one("#error-banner",  ErrorBanner).clear()
            self.query_one("#result-banner", ResultBanner).remove_class("visible")
        except Exception:
            pass

    def action_toggle_diff(self) -> None:
        self._diff_visible = not self._diff_visible
        try:
            diff_pane = self.query_one("#diff-pane", DiffPane)
            if self._diff_visible:
                diff_pane.remove_class("hidden")
            else:
                diff_pane.add_class("hidden")
        except Exception:
            pass

    # ── Replay control actions ────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        if not self._replay:
            return
        self._replay_paused = not self._replay_paused

    def action_step_forward(self) -> None:
        """Emit the next event immediately (step-by-step mode)."""
        if not self._replay:
            return
        events = self._replay_events
        if self._replay_idx < len(events):
            self._replay_paused = True
            ev = events[self._replay_idx]
            self._bus.emit(ev)
            self._replay_history.append(ev)
            self._replay_idx += 1
            try:
                self.query_one("#replay-bar", ReplayBar).update_status(
                    self._replay_speed, True, self._replay_idx, len(events)
                )
            except Exception:
                pass

    def action_step_back(self) -> None:
        """Re-emit all events up to idx-1 (re-renders from history)."""
        if not self._replay:
            return
        if self._replay_idx <= 1:
            return
        self._replay_paused = True
        self._replay_idx -= 1
        # Re-render: restart the UI then replay up to idx
        self.action_restart()
        for ev in self._replay_events[: self._replay_idx]:
            self._handle_event(ev)
        try:
            self.query_one("#replay-bar", ReplayBar).update_status(
                self._replay_speed, True, self._replay_idx, len(self._replay_events)
            )
        except Exception:
            pass

    def action_speed_1(self) -> None:
        if self._replay:
            self._replay_speed = 1.0

    def action_speed_4(self) -> None:
        if self._replay:
            self._replay_speed = 4.0

    def action_speed_instant(self) -> None:
        if self._replay:
            self._replay_speed = 0.0
