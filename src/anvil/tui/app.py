"""ANVIL Textual TUI — main application.

Layout
------
┌─────────────────────────────────────────────────────────────┐
│  Title bar  │  Issue URL input  │  [Start]                  │
├─────────────┬──────────────────────────────┬────────────────┤
│  Phase      │  Scrolling event log          │  Metrics       │
│  stepper    │  (messages / tools / results) │  (tokens, cost)│
├─────────────┴──────────────────────────────┴────────────────┤
│  Diff preview pane  (toggle with d)                         │
│  RESULT panel (appears on done event)                        │
└─────────────────────────────────────────────────────────────┘

Keyboard: q → quit, r → restart, d → toggle diff pane.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
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
# Phase display helpers
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
    Phase.INGEST: "① INGEST",
    Phase.PROFILE: "② PROFILE",
    Phase.UNDERSTAND: "③ UNDERSTAND",
    Phase.LOCALIZE: "④ LOCALIZE",
    Phase.REPRODUCE: "⑤ REPRODUCE",
    Phase.PATCH: "⑥ PATCH",
    Phase.VERIFY: "⑦ VERIFY",
    Phase.REVIEW: "⑧ REVIEW",
    Phase.FINALIZE: "⑨ FINALIZE",
}

_PHASE_STATE_CSS: dict[str, str] = {
    "pending": "phase-pending",
    "active": "phase-active",
    "done": "phase-done",
    "failed": "phase-failed",
}


# ---------------------------------------------------------------------------
# Phase stepper widget
# ---------------------------------------------------------------------------

class PhaseStepper(Widget):
    """Left column: shows each phase with pending / active / done / failed state."""

    DEFAULT_CSS = """
    PhaseStepper {
        width: 20;
        height: 100%;
        border-right: solid $primary-darken-2;
        padding: 1;
        background: $surface;
    }
    .phase-pending  { color: $text-muted; }
    .phase-active   { color: $warning; text-style: bold; }
    .phase-done     { color: $success; }
    .phase-failed   { color: $error; text-style: bold; }
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
        # Remove old class, add new
        for css_class in _PHASE_STATE_CSS.values():
            lbl.remove_class(css_class)
        lbl.add_class(_PHASE_STATE_CSS.get(state, "phase-pending"))
        prefix = {"pending": "  ", "active": "▶ ", "done": "✓ ", "failed": "✗ "}.get(state, "  ")
        lbl.update(prefix + _PHASE_LABEL[phase])

    def mark_active(self, phase: Phase) -> None:
        # Mark previous phases done
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


# ---------------------------------------------------------------------------
# Metrics widget
# ---------------------------------------------------------------------------

class MetricsPanel(Widget):
    """Right column: live step / token / cost / time metrics."""

    DEFAULT_CSS = """
    MetricsPanel {
        width: 22;
        height: 100%;
        border-left: solid $primary-darken-2;
        padding: 1;
        background: $surface;
    }
    .metric-label { color: $text-muted; }
    .metric-value { color: $accent; text-style: bold; }
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
        yield Label("Steps", classes="metric-label")
        yield Label("0", id="m-steps", classes="metric-value")
        yield Label("Prompt tokens", classes="metric-label")
        yield Label("0", id="m-prompt", classes="metric-value")
        yield Label("Completion tok", classes="metric-label")
        yield Label("0", id="m-completion", classes="metric-value")
        yield Label("Total tokens", classes="metric-label")
        yield Label("0", id="m-total-tok", classes="metric-value")
        yield Label("Est. cost ($)", classes="metric-label")
        yield Label("0.000000", id="m-cost", classes="metric-value")
        yield Label("Elapsed", classes="metric-label")
        yield Label("0s", id="m-elapsed", classes="metric-value")
        yield Static("─" * 18)
        yield Label("Budget remaining", classes="metric-label")
        yield Label("", id="m-budget", classes="metric-value")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        elapsed = int(time.time() - self._start_ts)
        self.query_one("#m-elapsed", Label).update(f"{elapsed}s")
        tok_remain = self._max_tokens - (self._prompt_tok + self._completion_tok)
        step_remain = self._max_steps - self._steps
        self.query_one("#m-budget", Label).update(
            f"steps:{step_remain}  tok:{tok_remain:,}"
        )

    def update_usage(self, prompt: int, completion: int, cost: float) -> None:
        self._prompt_tok += prompt
        self._completion_tok += completion
        self._cost += cost
        self._steps += 1
        total = self._prompt_tok + self._completion_tok
        self.query_one("#m-steps", Label).update(str(self._steps))
        self.query_one("#m-prompt", Label).update(f"{self._prompt_tok:,}")
        self.query_one("#m-completion", Label).update(f"{self._completion_tok:,}")
        self.query_one("#m-total-tok", Label).update(f"{total:,}")
        self.query_one("#m-cost", Label).update(f"{self._cost:.6f}")


# ---------------------------------------------------------------------------
# Diff pane widget
# ---------------------------------------------------------------------------

class DiffPane(Widget):
    """Bottom pane: shows the current unified diff, syntax-coloured."""

    DEFAULT_CSS = """
    DiffPane {
        height: 12;
        border-top: solid $primary-darken-2;
        padding: 0 1;
        background: $surface-darken-1;
        display: block;
    }
    DiffPane.hidden { display: none; }
    """

    def compose(self) -> ComposeResult:
        yield Label("── DIFF PREVIEW ─────────────────────────────", id="diff-header")
        yield RichLog(id="diff-log", highlight=True, markup=True, wrap=False)

    def update_diff(self, diff_text: str) -> None:
        """Replace the diff log with new content, coloured."""
        log = self.query_one("#diff-log", RichLog)
        log.clear()
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                log.write(f"[green]{line}[/green]")
            elif line.startswith("-") and not line.startswith("---"):
                log.write(f"[red]{line}[/red]")
            elif line.startswith("@@"):
                log.write(f"[cyan]{line}[/cyan]")
            else:
                log.write(line)


# ---------------------------------------------------------------------------
# Result banner widget
# ---------------------------------------------------------------------------

class ResultBanner(Widget):
    """Appears at the bottom when the done event arrives."""

    DEFAULT_CSS = """
    ResultBanner {
        height: 4;
        border: double $success;
        padding: 0 2;
        background: $success-darken-3;
        display: none;
    }
    ResultBanner.visible { display: block; }
    """

    def compose(self) -> ComposeResult:
        yield Label("", id="result-text")

    def show(self, data: dict) -> None:
        conf = data.get("resolved_confidence", 0)
        patch = data.get("patch_path", "output/patch.diff")
        steps = data.get("steps", "?")
        tokens = data.get("tokens", "?")
        secs = data.get("seconds", "?")
        text = (
            f"[bold green]✓ DONE[/bold green]  "
            f"confidence={conf:.0%}  steps={steps}  tokens={tokens:,}  time={secs}s  "
            f"patch → [link]{patch}[/link]"
        )
        self.query_one("#result-text", Label).update(text)
        self.add_class("visible")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class AnvilApp(App):
    """ANVIL Textual application.

    Consumes :class:`~anvil.events.AgentEvent` objects from *bus* and renders
    them. Works identically for live runs and replays.
    """

    TITLE = "ANVIL — Autonomous Coding Agent"
    CSS = """
    Screen {
        background: $background;
    }

    /* ── Title / Input bar ── */
    #top-bar {
        height: 3;
        background: $primary-darken-3;
        padding: 0 1;
        border-bottom: solid $primary-darken-1;
    }
    #app-title {
        width: 28;
        color: $accent;
        text-style: bold;
        padding: 0 1;
        content-align: left middle;
    }
    #issue-input {
        width: 1fr;
        border: none;
    }
    #start-btn {
        width: 10;
        margin: 0 1;
    }

    /* ── Body ── */
    #body {
        height: 1fr;
    }

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

    /* ── Phase header label ── */
    #phase-header {
        color: $accent;
        text-style: bold;
        padding-bottom: 1;
    }

    /* ── Metrics header label ── */
    #metrics-header {
        color: $accent;
        text-style: bold;
        padding-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "restart", "Restart"),
        Binding("d", "toggle_diff", "Toggle diff"),
    ]

    def __init__(
        self,
        bus: EventBus,
        issue_url: str = "",
        on_start: Callable[[str], None] | None = None,
        replay: bool = False,
        config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bus = bus
        self._issue_url = issue_url
        self._on_start = on_start
        self._replay = replay
        self._config = config or {}
        self._queue: asyncio.Queue[AgentEvent] | None = None
        self._started = False
        self._diff_visible = True
        self._current_phase: Phase | None = None

    # ── Layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Top bar
        with Horizontal(id="top-bar"):
            yield Label("⚒  ANVIL", id="app-title")
            yield Input(
                value=self._issue_url,
                placeholder="GitHub issue URL…",
                id="issue-input",
            )
            yield Button("▶  Start", id="start-btn", variant="primary")

        # Body: phase stepper | log | metrics
        with Horizontal(id="body"):
            yield PhaseStepper(id="phase-stepper")
            with ScrollableContainer(id="log-area"):
                yield RichLog(
                    id="event-log",
                    highlight=True,
                    markup=True,
                    wrap=True,
                )
            yield MetricsPanel(config=self._config, id="metrics-panel")

        # Diff pane (toggled with d)
        yield DiffPane(id="diff-pane")

        # Result banner (hidden until done)
        yield ResultBanner(id="result-banner")

        yield Footer()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and optionally auto-start if issue_url provided."""
        self._queue = self._bus.subscribe()
        self.set_interval(0.05, self._drain_queue)

        if self._issue_url and not self._replay:
            # Auto-start when URL was passed via --issue flag
            self._do_start(self._issue_url)
        elif self._replay:
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

    # ── Event handling ───────────────────────────────────────────────────────

    def _handle_event(self, ev: AgentEvent) -> None:  # noqa: C901 (complexity OK here)
        """Dispatch an AgentEvent to the appropriate UI update."""
        try:
            ev_type = ev.type
            phase = ev.phase
            data = ev.data or {}

            stepper: PhaseStepper = self.query_one("#phase-stepper", PhaseStepper)
            log: RichLog = self.query_one("#event-log", RichLog)
            metrics: MetricsPanel = self.query_one("#metrics-panel", MetricsPanel)

            if ev_type == "phase":
                new_phase = phase
                if new_phase and isinstance(new_phase, Phase):
                    # Mark previous phase done
                    if self._current_phase and self._current_phase != new_phase:
                        stepper.mark_done(self._current_phase)
                    self._current_phase = new_phase
                    stepper.mark_active(new_phase)
                    log.write(
                        f"\n[bold cyan]━━ {new_phase.value.upper()} ━━[/bold cyan]"
                    )

            elif ev_type == "message":
                role = data.get("role", "?")
                text = data.get("text", "")
                colour = "yellow" if role == "assistant" else "white"
                label = "🤖" if role == "assistant" else "👤"
                log.write(f"[{colour}]{label} [{role}][/{colour}] {text}")

            elif ev_type == "tool_call":
                tool = data.get("tool", "?")
                args = data.get("args", {})
                # Pretty-print args compactly
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items()) if args else ""
                log.write(f"[bold blue]⚙ CALL[/bold blue] [cyan]{tool}[/cyan]({args_str})")

            elif ev_type == "tool_result":
                tool = data.get("tool", "?")
                ok = data.get("ok", True)
                preview = data.get("output_preview", "")
                icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
                log.write(f"  {icon} [dim]{tool}[/dim]: {preview}")
                # Update diff pane if this looks like a git_diff result
                if tool == "git_diff" and ok and preview:
                    self.query_one("#diff-pane", DiffPane).update_diff(preview)

            elif ev_type == "llm_usage":
                prompt = data.get("prompt_tokens", 0)
                comp = data.get("completion_tokens", 0)
                cost = data.get("cost_estimate", 0.0)
                metrics.update_usage(prompt, comp, cost)
                log.write(
                    f"  [dim]tokens: +{prompt}p +{comp}c  cost Δ${cost:.6f}[/dim]"
                )

            elif ev_type == "error":
                kind = data.get("kind", "error")
                msg = data.get("message", "")
                log.write(f"[bold red]⚠ ERROR[/bold red] [{kind}] {msg}")

            elif ev_type == "done":
                if self._current_phase:
                    stepper.mark_all_done()
                self.query_one("#result-banner", ResultBanner).show(data)
                patch_path = data.get("patch_path", "")
                if patch_path:
                    import os
                    if os.path.exists(patch_path):
                        with open(patch_path) as f:
                            self.query_one("#diff-pane", DiffPane).update_diff(f.read())

            # Ignore unknown event types gracefully (future-proof)

        except Exception:
            # The TUI must never crash on a bad event
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
        """Trigger the harness for *url* (idempotent — only runs once)."""
        if self._started:
            return
        self._started = True
        self.query_one("#issue-input", Input).value = url
        log: RichLog = self.query_one("#event-log", RichLog)
        log.write(f"[bold green]▶ Starting run for:[/bold green] {url}\n")
        if self._on_start:
            self._on_start(url)

    # ── Key bindings ──────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        self.exit()

    def action_restart(self) -> None:
        self._started = False
        log: RichLog = self.query_one("#event-log", RichLog)
        log.clear()
        stepper: PhaseStepper = self.query_one("#phase-stepper", PhaseStepper)
        for p in _PHASE_ORDER:
            stepper.set_phase_state(p, "pending")
        self._current_phase = None

    def action_toggle_diff(self) -> None:
        diff_pane = self.query_one("#diff-pane", DiffPane)
        self._diff_visible = not self._diff_visible
        if self._diff_visible:
            diff_pane.remove_class("hidden")
        else:
            diff_pane.add_class("hidden")
