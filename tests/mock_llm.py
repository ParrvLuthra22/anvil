"""Scripted LLM double and fake-event-stream helper so tests/TUI dev never touch the network."""

from __future__ import annotations

import time
from collections.abc import Iterator

from anvil.events import AgentEvent, Phase
from anvil.llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# MockLLM — replays scripted responses in order
# ---------------------------------------------------------------------------

class MockLLM:
    """Replays a scripted list of :class:`~anvil.llm.client.LLMResponse` objects,
    one per :meth:`chat` call, in order.

    Raises :class:`RuntimeError` if the script runs out so test failures are
    obvious rather than silent.
    """

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self._next = 0

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """Return the next scripted response; raise if the script has run out."""
        if self._next >= len(self._script):
            raise RuntimeError(
                f"MockLLM script exhausted after {len(self._script)} responses"
            )
        response = self._script[self._next]
        self._next += 1
        return response


# ---------------------------------------------------------------------------
# fake_event_stream — deterministic stream for TUI development / replay tests
# ---------------------------------------------------------------------------

_PHASES_IN_ORDER: list[Phase] = [
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


def fake_event_stream(
    issue_url: str = "https://github.com/example/repo/issues/1",
    base_ts: float | None = None,
) -> list[AgentEvent]:
    """Return a deterministic list of :class:`AgentEvent` objects covering every
    phase, tool calls, tool results, LLM usage events, and a final *done* event.

    This stream is consumed by the TUI in demo / replay mode so UI development
    requires no API key and no network.

    Args:
        issue_url: The fictional issue URL embedded in the first INGEST message.
        base_ts:   Timestamp of the first event; defaults to ``time.time()``.
    """
    t = base_ts if base_ts is not None else time.time()
    events: list[AgentEvent] = []

    def _ev(ev_type: str, phase: Phase | None, data: dict, offset: float = 0.0) -> AgentEvent:
        nonlocal t
        t += offset
        return AgentEvent(ts=t, type=ev_type, phase=phase, data=data)

    total_prompt = 0
    total_completion = 0

    for i, phase in enumerate(_PHASES_IN_ORDER):
        # ── Phase transition ────────────────────────────────────────────────
        events.append(_ev("phase", phase, {"name": phase.value}, offset=0.5))

        # ── Synthetic assistant message ─────────────────────────────────────
        events.append(
            _ev(
                "message",
                phase,
                {
                    "role": "assistant",
                    "text": (
                        f"[{phase.value.upper()}] Analysing… "
                        f"This is a simulated step {i + 1} of 9."
                    ),
                },
                offset=0.3,
            )
        )

        # ── LLM usage ───────────────────────────────────────────────────────
        prompt_tok = 800 + i * 120
        completion_tok = 200 + i * 40
        total_prompt += prompt_tok
        total_completion += completion_tok
        events.append(
            _ev(
                "llm_usage",
                phase,
                {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": completion_tok,
                    "total_tokens": prompt_tok + completion_tok,
                    "cost_estimate": round((prompt_tok + completion_tok) / 1_000_000 * 0.15, 6),
                },
                offset=0.2,
            )
        )

        # ── Phase-specific tool calls ────────────────────────────────────────
        if phase == Phase.INGEST:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {"tool": "fetch_issue", "args": {"url": issue_url}},
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "fetch_issue",
                        "ok": True,
                        "output_preview": "Issue #1: KeyError when x is None — body: 42 chars",
                    },
                    offset=0.4,
                )
            )

        elif phase == Phase.PROFILE:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {"tool": "list_dir", "args": {"path": "."}},
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "list_dir",
                        "ok": True,
                        "output_preview": "src/  tests/  README.md  pyproject.toml",
                    },
                    offset=0.3,
                )
            )

        elif phase == Phase.UNDERSTAND:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {"tool": "grep", "args": {"pattern": "KeyError", "path": "src/"}},
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "grep",
                        "ok": True,
                        "output_preview": "src/lib/core.py:87:  raise KeyError(x)",
                    },
                    offset=0.35,
                )
            )

        elif phase == Phase.LOCALIZE:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {
                        "tool": "read_file",
                        "args": {"path": "src/lib/core.py", "start": 80, "end": 100},
                    },
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "read_file",
                        "ok": True,
                        "output_preview": "80: def process(x):\n81:     if x:\n...\n87:         raise KeyError(x)",
                    },
                    offset=0.3,
                )
            )

        elif phase == Phase.REPRODUCE:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {
                        "tool": "run_cmd",
                        "args": {"cmd": "python -m pytest tests/test_repro.py -x", "timeout": 30},
                    },
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "run_cmd",
                        "ok": True,
                        "output_preview": "FAILED tests/test_repro.py::test_none_key — KeyError: None",
                    },
                    offset=0.8,
                )
            )

        elif phase == Phase.PATCH:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "src/lib/core.py",
                            "old": "    if x:\n        raise KeyError(x)",
                            "new": "    if x is None:\n        return None",
                        },
                    },
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {"tool": "edit_file", "ok": True, "output_preview": "Patched 2 lines."},
                    offset=0.4,
                )
            )

        elif phase == Phase.VERIFY:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {"tool": "run_tests", "args": {"target": None}},
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "run_tests",
                        "ok": True,
                        "output_preview": "5 passed in 1.23s",
                    },
                    offset=1.2,
                )
            )

        elif phase == Phase.REVIEW:
            events.append(
                _ev(
                    "tool_call",
                    phase,
                    {"tool": "git_diff", "args": {}},
                    offset=0.1,
                )
            )
            events.append(
                _ev(
                    "tool_result",
                    phase,
                    {
                        "tool": "git_diff",
                        "ok": True,
                        "output_preview": (
                            "--- a/src/lib/core.py\n"
                            "+++ b/src/lib/core.py\n"
                            "@@ -80,3 +80,3 @@\n"
                            "-    if x:\n"
                            "-        raise KeyError(x)\n"
                            "+    if x is None:\n"
                            "+        return None"
                        ),
                    },
                    offset=0.3,
                )
            )

        elif phase == Phase.FINALIZE:
            events.append(
                _ev(
                    "message",
                    phase,
                    {"role": "assistant", "text": "Patch verified. Writing output files."},
                    offset=0.2,
                )
            )

    # ── Final done event ────────────────────────────────────────────────────
    events.append(
        _ev(
            "done",
            Phase.FINALIZE,
            {
                "resolved_confidence": 0.92,
                "patch_path": "output/patch.diff",
                "report_path": "output/report.md",
                "steps": 9,
                "tokens": total_prompt + total_completion,
                "seconds": round(t - (base_ts or 0), 1),
            },
            offset=0.5,
        )
    )

    return events
