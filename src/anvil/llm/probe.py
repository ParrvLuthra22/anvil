"""``python -m anvil.llm.probe``: find out how the configured endpoint really behaves, before a run depends on it.

It sends a tiny two-tool task (``add`` and ``shout``; "add 17 and 25") to the endpoint in each tool mode. The task is
two turns: the model must call ``add`` with the right numbers, then, given the tool's result in the history, answer.
The second turn is the part that matters: it proves the endpoint accepts our history format (assistant tool calls, tool
results) and not just our first message.

Per mode it reports whether it worked, the latency and token usage of each turn, whether reasoning (``<think>`` tags or
a reasoning field) appeared, and every HTTP 400 the client worked around (parameters dropped, roles normalised). It ends
with the ``tool_mode`` to put in ``config.yaml``.

The API key comes from ``AI_API_KEY`` as always and is never printed: every line of output is scrubbed of it, and the base
URL is shown without credentials or query string. Exit status: 0 if some mode worked, 1 if none did, 2 for a setup problem
(no key, unreadable config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

from anvil.llm.client import LLMResponse, OpenAICompatClient, make_client
from anvil.llm.config import LLMConfig
from anvil.llm.errors import LLMConfigError, LLMError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
MODES = ("native", "text", "auto")
EXPECTED = 17 + 25

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shout",
            "description": "Return the given text in upper case.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
]
SYSTEM = "You are a careful assistant. Use the tools you are given whenever they can answer the question."
QUESTION = "Use the add tool to add 17 and 25, then tell me the result in one short sentence."


@dataclass
class Turn:
    """One request of the probe and what came of it."""

    ok: bool
    seconds: float
    summary: str
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    response: LLMResponse | None = None


@dataclass
class ModeResult:
    mode: str
    turns: list[Turn]
    works: bool
    reasoning: bool
    reasoning_replies: int
    workarounds: list[str]
    active_mode: str


def main(
    argv: list[str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """Run the probe; ``transport`` and ``out`` let tests run it offline and capture what it prints."""
    args = _parser().parse_args(argv)
    key = os.environ.get("AI_API_KEY", "").strip()

    def emit(line: str = "") -> None:
        out(line.replace(key, "[REDACTED]") if key else line)

    try:
        config = _load_config(args.config)
    except (OSError, yaml.YAMLError) as exc:
        emit(f"Cannot read the config file {args.config}: {type(exc).__name__}: {exc}")
        return 2
    modes = MODES[:2] if args.mode == "both" else (args.mode,)
    try:
        llm_config = LLMConfig.from_mapping(config)
        _require_key()
    except LLMConfigError as exc:
        emit(f"Setup problem: {exc}")
        return 2

    _describe(llm_config, emit)
    results = []
    for mode in modes:
        emit()
        emit(f"== tool mode: {mode} ==")
        try:
            client = make_client(
                {**config, "tool_mode": mode, "llm_max_attempts": min(int(config.get("llm_max_attempts", 5)), args.attempts)},
                transport=transport,
            )
        except LLMConfigError as exc:
            emit(f"Setup problem: {exc}")
            return 2
        try:
            result = _run_mode(mode, client, emit)
        finally:
            client.close()
        results.append(result)
    _summarise(results, llm_config, emit)
    return 0 if any(r.works for r in results) else 1


# ---- one mode -----------------------------------------------------------------------------------------


def _run_mode(mode: str, client: OpenAICompatClient, emit: Callable[[str], None]) -> ModeResult:
    history: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]
    first = _timed(client, history, _judge_first)
    emit(_line("turn 1", first))
    turns = [first]
    if first.ok and first.response is not None:
        call = first.response.tool_calls[0]
        history.append(
            {
                "role": "assistant",
                "content": first.response.text,
                "tool_calls": [
                    {"id": call["id"], "type": "function", "function": {"name": call["tool"], "arguments": json.dumps(call["args"])}}
                ],
            }
        )
        history.append({"role": "tool", "tool_call_id": call["id"], "content": str(EXPECTED)})
        second = _timed(client, history, _judge_second)
        emit(_line("turn 2", second))
        turns.append(second)
    works = len(turns) == 2 and all(t.ok for t in turns)
    tags = "no"
    if client.reasoning_seen:
        tags = f"yes, in {client.totals.reasoning_replies} of {client.totals.calls} replies (removed from the text automatically)"
    emit(f"  reasoning seen: {tags}")
    if client.workarounds:
        emit("  worked around:")
        for note in client.workarounds:
            emit(f"    - {note}")
    else:
        emit("  worked around: nothing (no HTTP 400s)")
    if mode == "auto":
        emit(f"  auto mode settled on: {client.active_tool_mode}")
    emit(f"  result: {'WORKS' if works else 'DOES NOT WORK'}")
    return ModeResult(
        mode, turns, works, client.reasoning_seen, client.totals.reasoning_replies, list(client.workarounds), client.active_tool_mode
    )


def _timed(client: OpenAICompatClient, history: list[dict], judge: Callable[[LLMResponse], tuple[bool, str]]) -> Turn:
    started = time.monotonic()
    try:
        response = client.chat(history, TOOLS)
    except LLMError as exc:
        return Turn(False, time.monotonic() - started, "", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a diagnostic tool reports, it does not crash
        return Turn(False, time.monotonic() - started, "", error=f"{type(exc).__name__}: {exc}")
    ok, summary = judge(response)
    return Turn(ok, time.monotonic() - started, summary, response.usage, response=response)


def _judge_first(response: LLMResponse) -> tuple[bool, str]:
    if not response.tool_calls:
        return False, f"no tool call; the model said: {_clip(response.text) or '(nothing)'}"
    call = response.tool_calls[0]
    shown = f"{call['tool']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})"
    if call.get("error"):
        return False, f"{shown}: {call['error']}"
    if call["tool"] != "add":
        return False, f"called {shown} instead of add"
    numbers = [_as_number(call["args"].get(k)) for k in ("a", "b")]
    if None in numbers or sorted(numbers) != [17, 25]:
        return False, f"add called with the wrong arguments: {shown}"
    return True, shown


def _judge_second(response: LLMResponse) -> tuple[bool, str]:
    if response.tool_calls:
        return True, f"the endpoint accepted the tool result, but the model called {response.tool_calls[0]['tool']} again"
    text = response.text.strip()
    if not text:
        return False, "the endpoint accepted the tool result but the reply was empty"
    return True, f"{'correct' if str(EXPECTED) in text else 'no 42 in'} answer: {_clip(text)}"


# ---- output ----------------------------------------------------------------------------------------------


def _describe(cfg: LLMConfig, emit: Callable[[str], None]) -> None:
    emit("ANVIL LLM probe")
    rows = [
        ("model", cfg.model),
        ("endpoint", safe_url(cfg.base_url)),
        ("profile", cfg.profile),
        ("tool_mode", f"{cfg.tool_mode} (as configured; the probe tries each mode itself)"),
        ("temperature", f"{cfg.temperature:g}"),
        ("max_output", cfg.max_output_tokens if cfg.max_output_tokens is not None else "provider default"),
        ("strip_reasoning", str(cfg.strip_reasoning).lower()),
    ]
    if cfg.extra_params:
        rows.append(("extra params", ", ".join(sorted(cfg.extra_params))))
    for label, value in rows:
        emit(f"  {label + ':':<17}{value}")


def _summarise(results: list[ModeResult], cfg: LLMConfig, emit: Callable[[str], None]) -> None:
    emit()
    emit("== summary ==")
    for result in results:
        emit(f"  {result.mode:<6} {'works' if result.works else 'FAILS'}")
    worked = [r.mode for r in results if r.works]
    if worked:
        best = "native" if "native" in worked else worked[0]
        emit(f"  Recommended tool_mode: {best}" + ("   (text also works, as a fallback)" if best == "native" and "text" in worked else ""))
    else:
        emit("  No tool mode worked: see the errors above (credentials, model name, base_url, or an endpoint without chat completions).")
    seen_reasoning = any(r.reasoning for r in results)
    if seen_reasoning:
        emit(f"  Reasoning appeared; it is removed from replies and kept out of the history{'' if cfg.strip_reasoning else ' ONLY if strip_reasoning is true (it is off)'}.")
    if cfg.temperature == 0 and (seen_reasoning or cfg.profile.endswith("-reasoning")):
        emit("  WARNING: temperature is 0 with a reasoning model. Vendors recommend 0.6; greedy decoding makes these models repeat")
        emit("           themselves. Delete `temperature` from config.yaml to use the profile's default, or set it.")
    if any(r.workarounds for r in results):
        emit("  The endpoint rejected some request fields; the client copes, but you can stop sending them (see config.yaml).")


def _line(label: str, turn: Turn) -> str:
    if turn.error:
        return f"  {label}  FAILED  {turn.seconds * 1000:.0f} ms  {turn.error}"
    tokens = _tokens(turn.usage)
    status = "ok    " if turn.ok else "WRONG "
    return f"  {label}  {status} {turn.seconds * 1000:.0f} ms  {turn.summary}\n           tokens: {tokens}"


def _tokens(usage: dict[str, Any]) -> str:
    text = f"{usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = {usage.get('total_tokens', 0)}"
    if usage.get("reasoning_tokens"):
        text += f" (of which {usage['reasoning_tokens']} reasoning)"
    return text + (" [estimated: the provider reported no usage]" if usage.get("estimated") else "")


def safe_url(url: str) -> str:
    """``url`` without credentials, query string or fragment (some providers put the key in the URL)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


# ---- helpers ----------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anvil.llm.probe",
        description="Check how the configured LLM endpoint handles tool calling, reasoning output and request quirks.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="config.yaml to read (default: the repo's)")
    parser.add_argument(
        "--mode",
        choices=("both", *MODES),
        default="both",
        help="tool mode to try: native, text, both (default), or auto (the client's own fallback)",
    )
    parser.add_argument("--attempts", type=int, default=2, help="attempts per request on 429/5xx (default 2: a probe should be quick)")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise yaml.YAMLError("the top level of the config must be a mapping")
    return loaded


def _require_key() -> None:
    if not os.environ.get("AI_API_KEY", "").strip():
        raise LLMConfigError("AI_API_KEY is not set: export it first (it is read from the environment only and never printed)")


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _clip(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


if __name__ == "__main__":
    sys.exit(main())
