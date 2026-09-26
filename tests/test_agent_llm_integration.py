"""The orchestrator driving the real OpenAICompatClient over a mock HTTP server.

This is the check that the history the agent builds is something the client (and so a
real provider) accepts, in both tool dialects. The fake server validates every request.
"""

import json

import httpx
import pytest

from anvil.agent.orchestrator import run_harness
from anvil.agent.prompts import SUMMARIZER_PROMPT
from anvil.context.tokens import estimate_messages_tokens
from anvil.events import EventBus
from anvil.llm.client import make_client
from anvil.llm.toolcalls import render_call
from tests.fakes import ISSUE_URL, FakePipeline
from tests.test_agent_context import big_project_pipeline, wandering_script
from tests.test_orchestrator import HAPPY_STEPS, happy


def _violations(payload: dict, tool_mode: str) -> list[str]:
    """Ways in which a request would be rejected by a strict OpenAI-compatible provider."""
    problems = []
    messages = payload["messages"]
    if messages[0]["role"] != "system" or not messages[0]["content"]:
        problems.append("first message is not a non-empty system prompt")
    pending: set[str] = set()
    for msg in messages[1:]:
        role = msg["role"]
        if role not in ("user", "assistant", "tool"):
            problems.append(f"unexpected role {role!r}")
        if pending and role != "tool":
            problems.append(f"tool calls {sorted(pending)} were not answered before a {role} message")
            pending.clear()
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("type") != "function" or not isinstance(tc["function"].get("arguments"), str):
                    problems.append("malformed assistant tool_call")
                json.loads(tc["function"]["arguments"])
                pending.add(tc["id"])
        if role == "tool":
            if msg.get("tool_call_id") not in pending:
                problems.append(f"tool message answers unknown call {msg.get('tool_call_id')!r}")
            pending.discard(msg.get("tool_call_id"))
        if role != "assistant" and not isinstance(msg.get("content"), str):
            problems.append(f"{role} message without string content")
    if tool_mode == "native" and not payload.get("tools"):
        problems.append("no tools offered")
    return problems


def _server(
    script, tool_mode: str, problems: list[str], requests: list[dict], summaries: list[dict] | None = None
) -> httpx.MockTransport:
    replies = iter(script)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["messages"][0]["content"] == SUMMARIZER_PROMPT:
            if summaries is None:
                problems.append("unexpected summariser call")
            else:
                summaries.append(payload)
            reply_message = {"role": "assistant", "content": "Read helper functions in big.py; no bug there."}
            return httpx.Response(200, json={"choices": [{"message": reply_message}], "usage": {"total_tokens": 40}})
        requests.append(payload)
        problems.extend(_violations(payload, tool_mode))
        scripted = next(replies)
        message: dict = {"role": "assistant", "content": scripted.text}
        if tool_mode == "native":
            if scripted.tool_calls:
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["tool"], "arguments": json.dumps(c["args"])}}
                    for c in scripted.tool_calls
                ]
        else:
            blocks = [render_call(c["tool"], c["args"]) for c in scripted.tool_calls]
            message["content"] = "\n\n".join(p for p in (scripted.text, *blocks) if p)
        return httpx.Response(200, json={"choices": [{"message": message}], "usage": scripted.usage})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("tool_mode", ["native", "text"])
def test_happy_path_through_the_real_client_in_both_tool_dialects(tmp_path, monkeypatch, tool_mode):
    monkeypatch.setenv("AI_API_KEY", "test-key-not-a-secret")
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    problems: list[str] = []
    requests: list[dict] = []
    config = {
        "model": "fake-model",
        "base_url": "http://provider.invalid/v1",
        "tool_mode": tool_mode,
        "output_dir": str(tmp_path / "out"),
    }
    client = make_client(config, transport=_server(happy(), tool_mode, problems, requests))
    bus = EventBus()
    queue = bus.subscribe()

    run_harness(ISSUE_URL, config, bus, llm=client, pipeline=FakePipeline())

    assert problems == []
    assert len(requests) == HAPPY_STEPS
    patch = (tmp_path / "out" / "patch.diff").read_text()
    assert "+    return a + b" in patch and ".anvil" not in patch
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert not [e for e in events if e.type == "error"]
    assert events[-1].type == "done" and events[-1].data["resolved_confidence"] == pytest.approx(0.9)
    assert "test-key-not-a-secret" not in (tmp_path / "out" / "report.md").read_text()


def test_the_native_requests_carry_the_phase_tools_and_the_control_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "k")
    problems, requests = [], []
    config = {"model": "m", "base_url": "http://provider.invalid/v1", "tool_mode": "native", "output_dir": str(tmp_path / "o")}
    client = make_client(config, transport=_server(happy(), "native", problems, requests))
    run_harness(ISSUE_URL, config, EventBus(), llm=client, pipeline=FakePipeline())

    first, second = requests[0], requests[1]
    assert {t["function"]["name"] for t in first["tools"]} == {"phase_done", "give_up"}
    assert {t["function"]["name"] for t in second["tools"]} == {"list_dir", "grep", "read_file", "phase_done", "give_up"}
    assert first["temperature"] == 0


@pytest.mark.parametrize("tool_mode", ["native", "text"])
def test_a_compacted_history_is_still_accepted_by_a_strict_provider(tmp_path, monkeypatch, tool_mode):
    monkeypatch.setenv("AI_API_KEY", "test-key-not-a-secret")
    problems: list[str] = []
    requests: list[dict] = []
    summaries: list[dict] = []
    budget = 2500
    config = {
        "model": "fake-model",
        "base_url": "http://provider.invalid/v1",
        "tool_mode": tool_mode,
        "output_dir": str(tmp_path / "out"),
        "max_context_tokens": budget,
        "context_keep_steps": 8,
        "tool_output_char_cap": 1500,
    }
    transport = _server(wandering_script(20), tool_mode, problems, requests, summaries)
    client = make_client(config, transport=transport)
    bus = EventBus()
    queue = bus.subscribe()

    run_harness(ISSUE_URL, config, bus, llm=client, pipeline=big_project_pipeline())

    assert problems == []
    assert summaries, "the history never outgrew the budget"
    assert [m["role"] for m in summaries[0]["messages"]] == ["system", "user"] and "tools" not in summaries[0]
    assert "big.py" in summaries[0]["messages"][1]["content"]
    assert max(estimate_messages_tokens(r["messages"]) for r in requests) <= budget + 400  # text mode adds tool docs
    assert "+    return a + b" in (tmp_path / "out" / "patch.diff").read_text()
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert not [e for e in events if e.type == "error"]
    assert events[-1].type == "done" and events[-1].data["resolved_confidence"] == pytest.approx(0.9)
