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
        "features": {"token_budgets": False},  # the script wanders through 20 calls: no per-phase call caps here
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


# ---- DeepSeek- and Qwen-shaped endpoints, whole runs ----------------------------------------------------------

THOUGHT = "Let me think about what the next step should be."


def _text_violations(messages: list[dict]) -> list[str]:
    """What a Mistral/Gemma/Qwen chat template rejects, checked on a text-mode request."""
    problems = []
    for i, m in enumerate(messages):
        if m["role"] == "system" and i != 0:
            problems.append("system message that is not first")
        if m["role"] not in ("system", "user", "assistant") or "tool_calls" in m or "tool_call_id" in m:
            problems.append(f"tool-calling fields or role {m['role']!r} in a text-mode request")
        if not m.get("content"):
            problems.append(f"empty {m['role']} message")
    turns = [m["role"] for m in messages if m["role"] != "system"]
    if turns[:1] != ["user"] or any(a == b for a, b in zip(turns, turns[1:])):
        problems.append(f"roles do not alternate: {turns}")
    return problems


def _dialect_server(script, dialect: str, problems: list[str], requests: list[dict]) -> httpx.MockTransport:
    replies = iter(script)

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "tools" not in payload:
            problems.extend(_text_violations(payload["messages"]))
        elif payload["messages"][0]["role"] != "system":
            problems.append("native request without a leading system message")
        scripted = next(replies)
        call = scripted.tool_calls[0]
        usage = dict(scripted.usage)
        message: dict = {"role": "assistant", "content": scripted.text}
        if dialect == "qwen-hermes":  # vLLM without a tool parser: the call arrives as text, `tools` is ignored
            arguments = json.dumps({"name": call["tool"], "arguments": call["args"]})
            message["content"] = f"<think>\n{THOUGHT}\n</think>\n\n{scripted.text}\n<tool_call>\n{arguments}\n</tool_call>".strip()
        elif dialect == "qwen-xml":  # Qwen3-Coder
            params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in call["args"].items())
            message["content"] = f"<think>{THOUGHT}</think>\n<tool_call>\n<function={call['tool']}>\n{params}</function>\n</tool_call>"
        elif dialect == "deepseek-native":
            message["tool_calls"] = [
                {"id": call["id"], "type": "function", "function": {"name": call["tool"], "arguments": json.dumps(call["args"])}}
            ]
            message["reasoning_content"] = THOUGHT
            usage["completion_tokens_details"] = {"reasoning_tokens": 7}
        return httpx.Response(200, json={"choices": [{"message": message}], "usage": usage})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "dialect, tool_mode",
    [
        ("qwen-hermes", "auto"),
        ("qwen-hermes", "text"),
        ("qwen-xml", "text"),
        ("deepseek-native", "auto"),
        ("deepseek-native", "native"),
    ],
)
def test_a_whole_run_through_a_deepseek_or_qwen_shaped_endpoint(tmp_path, monkeypatch, dialect, tool_mode):
    monkeypatch.setenv("AI_API_KEY", "test-key-not-a-secret")
    problems: list[str] = []
    requests: list[dict] = []
    model = "deepseek-chat" if dialect.startswith("deepseek") else "Qwen/Qwen2.5-Coder-32B-Instruct"
    config = {
        "model": model, "base_url": "http://provider.invalid/v1", "tool_mode": tool_mode, "output_dir": str(tmp_path / "out"),
    }
    client = make_client(config, transport=_dialect_server(happy(), dialect, problems, requests))
    bus = EventBus()
    queue = bus.subscribe()

    run_harness(ISSUE_URL, config, bus, llm=client, pipeline=FakePipeline())

    assert problems == []
    assert len(requests) == HAPPY_STEPS
    assert "+    return a + b" in (tmp_path / "out" / "patch.diff").read_text()
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [e.data for e in events if e.type == "error"] == []
    done = events[-1]
    assert done.type == "done" and done.data["resolved_confidence"] == pytest.approx(0.9)
    assert done.data["tokens"] == 100 * HAPPY_STEPS, "usage is the provider's, reasoning included"

    sent = " ".join(json.dumps(r["messages"]) for r in requests)
    assert THOUGHT not in sent and "<think>" not in sent, "reasoning must not travel back to the model in the history"
    assert requests[0]["model"] == model and requests[0]["max_tokens"] == 8192, "the DeepSeek/Qwen profile's output cap is sent"
    if dialect == "qwen-hermes" and tool_mode == "auto":
        assert client.active_tool_mode == "text", "two calls arrived as text, so the client moved to text mode for good"
        assert "tools" not in requests[-1]
    assert client.reasoning_seen
