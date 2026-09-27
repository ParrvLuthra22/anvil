"""Endpoints that bill for a reply and return none of it (seen on a real Qwen3-Coder provider behind OpenRouter).

With ``tools`` attached, the provider's own tool-call parser sometimes consumes the model's output and answers
``content: null`` with no ``tool_calls``. Nothing is left for the client to read, so in auto mode it repeats the
request in text mode, where no server-side parser is involved, and from the first such reply stays there.
"""

import httpx
import pytest

from anvil.llm import make_client
from tests.test_llm_client import CONFIG, CALL_BLOCK, TOOLS, USER, Server, _env, build, native_call, ok  # noqa: F401 (_env: autouse)

# What the provider actually sent: HTTP 200, no text, no calls, 17 tokens spent, finish_reason "stop".
EMPTY_MESSAGE = {"role": "assistant", "content": None, "refusal": None, "reasoning": None}


def reply(message=EMPTY_MESSAGE, *, finish="stop", usage=None) -> httpx.Response:
    body = {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}
    body["usage"] = {"prompt_tokens": 430, "completion_tokens": 17, "total_tokens": 447} if usage is None else usage
    return httpx.Response(200, json=body)


def swallowed(**kwargs) -> httpx.Response:
    return reply(**kwargs)


def sent_tools(server: Server, index: int) -> bool:
    return "tools" in server.body(index)


# ---- the retry -------------------------------------------------------------------------------------------


def test_a_swallowed_native_reply_is_repeated_in_text_mode_and_the_text_reply_is_returned():
    server = Server(swallowed(), ok(CALL_BLOCK))
    client, _ = build(server)

    response = client.chat(USER, TOOLS)

    assert response.tool_calls[0]["tool"] == "read_file" and response.tool_calls[0]["args"] == {"path": "a.py"}
    assert len(server.requests) == 2
    assert sent_tools(server, 0) and not sent_tools(server, 1), "the repeat carries the tools in the prompt, not the API"
    assert client.active_tool_mode == "text", "the first swallowed reply moves the rest of the run to text mode"


def test_the_wasted_calls_tokens_are_still_counted_in_the_response_and_in_the_totals():
    server = Server(swallowed(usage={"prompt_tokens": 430, "completion_tokens": 17, "total_tokens": 447}),
                    ok(CALL_BLOCK, usage={"prompt_tokens": 300, "completion_tokens": 27, "total_tokens": 327}))
    client, _ = build(server)

    response = client.chat(USER, TOOLS)

    assert response.usage["prompt_tokens"] == 730
    assert response.usage["completion_tokens"] == 44
    assert response.usage["total_tokens"] == 774, "the caller's budget must see what the provider billed"
    assert (client.totals.calls, client.totals.total_tokens) == (2, 774)


def test_the_first_swallowed_reply_switches_to_text_mode_for_good():
    server = Server(swallowed(), ok(CALL_BLOCK), ok(CALL_BLOCK), ok(CALL_BLOCK))
    client, _ = build(server)

    client.chat(USER, TOOLS)
    assert client.active_tool_mode == "text"

    client.chat(USER, TOOLS)
    client.chat(USER, TOOLS)
    assert len(server.requests) == 4 and not sent_tools(server, 2) and not sent_tools(server, 3), "later calls go straight to text mode"


def test_a_healthy_native_reply_is_not_repeated():
    server = Server(ok("", tool_calls=[native_call()]))
    client, _ = build(server)
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 1 and response.tool_calls[0]["tool"] == "read_file"


def test_a_native_reply_with_only_text_is_an_answer_not_a_swallowed_reply():
    server = Server(ok("I will look at it."))
    client, _ = build(server)
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 1 and response.text == "I will look at it."


# ---- what is not a swallowed reply ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, response",
    [
        ("cut off by the token limit", swallowed(finish="length")),
        ("no tokens were spent", swallowed(usage={"prompt_tokens": 430, "completion_tokens": 0, "total_tokens": 430})),
        ("the provider does not report usage", swallowed(usage={})),
        ("the model only reasoned", reply({"role": "assistant", "content": "", "reasoning_content": "Let me think."})),
        ("stripped think tags were the whole reply", reply({"role": "assistant", "content": "<think>hmm</think>"})),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_an_empty_reply_that_is_explained_otherwise_is_returned_as_it_is(case, response):
    server = Server(response, ok(CALL_BLOCK))
    client, _ = build(server)
    result = client.chat(USER, TOOLS)
    assert len(server.requests) == 1 and result.tool_calls == [] and result.text == "", case
    assert client.active_tool_mode == "native"


def test_without_tools_an_empty_reply_is_not_repeated():
    server = Server(swallowed(), ok("second"))
    client, _ = build(server)
    assert client.chat(USER).text == ""
    assert len(server.requests) == 1


def test_an_explicit_native_mode_stays_strict_and_returns_the_empty_reply():
    server = Server(swallowed(), ok(CALL_BLOCK))
    client, _ = build(server, tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 1 and response.tool_calls == [] and response.text == ""


def test_the_text_mode_repeat_failing_raises_the_llm_error_but_keeps_the_wasted_tokens_counted():
    from anvil.llm import LLMError

    server = Server(swallowed(), httpx.Response(401, json={"error": {"message": "bad key"}}))
    client, _ = build(server)
    with pytest.raises(LLMError):
        client.chat(USER, TOOLS)
    assert client.totals.total_tokens == 447 and client.totals.calls == 1


def test_a_repeat_that_is_empty_too_gets_the_text_mode_retries_and_then_is_returned_empty():
    server = Server(swallowed())
    client, _ = build(server)
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 4, "the native attempt, the text-mode repeat, and its two retries"
    assert response.tool_calls == [] and response.text == ""


def test_the_key_is_never_in_what_is_logged_about_a_swallowed_reply(caplog):
    from tests.test_llm_client import KEY

    server = Server(swallowed(), ok(CALL_BLOCK))
    client, _ = build(server)
    with caplog.at_level("DEBUG", logger="anvil.llm"):
        client.chat(USER, TOOLS)
    assert "text mode" in caplog.text and KEY not in caplog.text


# ---- text mode: the same provider swallows replies that carry no tools at all ----------------------------
# In a real run about a third of Qwen3-Coder's text-mode replies were content:null with 24-190 tokens billed
# (the model wrote its native <tool_call> XML and the provider's parser removed it). The agent loop would nudge once
# and end the phase, so the client retries first, twice at most, and the agent never sees the glitch.

TEXT = {"tool_mode": "text"}


def test_an_empty_text_mode_reply_is_retried_with_a_reminder_and_the_retry_is_returned():
    server = Server(swallowed(), ok(CALL_BLOCK))
    client, _ = build(server, **TEXT)

    response = client.chat(USER, TOOLS)

    assert response.tool_calls[0]["tool"] == "read_file"
    assert len(server.requests) == 2
    first, second = server.body(0), server.body(1)
    assert first["messages"][-1]["content"] == "hello"
    assert "empty" in second["messages"][-1]["content"].lower() and "hello" in second["messages"][-1]["content"]
    assert "json" in second["messages"][-1]["content"].lower(), "the reminder names the format to reply in"
    assert second["temperature"] > first["temperature"] == 0, "a greedy repeat would be swallowed the same way"


def test_the_reminder_is_for_that_call_only():
    server = Server(swallowed(), ok(CALL_BLOCK), ok(CALL_BLOCK))
    client, _ = build(server, **TEXT)
    client.chat(USER, TOOLS)
    client.chat(USER, TOOLS)
    assert "empty" not in server.body(2)["messages"][-1]["content"].lower()
    assert server.body(2)["temperature"] == 0


def test_the_wasted_attempts_tokens_are_summed_into_the_response_and_the_totals():
    empty = {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340}
    good = {"prompt_tokens": 320, "completion_tokens": 25, "total_tokens": 345}
    server = Server(swallowed(usage=empty), swallowed(usage=empty), ok(CALL_BLOCK, usage=good))
    client, _ = build(server, **TEXT)
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 3
    assert (response.usage["prompt_tokens"], response.usage["completion_tokens"], response.usage["total_tokens"]) == (920, 105, 1025)
    assert (client.totals.calls, client.totals.total_tokens) == (3, 1025)


def test_at_most_two_retries_then_the_empty_reply_is_returned_for_the_agent_to_deal_with():
    server = Server(swallowed())
    client, _ = build(server, **TEXT)
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 3 and response.text == "" and response.tool_calls == []


@pytest.mark.parametrize(
    "response",
    [
        swallowed(finish="length"),
        swallowed(usage={"prompt_tokens": 300, "completion_tokens": 0, "total_tokens": 300}),
        reply({"role": "assistant", "content": "", "reasoning_content": "thinking"}),
    ],
    ids=["cut off", "no tokens spent", "only reasoning"],
)
def test_an_empty_text_reply_that_is_explained_otherwise_is_not_retried(response):
    server = Server(response, ok(CALL_BLOCK))
    client, _ = build(server, **TEXT)
    assert client.chat(USER, TOOLS).tool_calls == [] and len(server.requests) == 1


def test_without_tools_an_empty_text_reply_is_retried_asking_for_a_plain_answer():
    server = Server(swallowed(), ok("the answer"))
    client, _ = build(server, **TEXT)
    response = client.chat(USER)
    assert response.text == "the answer" and len(server.requests) == 2
    reminder = server.body(1)["messages"][-1]["content"].lower()
    assert "empty" in reminder and "json" not in reminder


def test_a_reply_with_prose_but_no_call_is_an_answer_not_an_empty_reply():
    server = Server(ok("I will look at it."), ok(CALL_BLOCK))
    client, _ = build(server, **TEXT)
    assert client.chat(USER, TOOLS).text == "I will look at it." and len(server.requests) == 1


def test_no_temperature_is_sent_on_the_retry_when_the_endpoint_refused_one_earlier():
    server = Server(
        httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature'"}}),
        swallowed(),
        ok(CALL_BLOCK),
    )
    client, _ = build(server, **TEXT)
    response = client.chat(USER, TOOLS)
    assert response.tool_calls and len(server.requests) == 3  # rejected; re-sent without it and came back empty; retried
    assert "temperature" in server.body(0), "the request the endpoint rejected"
    assert all("temperature" not in server.body(i) for i in (1, 2)), "including the warmer retry"


def test_a_native_swallow_repeated_in_text_mode_gets_the_text_mode_retries_too():
    server = Server(swallowed(), swallowed(), ok(CALL_BLOCK))
    client, _ = build(server)  # auto
    response = client.chat(USER, TOOLS)
    assert response.tool_calls and len(server.requests) == 3
    assert "tools" in server.body(0) and "tools" not in server.body(1) and "tools" not in server.body(2)
