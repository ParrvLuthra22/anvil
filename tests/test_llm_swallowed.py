"""Endpoints that bill for a reply and return none of it (seen on a real Qwen3-Coder provider behind OpenRouter).

With ``tools`` attached, the provider's own tool-call parser sometimes consumes the model's output and answers
``content: null`` with no ``tool_calls``. Nothing is left for the client to read, so in auto mode it repeats the
request in text mode, where no server-side parser is involved, and after two such replies stays there.
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
    assert client.active_tool_mode == "native", "one swallowed reply is not yet a reason to leave native mode"


def test_the_wasted_calls_tokens_are_still_counted_in_the_response_and_in_the_totals():
    server = Server(swallowed(usage={"prompt_tokens": 430, "completion_tokens": 17, "total_tokens": 447}),
                    ok(CALL_BLOCK, usage={"prompt_tokens": 300, "completion_tokens": 27, "total_tokens": 327}))
    client, _ = build(server)

    response = client.chat(USER, TOOLS)

    assert response.usage["prompt_tokens"] == 730
    assert response.usage["completion_tokens"] == 44
    assert response.usage["total_tokens"] == 774, "the caller's budget must see what the provider billed"
    assert (client.totals.calls, client.totals.total_tokens) == (2, 774)


def test_two_swallowed_replies_switch_to_text_mode_for_good():
    server = Server(swallowed(), ok(CALL_BLOCK), swallowed(), ok(CALL_BLOCK), ok(CALL_BLOCK))
    client, _ = build(server)

    client.chat(USER, TOOLS)
    assert client.active_tool_mode == "native"
    client.chat(USER, TOOLS)
    assert client.active_tool_mode == "text"

    client.chat(USER, TOOLS)
    assert len(server.requests) == 5 and not sent_tools(server, 4), "the third call goes straight to text mode"


def test_the_swallowed_replies_need_not_be_consecutive():
    """A provider that drops a third of its native replies is not reliable however the good ones fall between."""
    server = Server(swallowed(), ok(CALL_BLOCK), ok("fine", tool_calls=[native_call()]), swallowed(), ok(CALL_BLOCK))
    client, _ = build(server)
    for _ in range(3):
        client.chat(USER, TOOLS)
    assert client.active_tool_mode == "text"


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


def test_a_repeat_that_is_empty_too_is_returned_empty_not_repeated_again():
    server = Server(swallowed(), swallowed(), ok(CALL_BLOCK))
    client, _ = build(server)
    response = client.chat(USER, TOOLS)
    assert len(server.requests) == 2 and response.tool_calls == [] and response.text == ""


def test_the_key_is_never_in_what_is_logged_about_a_swallowed_reply(caplog):
    from tests.test_llm_client import KEY

    server = Server(swallowed(), ok(CALL_BLOCK))
    client, _ = build(server)
    with caplog.at_level("DEBUG", logger="anvil.llm"):
        client.chat(USER, TOOLS)
    assert "text mode" in caplog.text and KEY not in caplog.text
