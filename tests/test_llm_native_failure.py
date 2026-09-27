"""tool_mode "auto": the first native tool-call failure moves the run to text mode for good; native is never tried again.

Seen with Qwen3-Coder behind OpenRouter: the provider's own tool-call parser swallowed replies (HTTP 200, tokens billed, nothing
returned) or the call arrived as text. Each retry of native costs a billed call and a repeat in text mode, so one failure is
enough to stop using it.
"""

from __future__ import annotations

import json

import httpx

from anvil.llm import make_client
from tests.test_llm_client import CALL_BLOCK, CONFIG, TOOLS, USER, _env, native_call  # noqa: F401 (_env: autouse)

NATIVE_OK = {"role": "assistant", "content": None, "tool_calls": [native_call()]}
SWALLOWED = {"role": "assistant", "content": None, "refusal": None, "reasoning": None}
CALL_AS_TEXT = {"role": "assistant", "content": CALL_BLOCK}


def completion(message: dict) -> httpx.Response:
    body = {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}
    return httpx.Response(200, json=body)


class Endpoint:
    """A provider that behaves as scripted for requests WITH ``tools`` (native) and always reads text-mode requests."""

    def __init__(self, *native_behaviour):
        self.native_behaviour = list(native_behaviour)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if "tools" in body:
            step = self.native_behaviour.pop(0) if len(self.native_behaviour) > 1 else self.native_behaviour[0]
            return completion(step)
        return completion({"role": "assistant", "content": CALL_BLOCK})  # text mode: the call is in the text

    @property
    def native_requests(self) -> int:
        return sum(1 for body in self.requests if "tools" in body)


def build(endpoint: Endpoint, **overrides):
    return make_client({**CONFIG, **overrides}, transport=httpx.MockTransport(endpoint), sleep=lambda s: None, rng=lambda: 1.0)


# ---- one swallowed native reply -------------------------------------------------------------------------------------


def test_the_first_swallowed_native_reply_switches_the_rest_of_the_run_to_text_mode():
    endpoint = Endpoint(SWALLOWED)
    client = build(endpoint)

    first = client.chat(USER, TOOLS)
    assert first.tool_calls and first.tool_calls[0]["tool"] == "read_file", "the swallowed reply was repeated in text mode"
    assert client.active_tool_mode == "text", "and the client does not go back"
    for _ in range(4):
        assert client.chat(USER, TOOLS).tool_calls
    assert endpoint.native_requests == 1, "native was tried once in the whole run"
    assert [("tools" in body) for body in endpoint.requests] == [True, False, False, False, False, False]


def test_native_is_not_retried_even_when_the_endpoint_would_answer_it_properly_later():
    endpoint = Endpoint(SWALLOWED, NATIVE_OK)  # the second native request would have worked
    client = build(endpoint)
    for _ in range(6):
        client.chat(USER, TOOLS)
    assert endpoint.native_requests == 1 and client.active_tool_mode == "text"


# ---- one call written as text ---------------------------------------------------------------------------------------


def test_the_first_call_written_as_text_instead_of_tool_calls_switches_to_text_mode_too():
    endpoint = Endpoint(CALL_AS_TEXT)
    client = build(endpoint)

    first = client.chat(USER, TOOLS)
    assert first.tool_calls[0]["tool"] == "read_file", "it is rescued from the text"
    assert client.active_tool_mode == "text"
    for _ in range(3):
        client.chat(USER, TOOLS)
    assert endpoint.native_requests == 1


# ---- what must not change --------------------------------------------------------------------------------------------


def test_a_healthy_native_endpoint_stays_native_for_the_whole_run():
    endpoint = Endpoint(NATIVE_OK)
    client = build(endpoint)
    for _ in range(5):
        assert client.chat(USER, TOOLS).tool_calls
    assert client.active_tool_mode == "native" and endpoint.native_requests == 5


def test_a_reply_that_is_only_prose_is_an_answer_not_a_native_failure():
    endpoint = Endpoint({"role": "assistant", "content": "I have looked at it and it is fine."})
    client = build(endpoint)
    assert client.chat(USER, TOOLS).text.startswith("I have looked")
    assert client.active_tool_mode == "native"


def test_an_empty_reply_without_tools_is_not_a_native_failure():
    endpoint = Endpoint(SWALLOWED)
    client = build(endpoint)
    client.chat(USER, None)
    assert client.active_tool_mode == "native"


def test_explicit_native_mode_never_switches_however_often_it_fails():
    endpoint = Endpoint(SWALLOWED)
    client = build(endpoint, tool_mode="native")
    for _ in range(3):
        assert client.chat(USER, TOOLS).tool_calls == []
    assert client.active_tool_mode == "native" and endpoint.native_requests == 3


def test_explicit_text_mode_never_sends_tools():
    endpoint = Endpoint(NATIVE_OK)
    client = build(endpoint, tool_mode="text")
    for _ in range(3):
        assert client.chat(USER, TOOLS).tool_calls
    assert endpoint.native_requests == 0


def test_the_switch_belongs_to_the_run_a_new_client_starts_native_again():
    endpoint = Endpoint(SWALLOWED)
    build(endpoint).chat(USER, TOOLS)
    fresh = build(Endpoint(NATIVE_OK))
    assert fresh.active_tool_mode == "native"
    fresh.chat(USER, TOOLS)
    assert fresh.active_tool_mode == "native"


def test_the_wasted_native_reply_is_still_counted_in_the_usage_the_budget_sees():
    endpoint = Endpoint(SWALLOWED)
    client = build(endpoint)
    response = client.chat(USER, TOOLS)
    assert response.usage["completion_tokens"] == 20 and response.usage["prompt_tokens"] == 200, "the swallowed call and its text-mode repeat"
    later = client.chat(USER, TOOLS)
    assert later.usage["completion_tokens"] == 10, "afterwards a call costs one call"


# ---- an endpoint that rejects tools outright ------------------------------------------------------------------------


def test_an_http_400_on_tools_moves_to_text_mode_for_good_as_before():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls["n"] += 1
        if "tools" in body:
            return httpx.Response(400, json={"error": {"message": "tools are not supported"}})
        return completion({"role": "assistant", "content": CALL_BLOCK})

    client = make_client(CONFIG, transport=httpx.MockTransport(handler), sleep=lambda s: None, rng=lambda: 1.0)
    client.chat(USER, TOOLS)
    assert client.active_tool_mode == "text"
    before = calls["n"]
    client.chat(USER, TOOLS)
    assert calls["n"] == before + 1, "one request, not a native attempt first"


# ---- the shipped default -------------------------------------------------------------------------------------------------


def test_the_thresholds_are_one():
    from anvil.llm import client as module

    assert module._SWALLOWED_BEFORE_TEXT_MODE == 1 and module._MISSES_BEFORE_TEXT_MODE == 1
