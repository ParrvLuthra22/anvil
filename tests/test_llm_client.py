import json
import logging
import traceback
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from anvil.llm import LLMConfigError, LLMError, make_client
from anvil.llm.client import _retry_after_seconds

KEY = "sk-test-SECRET-abc123"
CONFIG = {
    "model": "test-model",
    "base_url": "https://llm.example.com/v1/",
    "temperature": 0,
    "tool_mode": "auto",
    "llm_max_attempts": 3,
    "llm_backoff_base_seconds": 1.0,
    "llm_backoff_max_seconds": 60,
}
USER = [{"role": "user", "content": "hello"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]
CALL_BLOCK = '```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```'


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", KEY)
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)


def completion(content="hi", tool_calls=None, usage=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body = {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
    body["usage"] = usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return body


def ok(*args, **kwargs) -> httpx.Response:
    return httpx.Response(200, json=completion(*args, **kwargs))


def native_call(name="read_file", arguments='{"path": "a.py"}', call_id="call_abc") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class Server:
    """Scripted transport handler: returns/raises items in order, repeating the last; records requests."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def body(self, index: int = -1) -> dict:
        return json.loads(self.requests[index].content)


def build(handler, **overrides):
    sleeps: list[float] = []
    client = make_client(
        {**CONFIG, **overrides}, transport=httpx.MockTransport(handler), sleep=sleeps.append, rng=lambda: 1.0
    )
    return client, sleeps


# ---- basic request / response ------------------------------------------------------------


def test_success_sends_expected_request_and_returns_response():
    server = Server(ok("hello there"))
    client, sleeps = build(server)
    response = client.chat(USER)

    assert response.text == "hello there"
    assert response.tool_calls == []
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "estimated": False}
    assert sleeps == []

    request = server.requests[0]
    assert str(request.url) == "https://llm.example.com/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert server.body() == {"model": "test-model", "messages": USER, "temperature": 0.0}
    assert "tools" not in server.body()


def test_env_overrides_model_and_base_url(monkeypatch):
    monkeypatch.setenv("AI_MODEL", "override-model")
    monkeypatch.setenv("AI_BASE_URL", "https://other.example.org/api")
    server = Server(ok())
    client, _ = build(server)
    client.chat(USER)
    assert str(server.requests[0].url) == "https://other.example.org/api/chat/completions"
    assert server.body()["model"] == "override-model"


def test_configured_timeouts_reach_the_request():
    server = Server(ok())
    client, _ = build(server, llm_timeout_seconds=42, llm_connect_timeout_seconds=3)
    client.chat(USER)
    timeout = server.requests[0].extensions["timeout"]
    assert timeout["read"] == 42.0 and timeout["connect"] == 3.0


# ---- retries -----------------------------------------------------------------------------


def test_429_with_retry_after_then_success():
    server = Server(httpx.Response(429, headers={"Retry-After": "7"}, text="slow down"), ok("done"))
    client, sleeps = build(server)
    assert client.chat(USER).text == "done"
    assert len(server.requests) == 2
    assert sleeps == [7.0]


def test_500_twice_then_success_uses_exponential_backoff():
    server = Server(httpx.Response(500), httpx.Response(502), ok("finally"))
    client, sleeps = build(server)
    assert client.chat(USER).text == "finally"
    assert len(server.requests) == 3
    assert sleeps == [1.0, 2.0]


def test_jitter_scales_delay_between_half_and_full():
    server = Server(httpx.Response(500), httpx.Response(500), ok())
    sleeps: list[float] = []
    client = make_client(
        CONFIG, transport=httpx.MockTransport(server), sleep=sleeps.append, rng=lambda: 0.0
    )
    client.chat(USER)
    assert sleeps == [0.5, 1.0]


def test_backoff_and_retry_after_are_capped():
    server = Server(httpx.Response(500), httpx.Response(429, headers={"Retry-After": "999"}), ok())
    client, sleeps = build(server, llm_backoff_max_seconds=5, llm_max_attempts=4)
    client.chat(USER)
    assert sleeps == [1.0, 5.0]


def test_permanent_server_error_raises_llm_error_after_max_attempts():
    server = Server(httpx.Response(503, text="overloaded"))
    client, sleeps = build(server)
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    err = info.value
    assert err.status_code == 503
    assert err.attempts == 3
    assert err.retryable is True
    assert "overloaded" in str(err)
    assert len(server.requests) == 3
    assert len(sleeps) == 2


def test_persistent_rate_limit_raises_llm_error():
    client, _ = build(Server(httpx.Response(429, headers={"Retry-After": "1"})))
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert info.value.status_code == 429


def test_client_error_is_not_retried():
    server = Server(httpx.Response(401, text="bad credentials"))
    client, sleeps = build(server)
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert info.value.status_code == 401
    assert info.value.retryable is False
    assert info.value.attempts == 1
    assert len(server.requests) == 1
    assert sleeps == []


def test_timeouts_and_connection_errors_are_retried():
    server = Server(httpx.ReadTimeout("timed out"), httpx.ConnectError("refused"), ok("back"))
    client, sleeps = build(server)
    assert client.chat(USER).text == "back"
    assert len(sleeps) == 2


def test_persistent_network_failure_raises_llm_error():
    client, _ = build(Server(httpx.ConnectTimeout("timed out")))
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert info.value.status_code is None
    assert info.value.retryable is True
    assert "ConnectTimeout" in str(info.value)


@pytest.mark.parametrize(
    "bad_first",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"error": {"code": 503, "message": "upstream down"}}),
        httpx.Response(200, json={"error": {"code": 429, "message": "slow"}}),
    ],
)
def test_transient_bad_bodies_are_retried(bad_first):
    client, sleeps = build(Server(bad_first, ok("recovered")))
    assert client.chat(USER).text == "recovered"
    assert len(sleeps) == 1


def test_error_payload_with_client_error_code_fails_immediately():
    server = Server(httpx.Response(200, json={"error": {"code": 400, "message": "bad model"}}))
    client, sleeps = build(server)
    with pytest.raises(LLMError, match="bad model"):
        client.chat(USER)
    assert len(server.requests) == 1 and sleeps == []


def test_retry_after_parsing():
    assert _retry_after_seconds("7") == 7.0
    assert _retry_after_seconds("1.5") == 1.5
    assert _retry_after_seconds(None) is None
    assert _retry_after_seconds("soon") is None
    assert _retry_after_seconds("-3") == 0.0
    future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    assert 25 <= _retry_after_seconds(future) <= 30
    past = format_datetime(datetime.now(timezone.utc) - timedelta(seconds=30), usegmt=True)
    assert _retry_after_seconds(past) == 0.0


# ---- usage accounting --------------------------------------------------------------------


def test_usage_is_estimated_from_chars_when_provider_omits_it():
    body = completion("b" * 40)
    del body["usage"]
    client, _ = build(Server(httpx.Response(200, json=body)))
    response = client.chat([{"role": "user", "content": "a" * 400}])
    assert response.usage == {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "estimated": True,
    }


def test_estimate_counts_tools_sent_and_native_call_arguments():
    body = completion(None, tool_calls=[native_call(arguments='{"path": "abcdefgh"}')])
    del body["usage"]
    server = Server(httpx.Response(200, json=body))
    client, _ = build(server)
    response = client.chat(USER, TOOLS)
    assert response.usage["prompt_tokens"] > len(json.dumps(TOOLS)) // 4
    assert response.usage["completion_tokens"] == (len("read_file") + len('{"path": "abcdefgh"}') + 3) // 4
    assert response.usage["estimated"] is True


def test_partial_usage_fills_the_gaps():
    usage = {"prompt_tokens": 7, "completion_tokens": 3}
    client, _ = build(Server(ok(usage=usage)))
    assert client.chat(USER).usage == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "estimated": False,
    }


def test_garbage_usage_values_are_treated_as_missing():
    client, _ = build(Server(ok("x" * 8, usage={"prompt_tokens": "lots", "completion_tokens": None, "total_tokens": True})))
    usage = client.chat([{"role": "user", "content": "y" * 8}]).usage
    assert usage["prompt_tokens"] == 2 and usage["completion_tokens"] == 2 and usage["estimated"] is True


def test_cumulative_totals_accumulate_and_ignore_failed_calls():
    server = Server(
        ok(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        httpx.Response(401),
        ok(usage={"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}),
    )
    client, _ = build(server)
    client.chat(USER)
    with pytest.raises(LLMError):
        client.chat(USER)
    client.chat(USER)
    totals = client.totals
    assert (totals.prompt_tokens, totals.completion_tokens, totals.total_tokens, totals.calls) == (30, 6, 36, 2)


# ---- native tool calling -----------------------------------------------------------------


def test_native_tool_call_is_parsed_and_tools_are_sent():
    server = Server(ok(None, tool_calls=[native_call()]))
    client, _ = build(server, tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert response.text == ""
    assert response.tool_calls == [{"id": "call_abc", "tool": "read_file", "args": {"path": "a.py"}}]
    assert server.body()["tools"] == TOOLS


def test_native_multiple_calls_dict_arguments_and_missing_ids():
    calls = [
        native_call("read_file", '{"path": "a.py"}', call_id=""),
        {"type": "function", "function": {"name": "git_diff", "arguments": ""}},
        {"id": "z", "type": "function", "function": {"name": "grep", "arguments": {"pattern": "x"}}},
    ]
    client, _ = build(Server(ok("Looking.", tool_calls=calls)), tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert response.text == "Looking."
    assert [c["tool"] for c in response.tool_calls] == ["read_file", "git_diff", "grep"]
    assert response.tool_calls[0]["id"] == "call_1" and response.tool_calls[1]["id"] == "call_2"
    assert response.tool_calls[1]["args"] == {}
    assert response.tool_calls[2] == {"id": "z", "tool": "grep", "args": {"pattern": "x"}}


def test_native_arguments_are_repaired_or_flagged():
    calls = [native_call(arguments="{'path': 'a.py',}"), native_call("grep", "not json", call_id="bad")]
    client, _ = build(Server(ok(None, tool_calls=calls)), tool_mode="native")
    first, second = client.chat(USER, TOOLS).tool_calls
    assert first["args"] == {"path": "a.py"} and "error" not in first
    assert second["args"] == {} and "not a valid JSON object" in second["error"]


def test_native_calls_without_a_name_are_dropped():
    client, _ = build(Server(ok(None, tool_calls=[{"id": "x", "function": {"arguments": "{}"}}, "junk"])), tool_mode="native")
    assert client.chat(USER, TOOLS).tool_calls == []


def test_bare_function_schemas_are_wrapped_for_the_provider():
    server = Server(ok())
    client, _ = build(server, tool_mode="native")
    client.chat(USER, [TOOLS[0]["function"]])
    assert server.body()["tools"] == TOOLS


def test_native_mode_passes_history_through_unchanged():
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [native_call()]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "print(1)"},
    ]
    server = Server(ok())
    client, _ = build(server, tool_mode="native")
    client.chat(history, TOOLS)
    assert server.body()["messages"] == history


def test_native_mode_does_not_look_for_calls_in_text_and_does_not_fall_back():
    server = Server(ok(CALL_BLOCK))
    client, _ = build(server, tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert response.tool_calls == [] and response.text == CALL_BLOCK

    rejecting = Server(httpx.Response(400, text="tools are not supported"))
    client, _ = build(rejecting, tool_mode="native")
    with pytest.raises(LLMError):
        client.chat(USER, TOOLS)
    assert len(rejecting.requests) == 1


# ---- text-mode tool calling ---------------------------------------------------------------


def test_text_mode_describes_tools_and_parses_prose_plus_json_block():
    reply = f"I'll read the file first.\n{CALL_BLOCK}\nThat should tell us more."
    server = Server(ok(reply))
    client, _ = build(server, tool_mode="text")
    response = client.chat([{"role": "system", "content": "Be careful."}, *USER], TOOLS)

    body = server.body()
    assert "tools" not in body
    assert body["messages"][0]["content"].startswith("Be careful.")
    assert "read_file: Read a file." in body["messages"][0]["content"]
    assert response.tool_calls == [{"id": "call_1", "tool": "read_file", "args": {"path": "a.py"}}]
    assert response.text == "I'll read the file first.\n\nThat should tell us more."


def test_text_mode_takes_first_of_several_blocks():
    second = CALL_BLOCK.replace("a.py", "b.py")
    client, _ = build(Server(ok(f"{CALL_BLOCK}\n{second}")), tool_mode="text")
    response = client.chat(USER, TOOLS)
    assert [c["args"]["path"] for c in response.tool_calls] == ["a.py"]


def test_text_mode_repairs_malformed_json():
    reply = '```json\n{"tool": "read_file", "args": {"path": "a.py",},}\n```'
    client, _ = build(Server(ok(reply)), tool_mode="text")
    assert client.chat(USER, TOOLS).tool_calls[0]["args"] == {"path": "a.py"}


def test_text_mode_unrepairable_json_returns_no_call_and_raw_text():
    reply = 'Trying:\n```json\n{"tool": "read_file", "args": {"path": }}\n```'
    client, _ = build(Server(ok(reply)), tool_mode="text")
    response = client.chat(USER, TOOLS)
    assert response.tool_calls == []
    assert response.text == reply


def test_text_mode_rewrites_tool_history_and_assigns_unique_ids():
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [native_call()]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "print(1)"},
    ]
    server = Server(ok(CALL_BLOCK), ok(CALL_BLOCK))
    client, _ = build(server, tool_mode="text")
    first = client.chat(history, TOOLS)
    second = client.chat(history, TOOLS)

    sent = server.body(0)["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert "tool_calls" not in json.dumps(sent)
    assert sent[3]["content"] == "[result of read_file]\nprint(1)"
    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"]


def test_text_mode_without_tools_does_not_parse_or_instruct():
    server = Server(ok(CALL_BLOCK))
    client, _ = build(server, tool_mode="text")
    response = client.chat(USER)
    assert response.tool_calls == [] and response.text == CALL_BLOCK
    assert server.body()["messages"] == USER


# ---- auto mode ----------------------------------------------------------------------------


def test_auto_falls_back_to_text_permanently_when_provider_rejects_tools():
    def handler(request):
        if "tools" in json.loads(request.content):
            return httpx.Response(400, json={"error": {"message": "This model does not support tools"}})
        return ok(CALL_BLOCK)

    requests: list[httpx.Request] = []

    def recording(request):
        requests.append(request)
        return handler(request)

    client, _ = build(recording)
    assert client.active_tool_mode == "native"
    first = client.chat(USER, TOOLS)
    assert first.tool_calls[0]["tool"] == "read_file"
    assert client.active_tool_mode == "text"
    assert len(requests) == 2  # rejected native attempt, then text mode

    client.chat(USER, TOOLS)
    assert len(requests) == 3  # straight to text mode now
    assert "tools" not in json.loads(requests[2].content)


def test_auto_unrelated_client_error_raises_and_stays_native():
    server = Server(httpx.Response(400, json={"error": {"message": "model not found"}}))
    client, _ = build(server)
    with pytest.raises(LLMError) as info:
        client.chat(USER, TOOLS)
    assert info.value.status_code == 400
    assert client.active_tool_mode == "native"


def test_auto_auth_failure_does_not_trigger_fallback():
    server = Server(httpx.Response(401, text="nope"))
    client, _ = build(server)
    with pytest.raises(LLMError):
        client.chat(USER, TOOLS)
    assert len(server.requests) == 1


def test_auto_switches_after_two_missed_native_calls_in_a_row_and_rescues_them():
    server = Server(ok(CALL_BLOCK), ok(CALL_BLOCK), ok(CALL_BLOCK))
    client, _ = build(server)

    first = client.chat(USER, TOOLS)
    assert first.tool_calls[0]["args"] == {"path": "a.py"} and first.text == ""
    assert client.active_tool_mode == "native"

    second = client.chat(USER, TOOLS)
    assert second.tool_calls and client.active_tool_mode == "text"

    client.chat(USER, TOOLS)
    assert "tools" in server.body(0) and "tools" in server.body(1)
    assert "tools" not in server.body(2)


def test_auto_a_working_native_call_resets_the_miss_counter():
    server = Server(ok(CALL_BLOCK), ok(None, tool_calls=[native_call()]), ok(CALL_BLOCK))
    client, _ = build(server)
    for _ in range(3):
        client.chat(USER, TOOLS)
    assert client.active_tool_mode == "native"


def test_auto_plain_answers_are_not_counted_as_attempts():
    client, _ = build(Server(ok("All done, the bug is fixed.")))
    for _ in range(4):
        response = client.chat(USER, TOOLS)
        assert response.tool_calls == []
    assert client.active_tool_mode == "native"


def test_auto_native_tool_calls_work_normally():
    client, _ = build(Server(ok(None, tool_calls=[native_call()])))
    response = client.chat(USER, TOOLS)
    assert response.tool_calls[0]["id"] == "call_abc"
    assert client.active_tool_mode == "native"


# ---- API key handling ----------------------------------------------------------------------


def test_missing_key_raises_clear_error_naming_the_variable(monkeypatch):
    monkeypatch.delenv("AI_API_KEY")
    with pytest.raises(LLMConfigError, match="AI_API_KEY") as info:
        make_client(CONFIG)
    assert isinstance(info.value, LLMError)


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_blank_key_raises(monkeypatch, value):
    monkeypatch.setenv("AI_API_KEY", value)
    with pytest.raises(LLMConfigError, match="AI_API_KEY"):
        make_client(CONFIG)


@pytest.mark.parametrize("value", ["sk-abc def", "sk-abc\ndef", "sk-“quoted”"])
def test_malformed_key_raises_without_echoing_it(monkeypatch, value):
    monkeypatch.setenv("AI_API_KEY", value)
    with pytest.raises(LLMConfigError) as info:
        make_client(CONFIG)
    assert value.strip() not in str(info.value)


def test_surrounding_whitespace_on_the_key_is_stripped(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", f"{KEY}\n")
    server = Server(ok())
    client, _ = build(server)
    client.chat(USER)
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


def assert_no_key(err: BaseException) -> None:
    rendered = "".join(traceback.format_exception(err))
    for text in (str(err), repr(err), getattr(err, "detail", ""), rendered):
        assert KEY not in text


def test_key_echoed_by_the_provider_is_scrubbed_from_errors():
    body = {"error": {"message": f"Incorrect API key provided: {KEY}."}}
    client, _ = build(Server(httpx.Response(401, json=body)))
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert_no_key(info.value)
    assert "[REDACTED]" in str(info.value)


def test_key_inside_a_network_exception_message_is_scrubbed():
    client, _ = build(Server(httpx.ConnectError(f"could not connect using {KEY}")))
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert_no_key(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_non_transport_request_error_is_not_retried_and_scrubbed():
    server = Server(httpx.DecodingError(f"cannot decode response for {KEY}"))
    client, sleeps = build(server)
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert_no_key(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert info.value.retryable is False
    assert len(server.requests) == 1 and sleeps == []


def test_key_inside_a_200_error_payload_is_scrubbed():
    body = {"error": {"code": 400, "message": f"invalid key {KEY}"}}
    client, _ = build(Server(httpx.Response(200, json=body)))
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert_no_key(info.value)


def test_key_never_reaches_repr_or_logs(caplog):
    server = Server(httpx.Response(500, text=f"boom {KEY}"), ok())
    client, _ = build(server)
    with caplog.at_level(logging.DEBUG):
        client.chat(USER)
    assert KEY not in caplog.text
    assert KEY not in repr(client)
    assert KEY not in repr(client._config)


# ---- construction -------------------------------------------------------------------------


def test_invalid_config_raises_llm_config_error():
    with pytest.raises(LLMConfigError):
        make_client({**CONFIG, "tool_mode": "psychic"})


def test_make_client_accepts_llm_config_object_and_close_is_safe():
    from anvil.llm import LLMConfig

    client = make_client(LLMConfig.from_mapping(CONFIG))
    assert "test-model" in repr(client)
    client.close()
