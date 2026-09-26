"""Endpoints that reject a parameter or a message shape: retry once without it, and remember for the rest of the run.

The 400 bodies are modelled on what OpenAI, vLLM (Qwen), DeepSeek and DashScope return.
"""

import json

import httpx
import pytest

from anvil.llm import LLMConfigError, LLMError
from tests.test_llm_client import KEY, TOOLS, USER, _env, build, ok  # noqa: F401 (_env: autouse)

OPENAI_PARALLEL = {
    "error": {
        "message": "Unsupported parameter: 'parallel_tool_calls' is not supported with this model.",
        "type": "invalid_request_error",
        "param": "parallel_tool_calls",
        "code": "unsupported_parameter",
    }
}
OPENAI_TEMPERATURE = {
    "error": {
        "message": "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
}
VLLM_TOP_P = {
    "object": "error",
    "message": "[{'type': 'extra_forbidden', 'loc': ('body', 'top_p'), 'msg': 'Extra inputs are not permitted', 'input': 0.8}]",
    "type": "BadRequestError",
    "param": None,
    "code": 400,
}
VLLM_TWO_FIELDS = {
    "object": "error",
    "message": (
        "[{'type': 'extra_forbidden', 'loc': ('body', 'top_p'), 'msg': 'Extra inputs are not permitted'}, "
        "{'type': 'extra_forbidden', 'loc': ('body', 'tool_choice'), 'msg': 'Extra inputs are not permitted'}]"
    ),
    "type": "BadRequestError",
    "code": 400,
}
DASHSCOPE_TOOL_CHOICE = {
    "error": {
        "message": "<400> InternalError.Algo.InvalidParameter: The parameter `tool_choice` is not supported for this model.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_parameter_error",
    }
}
DEEPSEEK_RESPONSE_FORMAT = {
    "error": {
        "message": "Invalid parameter: response_format type json_schema is not supported by this model",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_request_error",
    }
}
DASHSCOPE_THINKING = {
    "error": {
        "code": "invalid_parameter_error",
        "message": "parameter.enable_thinking must be set to false for non-streaming calls",
        "type": "invalid_request_error",
    }
}
OPENAI_MAX_COMPLETION = {
    "error": {
        "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
        "type": "invalid_request_error",
        "param": "max_tokens",
        "code": "unsupported_parameter",
    }
}
GENERIC_MAX_TOKENS = {"error": {"message": "max_tokens is not a valid field for this model", "type": "invalid_request_error"}}


class Picky:
    """A server that rejects a request whenever a forbidden field is present, with the given 400 body."""

    def __init__(self, forbidden: dict[str, dict], status: int = 400):
        self.forbidden, self.status = forbidden, status
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        for field, error in self.forbidden.items():
            if field in body:
                return httpx.Response(self.status, json=error)
        return ok("fine")


# ---- dropping a rejected parameter --------------------------------------------------------------


def test_a_rejected_temperature_is_dropped_and_the_request_retried_once():
    server = Picky({"temperature": OPENAI_TEMPERATURE})
    client, _ = build(server)
    assert client.chat(USER).text == "fine"
    assert [("temperature" in b) for b in server.bodies] == [True, False]


def test_the_decision_is_remembered_so_later_calls_do_not_repeat_the_failed_request():
    server = Picky({"temperature": OPENAI_TEMPERATURE})
    client, _ = build(server)
    client.chat(USER)
    client.chat(USER)
    client.chat(USER)
    assert len(server.bodies) == 4, "one wasted request in the whole run, then straight to the working shape"
    assert all("temperature" not in b for b in server.bodies[1:])


@pytest.mark.parametrize(
    "param, value, error",
    [
        ("top_p", 0.8, VLLM_TOP_P),
        ("tool_choice", "auto", DASHSCOPE_TOOL_CHOICE),
        ("parallel_tool_calls", False, OPENAI_PARALLEL),
        ("response_format", {"type": "json_object"}, DEEPSEEK_RESPONSE_FORMAT),
    ],
)
def test_each_named_parameter_is_dropped_when_the_endpoint_rejects_it(param, value, error):
    server = Picky({param: error})
    client, _ = build(server, llm_extra_params={param: value})
    assert client.chat(USER).text == "fine"
    assert param in server.bodies[0] and param not in server.bodies[1]
    assert server.bodies[1]["model"] == "test-model" and server.bodies[1]["messages"] == USER


def test_several_rejected_parameters_named_in_one_error_are_dropped_together():
    server = Picky({"top_p": VLLM_TWO_FIELDS, "tool_choice": VLLM_TWO_FIELDS})
    client, _ = build(server, llm_extra_params={"top_p": 0.8, "tool_choice": "auto", "presence_penalty": 0.1})
    client.chat(USER)
    assert len(server.bodies) == 2
    assert "top_p" not in server.bodies[1] and "tool_choice" not in server.bodies[1]
    assert server.bodies[1]["presence_penalty"] == 0.1, "parameters the endpoint did not complain about stay"


def test_a_422_validation_error_is_handled_like_a_400():
    server = Picky({"top_p": VLLM_TOP_P}, status=422)
    client, _ = build(server, llm_extra_params={"top_p": 0.8})
    assert client.chat(USER).text == "fine" and len(server.bodies) == 2


def test_the_parameter_is_dropped_in_text_mode_too_and_survives_the_switch_from_native():
    def handler(request):
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(400, json={"error": {"message": "This model does not support tools"}})
        if "top_p" in body:
            return httpx.Response(400, json=VLLM_TOP_P)
        return ok('```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```')

    bodies = []
    client, _ = build(lambda r: (bodies.append(json.loads(r.content)), handler(r))[1], llm_extra_params={"top_p": 0.8})
    first = client.chat(USER, TOOLS)
    assert first.tool_calls[0]["tool"] == "read_file" and client.active_tool_mode == "text"
    client.chat(USER, TOOLS)
    assert all("top_p" not in b for b in bodies[-2:]), "neither text-mode request carries the dropped parameter"


# ---- what must NOT be retried -----------------------------------------------------------------------


def test_an_error_that_names_a_parameter_we_never_sent_is_not_retried():
    server = Picky({"messages": VLLM_TOP_P})  # says 'top_p', but top_p is not in the request
    client, _ = build(server)
    with pytest.raises(LLMError) as info:
        client.chat(USER)
    assert info.value.status_code == 400 and len(server.bodies) == 1


def test_an_unrelated_400_is_not_retried():
    server = Picky({"messages": {"error": {"message": "model not found"}}})
    client, _ = build(server)
    with pytest.raises(LLMError):
        client.chat(USER)
    assert len(server.bodies) == 1


def test_a_parameter_name_inside_a_longer_identifier_does_not_count():
    """`tool_choice` must not be read as a complaint about `tools`, nor `top_p` inside `top_p_min`."""
    server = Picky({"messages": {"error": {"message": "unsupported field top_p_min"}}})
    client, _ = build(server, llm_extra_params={"top_p": 0.8})
    with pytest.raises(LLMError):
        client.chat(USER)
    assert len(server.bodies) == 1


def test_explicit_native_mode_still_refuses_an_endpoint_that_rejects_tools():
    """tool_mode: native means 'trust this endpoint'; only auto falls back to text."""
    server = Picky({"tools": {"error": {"message": "tools is not supported by this model"}}})
    client, _ = build(server, tool_mode="native")
    with pytest.raises(LLMError):
        client.chat(USER, TOOLS)
    assert len(server.bodies) == 1


def test_auth_and_server_errors_are_untouched():
    for status in (401, 403, 404):
        client, _ = build(Picky({"model": {"error": {"message": "temperature"}}}, status=status))
        with pytest.raises(LLMError) as info:
            client.chat(USER)
        assert info.value.status_code == status


def test_a_server_that_keeps_saying_400_is_given_up_on_after_a_bounded_number_of_tries():
    seen = []

    def always_reject(request):
        seen.append(1)
        return httpx.Response(400, json={"error": {"message": "temperature top_p tool_choice are unsupported"}})

    client, _ = build(always_reject, llm_extra_params={"top_p": 0.8, "tool_choice": "auto"})
    with pytest.raises(LLMError):
        client.chat(USER)
    assert len(seen) <= 3, "each parameter is dropped once; nothing loops"


# ---- fixes rather than drops ------------------------------------------------------------------------


def test_max_tokens_is_renamed_when_the_endpoint_wants_max_completion_tokens():
    server = Picky({"max_tokens": OPENAI_MAX_COMPLETION})
    client, _ = build(server, max_output_tokens=4096)
    client.chat(USER)
    assert server.bodies[0]["max_tokens"] == 4096
    assert server.bodies[1] == {**{k: v for k, v in server.bodies[0].items() if k != "max_tokens"}, "max_completion_tokens": 4096}
    client.chat(USER)
    assert "max_completion_tokens" in server.bodies[2] and "max_tokens" not in server.bodies[2]


def test_max_tokens_is_dropped_when_it_is_rejected_outright():
    server = Picky({"max_tokens": GENERIC_MAX_TOKENS})
    client, _ = build(server, max_output_tokens=4096)
    client.chat(USER)
    assert "max_tokens" not in server.bodies[1] and "max_completion_tokens" not in server.bodies[1]


def test_dashscope_qwen3_enable_thinking_is_set_to_false_and_remembered():
    def handler(request):
        body = json.loads(request.content)
        return ok("fine") if body.get("enable_thinking") is False else httpx.Response(400, json=DASHSCOPE_THINKING)

    bodies = []
    client, _ = build(lambda r: (bodies.append(json.loads(r.content)), handler(r))[1])
    assert client.chat(USER).text == "fine"
    client.chat(USER)
    assert "enable_thinking" not in bodies[0] and bodies[1]["enable_thinking"] is False and bodies[2]["enable_thinking"] is False
    assert len(bodies) == 3


# ---- message roles in native mode ----------------------------------------------------------------------


def test_a_role_complaint_turns_on_message_normalisation_for_native_mode_and_is_remembered():
    def handler(request):
        messages = json.loads(request.content)["messages"]
        turns = [m["role"] for m in messages if m["role"] != "system"]
        if any(a == b for a, b in zip(turns, turns[1:])):
            return httpx.Response(
                400, json={"object": "error", "message": "Conversation roles must alternate user/assistant/user/assistant/...", "code": 400}
            )
        return ok("fine")

    bodies = []
    client, _ = build(lambda r: (bodies.append(json.loads(r.content)), handler(r))[1], tool_mode="native")
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "brief"}, {"role": "user", "content": "kickoff"}]
    assert client.chat(history).text == "fine"
    client.chat(history)
    assert [len(b["messages"]) for b in bodies] == [3, 2, 2], "the second call is already in the accepted shape"
    assert bodies[1]["messages"][1]["content"] == "brief\n\nkickoff"


# ---- the record of what was worked around ------------------------------------------------------------------


def test_workarounds_are_recorded_with_the_status_and_the_providers_words():
    server = Picky({"temperature": OPENAI_TEMPERATURE})
    client, _ = build(server)
    client.chat(USER)
    client.chat(USER)
    assert len(client.workarounds) == 1
    note = client.workarounds[0]
    assert note.startswith("dropped parameter 'temperature' (HTTP 400: ") and "does not support 0" in note


def test_a_key_echoed_in_the_error_never_reaches_the_workaround_record():
    error = {"error": {"message": f"Unsupported parameter: 'temperature'. (request from key {KEY})"}}
    client, _ = build(Picky({"temperature": error}))
    client.chat(USER)
    assert client.workarounds and KEY not in " ".join(client.workarounds)


# ---- configuration ------------------------------------------------------------------------------------------


def test_extra_params_and_max_output_tokens_are_sent_as_configured():
    server = Picky({})
    client, _ = build(server, llm_extra_params={"top_p": 0.8, "enable_thinking": False}, max_output_tokens=2048)
    client.chat(USER)
    body = server.bodies[0]
    assert body["top_p"] == 0.8 and body["enable_thinking"] is False and body["max_tokens"] == 2048


def test_without_them_the_request_is_exactly_what_it_always_was():
    server = Picky({})
    client, _ = build(server)
    client.chat(USER)
    assert set(server.bodies[0]) == {"model", "messages", "temperature"}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"llm_extra_params": ["top_p"]}, "llm_extra_params"),
        ({"llm_extra_params": {"model": "x"}}, "may not set model"),
        ({"llm_extra_params": {"messages": []}}, "may not set messages"),
        ({"llm_extra_params": {"tools": []}}, "may not set tools"),
        ({"llm_extra_params": {1: 2}}, "llm_extra_params"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_output_tokens": -5}, "max_output_tokens"),
        ({"max_output_tokens": 10.5}, "max_output_tokens"),
        ({"max_output_tokens": True}, "max_output_tokens"),
        ({"max_output_tokens": "big"}, "max_output_tokens"),
    ],
)
def test_bad_extra_params_or_output_limits_are_rejected_when_the_client_is_built(overrides, message):
    with pytest.raises(LLMConfigError, match=message):
        build(Picky({}), **overrides)


def test_null_means_unset():
    client, _ = build(Picky({}), max_output_tokens=None, llm_extra_params=None)
    assert client is not None
