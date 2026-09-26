"""Message normalisation for endpoints that are stricter than OpenAI's (Mistral/Gemma/Qwen templates, some gateways)."""

import copy
import json

import httpx
import pytest

from anvil.llm.messages import EMPTY, normalize_messages
from tests.test_llm_client import CALL_BLOCK, TOOLS, Server, _env, build, native_call, ok  # noqa: F401 (_env: autouse)


def roles(messages):
    return [m["role"] for m in messages]


# ---- the agent's real history shape ------------------------------------------------------------


def agent_history():
    """What the orchestrator sends in its PATCH phase: three pinned user messages, then kickoff, then tool turns."""
    return [
        {"role": "system", "content": "You are ANVIL."},
        {"role": "user", "content": "# Issue: add() is wrong"},
        {"role": "user", "content": "## Repository overview\ncalc.py"},
        {"role": "user", "content": "[LOCALIZE summary]\ncalc.py:2"},
        {"role": "user", "content": "Begin phase PATCH."},
        {"role": "assistant", "content": "", "tool_calls": [native_call("edit_file", "{}", "c1"), native_call("git_diff", "{}", "c2")]},
        {"role": "tool", "tool_call_id": "c1", "content": "edited"},
        {"role": "tool", "tool_call_id": "c2", "content": "diff"},
    ]


def test_consecutive_user_messages_are_merged_in_order_without_losing_text():
    out = normalize_messages(agent_history()[:5])
    assert roles(out) == ["system", "user"]
    assert out[1]["content"] == "# Issue: add() is wrong\n\n## Repository overview\ncalc.py\n\n[LOCALIZE summary]\ncalc.py:2\n\nBegin phase PATCH."


def test_two_tool_results_in_a_row_become_one_user_message_in_text_mode():
    from anvil.llm.toolcalls import adapt_messages_for_text_mode

    out = normalize_messages(adapt_messages_for_text_mode(agent_history(), TOOLS))
    assert roles(out) == ["system", "user", "assistant", "user"]
    assert "[result of edit_file]\nedited" in out[3]["content"] and "[result of git_diff]\ndiff" in out[3]["content"]
    assert out[3]["content"].index("edited") < out[3]["content"].index("diff")


# ---- the rules ----------------------------------------------------------------------------------


def test_a_system_message_that_is_not_first_becomes_a_user_message():
    out = normalize_messages(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "u"}, {"role": "system", "content": "late note"}]
    )
    assert roles(out) == ["system", "user"] and out[1]["content"] == "u\n\nlate note"


def test_leading_system_messages_are_merged_into_one():
    out = normalize_messages(
        [{"role": "system", "content": "one"}, {"role": "system", "content": "two"}, {"role": "user", "content": "u"}]
    )
    assert out[0] == {"role": "system", "content": "one\n\ntwo"} and roles(out) == ["system", "user"]


def test_a_system_message_after_the_first_turn_with_no_leading_system_leaves_none_first():
    out = normalize_messages([{"role": "user", "content": "u"}, {"role": "system", "content": "s"}])
    assert roles(out) == ["user"] and "system" not in roles(out)


def test_consecutive_assistant_messages_are_merged():
    out = normalize_messages(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a1"}, {"role": "assistant", "content": "a2"}]
    )
    assert roles(out) == ["user", "assistant"] and out[1]["content"] == "a1\n\na2"


def test_a_conversation_that_starts_with_the_assistant_gets_a_user_opening():
    out = normalize_messages([{"role": "assistant", "content": "hello"}, {"role": "user", "content": "hi"}])
    assert roles(out) == ["user", "assistant", "user"]


def test_empty_messages_get_a_placeholder_and_empty_system_prompts_are_dropped():
    out = normalize_messages(
        [
            {"role": "system", "content": "  "},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": None},
        ]
    )
    assert roles(out) == ["user", "assistant", "user"]
    assert [m["content"] for m in out] == ["u", EMPTY, EMPTY]


def test_empty_pieces_do_not_leave_stray_blank_lines_when_merging():
    out = normalize_messages([{"role": "user", "content": "a"}, {"role": "user", "content": ""}, {"role": "user", "content": "b"}])
    assert out == [{"role": "user", "content": "a\n\nb"}]


def test_content_given_as_parts_is_flattened():
    out = normalize_messages([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert out == [{"role": "user", "content": "ab"}]


def test_a_leftover_tool_message_in_text_mode_becomes_a_user_message_and_tool_fields_are_dropped():
    out = normalize_messages(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "calling", "tool_calls": [native_call()]},
            {"role": "tool", "tool_call_id": "call_abc", "name": "read_file", "content": "print(1)"},
        ]
    )
    assert roles(out) == ["user", "assistant", "user"]
    assert out[2]["content"] == "[result of read_file]\nprint(1)"
    assert "tool_calls" not in json.dumps(out)


def test_the_input_is_never_mutated_and_the_result_is_stable():
    history = agent_history()
    before = copy.deepcopy(history)
    once = normalize_messages(history)
    assert history == before
    assert normalize_messages(once) == once


def test_an_already_valid_history_comes_back_equal():
    valid = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u2"},
    ]
    assert normalize_messages(valid) == valid


# ---- native mode keeps tool pairing ------------------------------------------------------------------


def test_native_mode_keeps_tool_calls_and_results_and_only_merges_around_them():
    out = normalize_messages(agent_history(), native=True)
    assert roles(out) == ["system", "user", "assistant", "tool", "tool"]
    assert out[2]["tool_calls"] == agent_history()[5]["tool_calls"]
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["c1", "c2"]
    assert out[1]["content"].startswith("# Issue") and out[1]["content"].endswith("Begin phase PATCH.")


def test_native_mode_does_not_merge_across_a_tool_call_message():
    history = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "thinking"},
        {"role": "assistant", "content": "", "tool_calls": [native_call()]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "x"},
        {"role": "user", "content": "next"},
        {"role": "user", "content": "more"},
    ]
    out = normalize_messages(history, native=True)
    assert roles(out) == ["user", "assistant", "assistant", "tool", "user"]
    assert out[-1]["content"] == "next\n\nmore"


# ---- through the client, against a strict server -----------------------------------------------------


def strict_server(seen: list):
    """A server that rejects what Mistral/Gemma/Qwen chat templates reject, with vLLM-style error text."""

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        seen.append(messages)
        for i, m in enumerate(messages):
            if m["role"] == "system" and i != 0:
                return httpx.Response(400, json={"object": "error", "message": "System message must be at the beginning.", "code": 400})
            if m["role"] not in ("system", "user", "assistant"):
                return httpx.Response(400, json={"object": "error", "message": f"Unexpected role '{m['role']}'", "code": 400})
            if not m.get("content"):
                return httpx.Response(400, json={"object": "error", "message": "Message content must not be empty.", "code": 400})
        turns = [m["role"] for m in messages if m["role"] != "system"]
        if turns[:1] != ["user"] or any(a == b for a, b in zip(turns, turns[1:])):
            return httpx.Response(
                400, json={"object": "error", "message": "Conversation roles must alternate user/assistant/user/assistant/...", "code": 400}
            )
        return ok(CALL_BLOCK)

    return handler


def test_text_mode_sends_a_history_that_a_strict_server_accepts():
    seen: list = []
    client, _ = build(strict_server(seen), tool_mode="text")
    response = client.chat(agent_history(), TOOLS)
    assert response.tool_calls[0]["tool"] == "read_file"
    assert roles(seen[0]) == ["system", "user", "assistant", "user"]


def test_text_mode_survives_empty_replies_and_late_system_messages_in_the_history():
    history = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Reply with a tool call."},
        {"role": "system", "content": "budget note"},
    ]
    seen: list = []
    client, _ = build(strict_server(seen), tool_mode="text")
    assert client.chat(history, TOOLS).tool_calls
    assert all(m["content"] for m in seen[0])


def test_the_same_history_is_rejected_by_that_server_without_normalisation():
    """Documents the failure being prevented: native mode passes the history through, and the strict server refuses it."""
    seen: list = []
    client, _ = build(strict_server(seen), tool_mode="native")
    with pytest.raises(Exception) as info:
        client.chat(agent_history(), TOOLS)
    assert "400" in str(info.value) or "roles" in str(info.value)
