"""Reasoning output (DeepSeek-R1, Qwen QwQ/Qwen3): removed from the text, kept out of the history, still counted.

The response bodies follow what these providers actually return: ``reasoning_content`` on the official DeepSeek API and
on vLLM/DashScope, ``<think>`` blocks in ``content`` (Qwen3 on vLLM/Ollama), the closing tag alone when the chat template
opened the block, and ``reasoning`` on OpenRouter.
"""

import httpx
import pytest

from anvil.llm import LLMConfigError
from anvil.llm.reasoning import content_text, split_message, split_reasoning
from tests.test_llm_client import CALL_BLOCK, TOOLS, USER, Server, _env, build, native_call  # noqa: F401 (_env: autouse)

# ---- recorded-style provider bodies -----------------------------------------------------------

DEEPSEEK_REASONER = {
    "id": "0f6c1d2e-9c1a-4c3e-8d0b-1f2e3a4b5c6d",
    "object": "chat.completion",
    "model": "deepseek-reasoner",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The bug is in add(): it subtracts.",
                "reasoning_content": "Okay, let me read the function. It returns a - b, but the name says add, so...",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 24,
        "completion_tokens": 312,
        "total_tokens": 336,
        "prompt_cache_hit_tokens": 0,
        "completion_tokens_details": {"reasoning_tokens": 298},
    },
}

QWEN3_VLLM_THINK = (
    "<think>\nOkay, the user wants me to read a.py. I should call the read_file tool. Something like "
    '{"tool": "read_file", "args": {"path": "wrong.py"}} but the path is a.py, so let me fix that.\n</think>\n\n'
    + CALL_BLOCK
)

QWQ_CLOSING_TAG_ONLY = (
    "Alright, so the function subtracts instead of adding. I should say so plainly.\n</think>\n\n"
    "add() subtracts its arguments; it should add them."
)


def body(message: dict, usage: dict | None = None) -> httpx.Response:
    payload = {"choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}]}
    if usage is not None:
        payload["usage"] = usage
    return httpx.Response(200, json=payload)


# ---- the field ------------------------------------------------------------------------------


def test_a_reasoning_content_field_never_reaches_the_text_and_its_tokens_are_reported():
    client, _ = build(Server(httpx.Response(200, json=DEEPSEEK_REASONER)))
    response = client.chat(USER)

    assert response.text == "The bug is in add(): it subtracts."
    assert "Okay, let me read" not in response.text
    assert response.usage["reasoning_tokens"] == 298 and response.usage["completion_tokens"] == 312
    assert client.totals.reasoning_tokens == 298 and client.totals.reasoning_replies == 1
    assert client.reasoning_seen is True


def test_the_openrouter_reasoning_field_is_treated_the_same():
    client, _ = build(Server(body({"content": "Done.", "reasoning": "Let me think about it first."})))
    response = client.chat(USER)
    assert response.text == "Done." and client.reasoning_seen


def test_an_empty_reasoning_field_is_not_reasoning():
    client, _ = build(Server(body({"content": "Done.", "reasoning_content": "  ", "reasoning": None})))
    assert client.chat(USER).text == "Done." and client.reasoning_seen is False


def test_reasoning_tokens_are_omitted_from_usage_when_the_provider_does_not_report_them():
    usage = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    client, _ = build(Server(body({"content": "hi", "reasoning_content": "hmm"}, usage)))
    assert "reasoning_tokens" not in client.chat(USER).usage


# ---- tags in the content ----------------------------------------------------------------------


def test_think_blocks_are_removed_from_the_text():
    client, _ = build(Server(body({"content": "<think>\nplanning...\n</think>\n\nThe answer is 5."})))
    response = client.chat(USER)
    assert response.text == "The answer is 5." and client.reasoning_seen


def test_the_closing_tag_alone_means_everything_before_it_was_reasoning():
    """QwQ and Qwen3 templates open <think> in the prompt, so the reply starts mid-thought."""
    client, _ = build(Server(body({"content": QWQ_CLOSING_TAG_ONLY})))
    assert client.chat(USER).text == "add() subtracts its arguments; it should add them."


def test_a_reply_cut_off_while_thinking_has_no_answer_at_all():
    """finish_reason=length inside <think>: nothing after the tag is an answer, and it must not be parsed as one."""
    cut = 'Let me check.\n<think>\nI could call {"tool": "read_file", "args": {"path": "a.py"}} but first I need to'
    client, _ = build(Server(body({"content": cut})), tool_mode="text")
    response = client.chat(USER, TOOLS)
    assert response.text == "Let me check." and response.tool_calls == []


def test_several_blocks_and_the_thinking_tag_variant_are_all_removed():
    text = "<think>a</think>One. <thinking>b</thinking>Two. <THINK>c</THINK>Three."
    assert split_reasoning(text).text == "One. Two. Three."
    assert split_reasoning(text).reasoning == "a\n\nb\n\nc"


def test_an_empty_think_block_still_counts_as_reasoning_having_been_present():
    reply = split_reasoning("<think></think>\nAnswer.")
    assert reply.text == "Answer." and reply.found and reply.reasoning == ""


def test_text_without_tags_is_returned_exactly_as_it_came():
    text = "  I think the bug is here.\nLet me think about it.  "
    reply = split_reasoning(text)
    assert reply.text == text and not reply.found
    client, _ = build(Server(body({"content": text})))
    assert client.chat(USER).text == text and client.reasoning_seen is False


def test_content_given_as_a_list_of_parts_is_joined():
    assert content_text([{"type": "text", "text": "a"}, {"type": "image_url"}, "b", {"text": "c"}]) == "abc"
    assert content_text(None) == "" and content_text(5) == ""


def test_the_field_and_the_tags_are_both_collected():
    reply = split_message({"content": "<think>inline</think>Answer", "reasoning_content": "field"})
    assert reply.text == "Answer" and reply.reasoning == "field\n\ninline"


# ---- reasoning must not leak into tool-call parsing or the history ---------------------------


def test_a_call_quoted_inside_the_reasoning_is_not_taken_for_the_models_call():
    client, _ = build(Server(body({"content": QWEN3_VLLM_THINK})), tool_mode="text")
    response = client.chat(USER, TOOLS)
    assert [c["args"] for c in response.tool_calls] == [{"path": "a.py"}], "the real call, not the one in the thinking"
    assert response.text == ""


def test_a_native_reply_with_thinking_and_tool_calls_keeps_the_calls_and_drops_the_thinking():
    message = {"content": "<think>I need to read it.</think>", "tool_calls": [native_call()]}
    client, _ = build(Server(body(message)), tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert response.text == "" and response.tool_calls[0]["tool"] == "read_file"


def test_auto_mode_rescues_a_call_that_follows_the_thinking():
    client, _ = build(Server(body({"content": QWEN3_VLLM_THINK})))
    response = client.chat(USER, TOOLS)
    assert [c["args"] for c in response.tool_calls] == [{"path": "a.py"}]


def test_the_text_that_goes_into_the_history_is_free_of_reasoning():
    """The agent appends response.text to its history verbatim; a reasoning trace there would eat the context."""
    client, _ = build(Server(httpx.Response(200, json=DEEPSEEK_REASONER)))
    assert "Okay, let me read" not in client.chat(USER).text


# ---- usage --------------------------------------------------------------------------------------


def test_estimated_usage_counts_the_reasoning_the_provider_generated():
    long_thought = "x" * 4000
    plain, _ = build(Server(body({"content": "Done."})))
    thinking, _ = build(Server(body({"content": "Done.", "reasoning_content": long_thought})))
    with_reasoning = thinking.chat(USER).usage
    without = plain.chat(USER).usage
    assert with_reasoning["estimated"] is True
    assert with_reasoning["completion_tokens"] - without["completion_tokens"] == 1000


def test_estimated_usage_also_counts_think_tags_inside_the_content():
    plain, _ = build(Server(body({"content": "Done."})))
    tagged, _ = build(Server(body({"content": "<think>" + "y" * 400 + "</think>Done."})))
    assert tagged.chat(USER).usage["completion_tokens"] > plain.chat(USER).usage["completion_tokens"] + 90


def test_totals_accumulate_reasoning_over_calls():
    client, _ = build(Server(httpx.Response(200, json=DEEPSEEK_REASONER)))
    client.chat(USER)
    client.chat(USER)
    assert client.totals.reasoning_tokens == 596 and client.totals.reasoning_replies == 2 and client.totals.calls == 2


# ---- switching it off ---------------------------------------------------------------------------


def test_strip_reasoning_false_leaves_tags_in_the_text_but_never_the_field():
    message = {"content": "<think>keep me</think>Answer", "reasoning_content": "separate field"}
    client, _ = build(Server(body(message)), strip_reasoning=False)
    response = client.chat(USER)
    assert response.text == "<think>keep me</think>Answer"


@pytest.mark.parametrize("value", ["false", 0, None, "no"])
def test_strip_reasoning_must_be_a_real_boolean(value):
    with pytest.raises(LLMConfigError, match="strip_reasoning"):
        build(Server(body({"content": "x"})), strip_reasoning=value)
