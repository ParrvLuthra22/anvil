"""Tool-call dialects a Qwen / DeepSeek / Hermes-family model actually produces in text, and through the client.

Each sample is what a model emits, not what we would like it to emit: prose around the call, a closing tag the server
cut off, arguments as a JSON string, several calls at once.
"""

import json

import httpx
import pytest

from anvil.llm.toolcalls import parse_text_tool_call, parse_text_tool_calls
from tests.test_llm_client import CALL_BLOCK, TOOLS, USER, Server, _env, build, native_call  # noqa: F401 (_env: autouse)


def schema(name: str, **properties: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in properties.items()}},
        },
    }


ALL_TOOLS = [
    schema("read_file", path="string", start="integer", end="integer"),
    schema("grep", pattern="string", path="string"),
    schema("edit_file", path="string", old="string", new="string"),
    schema("git_diff"),
    schema("run_tests", target="string"),
]


def calls(text: str, tools=ALL_TOOLS) -> list[tuple[str, dict]]:
    return [(c["tool"], c["args"]) for c in parse_text_tool_calls(text, tools).calls]


# ---- our own fenced block ---------------------------------------------------------------------


def test_our_fenced_block_with_prose_around_it():
    reply = 'Let me read it.\n```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```\nThen I will edit.'
    parsed = parse_text_tool_calls(reply)
    assert parsed.calls == [{"tool": "read_file", "args": {"path": "a.py"}}]
    assert parsed.text == "Let me read it.\n\nThen I will edit."


# ---- Hermes / Qwen2.5 <tool_call> tags ------------------------------------------------------------

HERMES = """Let me start by reading the file.

<tool_call>
{"name": "read_file", "arguments": {"path": "src/calc.py", "start": 1, "end": 40}}
</tool_call>"""


def test_hermes_tag_with_prose_before_it():
    parsed = parse_text_tool_calls(HERMES)
    assert parsed.calls == [{"tool": "read_file", "args": {"path": "src/calc.py", "start": 1, "end": 40}}]
    assert parsed.text == "Let me start by reading the file."


def test_hermes_tag_with_prose_after_it():
    reply = HERMES + "\nI will report back once it has run."
    assert parse_text_tool_calls(reply).text == "Let me start by reading the file.\n\nI will report back once it has run."


def test_an_unclosed_tag_cut_off_at_the_end_of_the_reply():
    """The server dropped the stop token: <tool_call> opens, the JSON is complete, </tool_call> never comes."""
    reply = 'I will search first.\n<tool_call>\n{"name": "grep", "arguments": {"pattern": "def add", "path": "src"}}\n'
    parsed = parse_text_tool_calls(reply)
    assert parsed.calls == [{"tool": "grep", "args": {"pattern": "def add", "path": "src"}}]
    assert parsed.text == "I will search first."


def test_an_unclosed_tag_followed_by_more_prose_keeps_the_prose():
    reply = '<tool_call>{"name": "git_diff", "arguments": {}}\nThat shows what changed so far.'
    parsed = parse_text_tool_calls(reply)
    assert parsed.calls == [{"tool": "git_diff", "args": {}}]
    assert parsed.text == "That shows what changed so far."


def test_an_unclosed_tag_with_json_missing_its_closing_braces_is_repaired():
    reply = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"'
    assert calls(reply) == [("read_file", {"path": "a.py"})]


def test_a_tag_cut_off_inside_a_string_is_not_a_call_and_the_text_is_left_alone():
    reply = 'Editing now.\n<tool_call>{"name": "edit_file", "arguments": {"path": "a.py", "new": "def f():\n    ret'
    parsed = parse_text_tool_calls(reply)
    assert parsed.calls == [] and parsed.text == reply


def test_arguments_given_as_a_json_string_are_decoded():
    reply = '<tool_call>{"name": "grep", "arguments": "{\\"pattern\\": \\"def add\\"}"}</tool_call>'
    assert calls(reply) == [("grep", {"pattern": "def add"})]


def test_arguments_given_as_a_dict_are_used_as_they_are():
    reply = '<tool_call>{"name": "grep", "arguments": {"pattern": "def add"}}</tool_call>'
    assert calls(reply) == [("grep", {"pattern": "def add"})]


def test_a_tag_call_without_arguments_has_empty_args():
    assert calls("<tool_call>\n{\"name\": \"git_diff\"}\n</tool_call>") == [("git_diff", {})]


def test_tag_names_are_case_insensitive_and_the_function_call_variant_works():
    assert calls('<TOOL_CALL>{"name": "git_diff", "arguments": {}}</TOOL_CALL>') == [("git_diff", {})]
    assert calls('<function_call>{"name": "git_diff", "arguments": {}}</function_call>') == [("git_diff", {})]


def test_a_json_array_inside_one_tag_is_several_calls():
    reply = '<tool_call>[{"name": "git_diff", "arguments": {}}, {"name": "grep", "arguments": {"pattern": "x"}}]</tool_call>'
    assert calls(reply) == [("git_diff", {}), ("grep", {"pattern": "x"})]


def test_a_tag_with_a_code_fence_inside_is_handled():
    reply = '<tool_call>\n```json\n{"name": "git_diff", "arguments": {}}\n```\n</tool_call>'
    assert calls(reply) == [("git_diff", {})]


def test_a_tag_with_no_json_at_all_is_not_a_call():
    reply = "<tool_call>I want to read the file</tool_call>"
    assert parse_text_tool_calls(reply).calls == []


# ---- Qwen3-Coder XML -------------------------------------------------------------------------------

QWEN3_CODER = """I'll fix the operator.

<tool_call>
<function=edit_file>
<parameter=path>
calc.py
</parameter>
<parameter=old>
    return a - b
</parameter>
<parameter=new>
    return a + b
</parameter>
</function>
</tool_call>"""


def test_qwen3_coder_xml_function_and_parameters():
    parsed = parse_text_tool_calls(QWEN3_CODER, ALL_TOOLS)
    assert parsed.calls == [
        {"tool": "edit_file", "args": {"path": "calc.py", "old": "    return a - b", "new": "    return a + b"}}
    ]
    assert parsed.text == "I'll fix the operator."


def test_xml_parameters_are_typed_by_the_schema():
    reply = "<tool_call><function=read_file><parameter=path>a.py</parameter><parameter=start>1</parameter><parameter=end>40</parameter></function></tool_call>"
    assert calls(reply) == [("read_file", {"path": "a.py", "start": 1, "end": 40})]


def test_an_xml_parameter_declared_string_stays_a_string_even_if_it_looks_like_a_number():
    reply = "<tool_call><function=grep><parameter=pattern>404</parameter></function></tool_call>"
    assert calls(reply) == [("grep", {"pattern": "404"})]


def test_without_a_schema_xml_values_that_look_like_json_are_decoded():
    reply = "<tool_call><function=x><parameter=n>10</parameter><parameter=flag>true</parameter><parameter=s>hello</parameter></function></tool_call>"
    assert calls(reply, tools=None) == [("x", {"n": 10, "flag": True, "s": "hello"})]


def test_xml_call_cut_off_before_its_closing_tags():
    reply = "<tool_call>\n<function=grep>\n<parameter=pattern>\ndef add\n</parameter>\n"
    assert calls(reply) == [("grep", {"pattern": "def add"})]


def test_xml_multiline_values_keep_their_inner_newlines_and_indentation():
    reply = "<tool_call><function=edit_file><parameter=new>\ndef f():\n    return 1\n</parameter></function></tool_call>"
    assert calls(reply) == [("edit_file", {"new": "def f():\n    return 1"})]


# ---- bare JSON --------------------------------------------------------------------------------------


def test_a_bare_object_with_name_and_arguments():
    parsed = parse_text_tool_calls('{"name": "run_tests", "arguments": {"target": "tests/test_calc.py"}}', ALL_TOOLS)
    assert parsed.calls == [{"tool": "run_tests", "args": {"target": "tests/test_calc.py"}}] and parsed.text == ""


def test_a_bare_object_with_tool_and_args_in_the_middle_of_prose():
    reply = 'Sure, running it: {"tool": "run_tests", "args": {"target": "t.py"}} and then we will see.'
    parsed = parse_text_tool_calls(reply)
    assert parsed.calls == [{"tool": "run_tests", "args": {"target": "t.py"}}]
    assert parsed.text == "Sure, running it:\n\nand then we will see."


def test_llama_style_parameters_key_and_python_tag():
    assert calls('<|python_tag|>{"name": "read_file", "parameters": {"path": "a.py"}}') == [("read_file", {"path": "a.py"})]


def test_a_mistral_style_array_after_a_marker():
    reply = '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "a.py"}}]'
    assert calls(reply) == [("read_file", {"path": "a.py"})]


def test_arguments_before_the_name_still_work():
    assert calls('{"arguments": {"path": "a.py"}, "name": "read_file"}') == [("read_file", {"path": "a.py"})]


def test_a_bare_name_alone_is_a_call_only_when_it_is_a_known_tool():
    assert calls('{"name": "git_diff"}') == [("git_diff", {})]
    assert calls('{"name": "git_diff"}', tools=None) == []


def test_name_with_the_short_args_key_needs_a_known_tool():
    assert calls('{"name": "read_file", "args": {"path": "a.py"}}') == [("read_file", {"path": "a.py"})]
    assert calls('{"name": "pkg", "args": {"x": 1}}') == []


@pytest.mark.parametrize(
    "prose",
    [
        'The package.json says {"name": "my-app", "version": "1.0.0", "main": "index.js"}.',
        'A user record is {"name": "Alice", "arguments": "about the invoice"}.',
        'The config is {"name": "debug", "value": true}.',
        "Use the `name` field and the `arguments` list.",
        "All done. The bug was an off-by-one.",
    ],
)
def test_json_and_prose_that_only_look_a_little_like_a_call_are_not_calls(prose):
    parsed = parse_text_tool_calls(prose, ALL_TOOLS)
    assert parsed.calls == [] and parsed.text == prose


# ---- OpenAI-shaped JSON in the text -------------------------------------------------------------------


def test_openai_tool_calls_wrapper_with_string_arguments():
    reply = json.dumps(
        {"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]}
    )
    assert calls(reply) == [("read_file", {"path": "a.py"})]


def test_openai_function_object_with_dict_arguments():
    reply = '{"type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}'
    assert calls(reply) == [("read_file", {"path": "a.py"})]


def test_two_calls_in_an_openai_wrapper():
    wrapper = {
        "tool_calls": [
            {"function": {"name": "git_diff", "arguments": "{}"}},
            {"function": {"name": "grep", "arguments": {"pattern": "x"}}},
        ]
    }
    assert calls(json.dumps(wrapper)) == [("git_diff", {}), ("grep", {"pattern": "x"})]


# ---- several calls in one reply -----------------------------------------------------------------------


def test_several_tags_are_all_found_in_order_and_all_removed_from_the_text():
    reply = (
        "First I check the diff.\n"
        '<tool_call>{"name": "git_diff", "arguments": {}}</tool_call>\n'
        "Then I search.\n"
        '<tool_call>{"name": "grep", "arguments": {"pattern": "add"}}</tool_call>\n'
        "That is all."
    )
    parsed = parse_text_tool_calls(reply)
    assert [c["tool"] for c in parsed.calls] == ["git_diff", "grep"]
    assert parsed.text == "First I check the diff.\n\nThen I search.\n\nThat is all."


def test_a_tag_a_fence_and_a_bare_object_in_one_reply_come_out_in_text_order():
    reply = (
        '{"tool": "git_diff", "args": {}}\n'
        '<tool_call>{"name": "grep", "arguments": {"pattern": "a"}}</tool_call>\n'
        '```json\n{"tool": "read_file", "args": {"path": "b.py"}}\n```'
    )
    assert [c["tool"] for c in parse_text_tool_calls(reply, ALL_TOOLS).calls] == ["git_diff", "grep", "read_file"]


def test_the_single_call_function_takes_the_first_and_leaves_the_rest_in_the_text():
    reply = '<tool_call>{"name": "git_diff", "arguments": {}}</tool_call> and <tool_call>{"name": "grep", "arguments": {"pattern": "x"}}</tool_call>'
    call, rest = parse_text_tool_call(reply)
    assert call["tool"] == "git_diff" and "grep" in rest and "git_diff" not in rest


def test_a_call_inside_another_calls_arguments_is_not_a_second_call():
    reply = '<tool_call>{"name": "edit_file", "arguments": {"path": "x.py", "new": "{\\"name\\": \\"grep\\", \\"arguments\\": {}}"}}</tool_call>'
    assert [c["tool"] for c in parse_text_tool_calls(reply).calls] == ["edit_file"]


# ---- through the client: text mode, auto-mode rescue, native quirks --------------------------------------


def chat_body(message: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", **message}}]})


def test_text_mode_runs_a_qwen_tag_reply_end_to_end():
    client, _ = build(Server(chat_body({"content": HERMES.replace("src/calc.py", "a.py")})), tool_mode="text")
    response = client.chat(USER, ALL_TOOLS)
    assert response.tool_calls == [
        {"id": "call_1", "tool": "read_file", "args": {"path": "a.py", "start": 1, "end": 40}}
    ]
    assert response.text == "Let me start by reading the file."


def test_text_mode_takes_the_first_of_several_calls_and_reports_the_rest():
    reply = (
        '<tool_call>{"name": "git_diff", "arguments": {}}</tool_call>\n'
        '<tool_call>{"name": "grep", "arguments": {"pattern": "x"}}</tool_call>\n'
        '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    )
    client, _ = build(Server(chat_body({"content": reply})), tool_mode="text")
    response = client.chat(USER, ALL_TOOLS)
    assert [c["tool"] for c in response.tool_calls] == ["git_diff"]
    assert response.tool_calls[0]["ignored_calls"] == 2
    assert client.totals.ignored_calls == 2
    assert response.text == "", "the dropped calls do not linger in the text either"


def test_auto_mode_rescues_a_qwen_tag_reply_from_a_server_with_no_tool_parser():
    """vLLM without --enable-auto-tool-choice: `tools` is accepted and ignored, the call arrives as text."""
    client, _ = build(Server(chat_body({"content": HERMES.replace("src/calc.py", "a.py")})))
    response = client.chat(USER, ALL_TOOLS)
    assert response.tool_calls[0]["tool"] == "read_file" and response.tool_calls[0]["args"]["path"] == "a.py"
    assert client.active_tool_mode == "native", "one miss is not yet a reason to switch"


def test_auto_mode_switches_to_text_after_two_qwen_style_misses():
    client, _ = build(Server(chat_body({"content": HERMES})))
    client.chat(USER, ALL_TOOLS)
    client.chat(USER, ALL_TOOLS)
    assert client.active_tool_mode == "text"


def test_native_mode_still_does_not_read_tags_from_the_text():
    client, _ = build(Server(chat_body({"content": HERMES})), tool_mode="native")
    response = client.chat(USER, ALL_TOOLS)
    assert response.tool_calls == [] and "<tool_call>" in response.text


def test_the_legacy_function_call_field_is_a_tool_call():
    message = {"content": None, "function_call": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
    client, _ = build(Server(chat_body(message)), tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert [(c["tool"], c["args"]) for c in response.tool_calls] == [("read_file", {"path": "a.py"})]


@pytest.mark.parametrize(
    "arguments",
    ['{"path": "a.py"}', {"path": "a.py"}, '{"path": "a.py",}', "{'path': 'a.py'}"],
    ids=["json-string", "dict", "trailing-comma", "single-quotes"],
)
def test_native_arguments_as_a_json_string_or_already_a_dict(arguments):
    """DashScope, vLLM and DeepSeek send a string; some gateways send an object; small models send sloppy JSON."""
    client, _ = build(Server(chat_body({"content": "", "tool_calls": [native_call(arguments=arguments)]})), tool_mode="native")
    response = client.chat(USER, TOOLS)
    assert response.tool_calls[0]["args"] == {"path": "a.py"} and "error" not in response.tool_calls[0]


def test_native_arguments_that_are_a_list_are_flagged_not_silently_emptied():
    client, _ = build(Server(chat_body({"content": "", "tool_calls": [native_call(arguments=[1, 2])]})), tool_mode="native")
    call = client.chat(USER, TOOLS).tool_calls[0]
    assert call["args"] == {} and "not a valid JSON object" in call["error"]


def test_the_text_mode_call_syntax_the_client_asks_for_still_round_trips():
    client, _ = build(Server(chat_body({"content": CALL_BLOCK})), tool_mode="text")
    assert client.chat(USER, TOOLS).tool_calls[0]["args"] == {"path": "a.py"}
