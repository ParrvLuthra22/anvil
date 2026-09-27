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


def test_xml_strips_exactly_the_templates_own_newline_so_a_files_final_newline_survives():
    # the template writes "\n" + value + "\n": a value that itself ends in "\n" arrives with a blank line before the closer
    reply = "<tool_call><function=edit_file><parameter=new>\ndef f():\n    return 1\n\n</parameter></function></tool_call>"
    assert calls(reply) == [("edit_file", {"new": "def f():\n    return 1\n"})]


def test_xml_value_of_only_newlines_is_not_eaten_whole():
    reply = "<tool_call><function=edit_file><parameter=new>\n\n\n</parameter></function></tool_call>"
    assert calls(reply) == [("edit_file", {"new": "\n"})]


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
    assert client.active_tool_mode == "text", "the first miss switches the run to text mode for good"


def test_auto_mode_switches_to_text_at_the_first_qwen_style_miss():
    client, _ = build(Server(chat_body({"content": HERMES})))
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


# ---- Python call syntax: tool_name(key="value") ---------------------------------------------------------------
# Seen from Qwen3-Coder on a real endpoint, several times in one run: in VERIFY the model closed with
# phase_done(summary="...") plus a stray closing fence, in plain text, and the phase stalled with a passing repro.

PHASE_TOOLS = ALL_TOOLS + [schema("phase_done", summary="string"), schema("give_up", reason="string")]

REAL_SAMPLE = (
    'phase_done(summary="The patch has been verified and works correctly. The issue was that the add() function in '
    "calc.py was implementing subtraction (a - b) instead of addition (a + b). All tests pass, including the specific "
    'test for add(2, 3) which now correctly returns 5 instead of -1.")\n```'
)


def fcalls(text: str) -> list[tuple[str, dict]]:
    return calls(text, PHASE_TOOLS)


def test_the_call_a_real_model_wrote_in_python_syntax_is_a_call_and_the_stray_fence_goes_with_it():
    parsed = parse_text_tool_calls(REAL_SAMPLE, PHASE_TOOLS)
    assert [c["tool"] for c in parsed.calls] == ["phase_done"]
    assert parsed.calls[0]["args"]["summary"].startswith("The patch has been verified") and "returns 5" in parsed.calls[0]["args"]["summary"]
    assert parsed.text == "", "no orphan ``` left behind as the reply's text"


def test_prose_before_the_call_is_kept():
    parsed = parse_text_tool_calls("The fix works.\n\nphase_done(summary='changed - to +')", PHASE_TOOLS)
    assert parsed.calls == [{"tool": "phase_done", "args": {"summary": "changed - to +"}}]
    assert parsed.text == "The fix works."


def test_keyword_arguments_keep_their_python_types():
    assert fcalls('read_file(path="a.py", start=1, end=40)') == [("read_file", {"path": "a.py", "start": 1, "end": 40})]
    assert fcalls("run_tests(target=None)") == [("run_tests", {"target": None})]
    assert fcalls("edit_file(path='a.py', old='x', new='')") == [("edit_file", {"path": "a.py", "old": "x", "new": ""})]


def test_positional_arguments_follow_the_order_the_schema_declares():
    assert fcalls('read_file("a.py", 1, 40)') == [("read_file", {"path": "a.py", "start": 1, "end": 40})]
    assert fcalls("git_diff()") == [("git_diff", {})]


def test_a_call_inside_a_python_fence_is_a_call_and_the_fence_goes_with_it():
    parsed = parse_text_tool_calls('Checking:\n```python\nread_file(path="a.py")\n```', PHASE_TOOLS)
    assert parsed.calls == [{"tool": "read_file", "args": {"path": "a.py"}}]
    assert parsed.text == "Checking:"


def test_parentheses_and_quotes_inside_a_string_do_not_end_the_call_early():
    reply = "grep(pattern=\"def add(a, b):\", path='src')"
    assert fcalls(reply) == [("grep", {"pattern": "def add(a, b):", "path": "src"})]
    assert fcalls('edit_file(path="a.py", old="it\'s (a)", new="it\\"s [b]")')[0][1]["old"] == "it's (a)"


def test_a_raw_newline_inside_a_quoted_string_is_read_as_a_newline():
    reply = 'edit_file(path="a.py", old="def f():\n    return 1", new="def f():\n    return 2")'
    assert fcalls(reply) == [("edit_file", {"path": "a.py", "old": "def f():\n    return 1", "new": "def f():\n    return 2"})]


def test_triple_quoted_strings_work():
    reply = 'phase_done(summary="""Fixed add().\nIt subtracted; it now adds.""")'
    assert fcalls(reply) == [("phase_done", {"summary": "Fixed add().\nIt subtracted; it now adds."})]


def test_a_call_may_be_indented_or_marked_as_a_list_item():
    assert fcalls("  - read_file(path='a.py')") == [("read_file", {"path": "a.py"})]
    assert fcalls("> `git_diff()`") == [("git_diff", {})]


@pytest.mark.parametrize(
    "reply",
    [
        "Then call phase_done(summary) when you are finished.",  # mentioned mid-sentence, and not literals
        "phase_done(summary)",  # a name, not a value
        "You could run_tests() to check.",  # mid-line
        'frobnicate(x="1")',  # not one of the tools
        'read_file(path="a.py") and then I will read more of it.',  # followed by prose: not the last thing
        'phase_done(summary="cut off in the middle of a sentence',  # never completed by guesswork
        'read_file(path=os.path.join("a", "b.py"))',  # a call in the argument, not a literal
        'read_file(path=f"{name}.py")',  # an f-string
        'read_file(**{"path": "a.py"})',  # unpacking
        'read_file(*["a.py"])',
        'read_file("a.py", 1, 40, 99)',  # more positionals than the schema has parameters
        "read_file(path='a.py', path='b.py')",  # a keyword given twice: which value was meant?
        "read_file('a.py', path='b.py')",  # the same parameter positionally and by name
    ],
)
def test_things_that_only_look_like_calls_are_not_run(reply):
    assert fcalls(reply) == []


@pytest.mark.parametrize(
    "reply",
    [
        "I will now call read_file(path='a.py')",
        "The call that failed was git_diff()",
        "Next: run_tests(target='tests/test_calc.py')",
    ],
)
def test_a_call_in_the_middle_of_a_line_is_talk_about_a_call_even_when_it_ends_the_reply(reply):
    assert fcalls(reply) == []


def test_of_several_calls_written_in_python_syntax_only_the_last_can_run():
    """The protocol is one call per reply and nothing after it: an earlier one is followed by more reply."""
    assert fcalls('read_file(path="a.py")\nread_file(path="b.py")') == [("read_file", {"path": "b.py"})]


def test_without_the_tool_list_python_syntax_is_never_guessed():
    assert calls('phase_done(summary="x")', tools=None) == []
    assert calls('phase_done(summary="x")', tools=ALL_TOOLS) == [], "phase_done is not one of these tools"


def test_a_json_call_wins_over_python_syntax_in_the_same_reply():
    reply = '```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```\nphase_done(summary="x")'
    assert fcalls(reply) == [("read_file", {"path": "a.py"})]


def test_dangerous_looking_arguments_are_data_never_evaluated():
    """ast.literal_eval only: nothing in the reply is executed."""
    assert fcalls('run_tests(target=__import__("os").system("echo hi"))') == []
    assert fcalls("phase_done(summary=[x for x in range(10**9)])") == []
    assert fcalls('phase_done(summary="__import__(\'os\').system(\'echo hi\')")') == [
        ("phase_done", {"summary": "__import__('os').system('echo hi')"})
    ]


def test_the_client_rescues_a_python_syntax_call_from_a_native_reply_and_counts_it_as_a_miss():
    from tests.test_llm_client import ok

    tools = PHASE_TOOLS
    server = Server(ok('phase_done(summary="done")\n```'), ok('phase_done(summary="done")'), ok("switched"))
    client, _ = build(server)
    first = client.chat(USER, tools)
    assert first.tool_calls[0]["tool"] == "phase_done" and first.tool_calls[0]["args"] == {"summary": "done"}
    assert first.text == "" and client.active_tool_mode == "text", "a miss: the endpoint is not doing native tool calls"
    client.chat(USER, tools)
    assert "tools" not in server.body(1), "and native is not tried again"


# ---- Qwen3-Coder XML with a missing wrapper --------------------------------------------------------------------
# Captured from a real run: the reply had <function=...> and a closing </tool_call>, but no opening <tool_call>
# (the provider's own parser had consumed it). Every phase stalled on replies like this one.

XML_TOOLS = PHASE_TOOLS + [schema("list_dir", path="string")]

NO_OPENING_TAG = (
    "I'll start by examining the repository structure and locating the problematic code.\n\n"
    "<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>"
)


def test_a_function_block_without_its_opening_tool_call_tag_is_a_call():
    parsed = parse_text_tool_calls(NO_OPENING_TAG, XML_TOOLS)
    assert parsed.calls == [{"tool": "list_dir", "args": {"path": "."}}]
    assert parsed.text == "I'll start by examining the repository structure and locating the problematic code."


def test_the_stray_closing_tag_goes_with_the_call():
    assert parse_text_tool_calls("<function=git_diff>\n</function>\n</tool_call>", XML_TOOLS).text == ""
    assert parse_text_tool_calls("<function=git_diff></function></function_call>", XML_TOOLS).text == ""
    assert parse_text_tool_calls("<function=git_diff></function>", XML_TOOLS).calls == [{"tool": "git_diff", "args": {}}]


def test_a_bare_function_blocks_parameters_are_typed_by_the_schema():
    reply = "<function=read_file>\n<parameter=path>\ncalc.py\n</parameter>\n<parameter=start>\n1\n</parameter>\n</function>"
    assert calls(reply, XML_TOOLS) == [("read_file", {"path": "calc.py", "start": 1})]


def test_several_bare_function_blocks_are_several_calls():
    reply = "<function=git_diff></function>\n<function=list_dir><parameter=path>src</parameter></function>"
    assert calls(reply, XML_TOOLS) == [("git_diff", {}), ("list_dir", {"path": "src"})]


def test_a_bare_function_block_for_something_that_is_not_a_tool_is_text():
    reply = "<function=frobnicate>\n<parameter=x>1</parameter>\n</function>"
    assert calls(reply, XML_TOOLS) == []
    assert parse_text_tool_calls(reply, XML_TOOLS).text == reply


def test_a_bare_function_block_that_was_cut_off_is_not_run():
    """Unlike a wrapped call, there is nothing to say the model meant it: never act on half of an edit."""
    assert calls("<function=edit_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=old>\nreturn a - b", XML_TOOLS) == []


def test_a_lone_opening_tag_is_no_call_and_does_not_crash():
    parsed = parse_text_tool_calls("<tool_call>", XML_TOOLS)
    assert parsed.calls == []


def test_a_wrapped_call_is_still_read_by_the_wrapper_path():
    reply = "<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n</parameter>\n</function>\n</tool_call>"
    parsed = parse_text_tool_calls(reply, XML_TOOLS)
    assert parsed.calls == [{"tool": "list_dir", "args": {"path": "."}}] and parsed.text == ""


def test_the_client_rescues_a_wrapperless_function_block_from_a_native_reply():
    from tests.test_llm_client import ok

    server = Server(ok(NO_OPENING_TAG))
    client, _ = build(server)
    response = client.chat(USER, XML_TOOLS)
    assert response.tool_calls[0]["tool"] == "list_dir" and response.tool_calls[0]["args"] == {"path": "."}
    assert response.text.startswith("I'll start by examining")
