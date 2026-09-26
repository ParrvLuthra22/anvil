import json

import pytest

from anvil.llm.toolcalls import adapt_messages_for_text_mode, loads_lenient, parse_text_tool_call

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


def fenced(body: str) -> str:
    return f"```json\n{body}\n```"


# ---- parsing -------------------------------------------------------------------------


def test_plain_block():
    call, rest = parse_text_tool_call(fenced('{"tool": "read_file", "args": {"path": "a.py"}}'))
    assert call == {"tool": "read_file", "args": {"path": "a.py"}}
    assert rest == ""


def test_prose_before_and_after_block_is_kept_as_remaining_text():
    text = 'I will look at the file.\n' + fenced('{"tool": "read_file", "args": {"path": "a.py"}}') + "\nThen I will fix it."
    call, rest = parse_text_tool_call(text)
    assert call["tool"] == "read_file"
    assert rest == "I will look at the file.\n\nThen I will fix it."


def test_several_blocks_take_the_first():
    text = (
        fenced('{"tool": "read_file", "args": {"path": "first.py"}}')
        + "\nand then\n"
        + fenced('{"tool": "read_file", "args": {"path": "second.py"}}')
    )
    call, rest = parse_text_tool_call(text)
    assert call["args"]["path"] == "first.py"
    assert "second.py" in rest


def test_non_tool_json_block_is_skipped_for_a_later_tool_block():
    text = fenced('{"name": "pkg", "version": "1.0"}') + "\n" + fenced('{"tool": "read_file", "args": {}}')
    call, _ = parse_text_tool_call(text)
    assert call == {"tool": "read_file", "args": {}}


def test_json_without_tool_key_is_not_a_call():
    text = "package.json has " + fenced('{"name": "pkg", "args": {"x": 1}}')
    assert parse_text_tool_call(text) == (None, text)


def test_bare_json_without_fence():
    call, rest = parse_text_tool_call('Sure. {"tool": "read_file", "args": {"path": "a.py"}} done')
    assert call == {"tool": "read_file", "args": {"path": "a.py"}}
    assert rest == "Sure.\n\ndone"


def test_untagged_fence_and_single_line_fence():
    call, _ = parse_text_tool_call('```\n{"tool": "x", "args": {}}\n```')
    assert call["tool"] == "x"
    call, _ = parse_text_tool_call('```json {"tool": "y", "args": {"a": 1}} ```')
    assert call == {"tool": "y", "args": {"a": 1}}


def test_missing_args_defaults_to_empty_and_arguments_alias_and_string_args():
    assert parse_text_tool_call(fenced('{"tool": "git_diff"}'))[0] == {"tool": "git_diff", "args": {}}
    assert parse_text_tool_call(fenced('{"tool": "t", "arguments": {"a": 1}}'))[0]["args"] == {"a": 1}
    assert parse_text_tool_call(fenced('{"tool": "t", "args": "{\\"a\\": 1}"}'))[0]["args"] == {"a": 1}


def test_args_key_before_tool_key_inside_a_fence():
    call, _ = parse_text_tool_call(fenced('{"args": {"path": "a"}, "tool": "read_file"}'))
    assert call == {"tool": "read_file", "args": {"path": "a"}}


@pytest.mark.parametrize(
    "body",
    [
        '{"tool": "read_file", "args": {"path": "a.py",}}',  # trailing comma
        "{'tool': 'read_file', 'args': {'path': 'a.py'}}",  # single quotes
        '{"tool": "read_file", "args": {"path": "a.py"}',  # missing closing brace
        '{"tool": "read_file", "args": {"path": "a.py", "flag": True, "other": None}}',  # python literals
        '{"tool": "read_file", "args": {"path": "a.py"}},',  # dangling comma
    ],
)
def test_malformed_json_is_repaired(body):
    call, rest = parse_text_tool_call(fenced(body))
    assert call["tool"] == "read_file"
    assert call["args"]["path"] == "a.py"
    assert rest == ""


def test_raw_newlines_and_stray_backslashes_inside_strings_are_repaired():
    body = '{"tool": "edit_file", "args": {"old": "def f():\n\treturn 1", "pattern": "\\d+\\.py"}}'
    call, _ = parse_text_tool_call(fenced(body))
    assert call["args"]["old"] == "def f():\n\treturn 1"
    assert call["args"]["pattern"] == "\\d+\\.py"


def test_python_literals_inside_strings_are_left_alone():
    call, _ = parse_text_tool_call(fenced('{"tool": "edit_file", "args": {"new": "x = True  # None"}}'))
    assert call["args"]["new"] == "x = True  # None"


def test_unrepairable_json_returns_no_call_and_the_raw_text():
    text = "Here you go:\n" + fenced('{"tool": "read_file", "args": {"path": }}')
    assert parse_text_tool_call(text) == (None, text)


def test_value_cut_off_inside_a_string_is_never_completed():
    text = '```json\n{"tool": "edit_file", "args": {"path": "a.py", "new": "def f():\n    ret'
    assert parse_text_tool_call(text) == (None, text)


def test_text_without_any_call_is_returned_untouched():
    text = "All done. The bug was an off-by-one."
    assert parse_text_tool_call(text) == (None, text)


def test_non_dict_or_blank_tool_name_is_rejected():
    assert parse_text_tool_call(fenced('{"tool": 5, "args": {}}'))[0] is None
    assert parse_text_tool_call(fenced('{"tool": "  ", "args": {}}'))[0] is None
    assert parse_text_tool_call(fenced('{"tool": "t", "args": [1, 2]}'))[0] is None


def test_loads_lenient_raises_value_error_on_garbage():
    with pytest.raises(ValueError):
        loads_lenient("not json at all {")


# ---- outgoing message adaptation -----------------------------------------------------


def test_instructions_are_added_to_an_existing_system_message_without_mutating_input():
    messages = [{"role": "system", "content": "You are an engineer."}, {"role": "user", "content": "fix it"}]
    adapted = adapt_messages_for_text_mode(messages, TOOLS)
    assert adapted[0]["content"].startswith("You are an engineer.")
    assert "read_file: Read a file." in adapted[0]["content"]
    assert '"tool"' in adapted[0]["content"]
    assert messages[0]["content"] == "You are an engineer."
    assert adapted[1] == messages[1]


def test_a_system_message_is_created_when_missing():
    adapted = adapt_messages_for_text_mode([{"role": "user", "content": "hi"}], TOOLS)
    assert adapted[0]["role"] == "system"
    assert adapted[1]["content"] == "hi"


def test_no_tools_means_no_instructions():
    messages = [{"role": "user", "content": "hi"}]
    assert adapt_messages_for_text_mode(messages, None) == messages


def test_tool_history_is_rewritten_as_plain_turns():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "Reading.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "print(1)"},
    ]
    adapted = adapt_messages_for_text_mode(messages, None)
    assert [m["role"] for m in adapted] == ["user", "assistant", "user"]
    assert all("tool_calls" not in m and "tool_call_id" not in m for m in adapted)
    reparsed, remaining = parse_text_tool_call(adapted[1]["content"])
    assert reparsed == {"tool": "read_file", "args": {"path": "a.py"}}
    assert remaining == "Reading."
    assert adapted[2]["content"] == "[result of read_file]\nprint(1)"


def test_tool_message_without_a_matching_call_still_converts():
    adapted = adapt_messages_for_text_mode([{"role": "tool", "tool_call_id": "zzz", "content": "out"}], None)
    assert adapted == [{"role": "user", "content": "[result of tool]\nout"}]


def test_assistant_message_with_empty_tool_calls_keeps_only_text():
    adapted = adapt_messages_for_text_mode([{"role": "assistant", "content": "ok", "tool_calls": []}], None)
    assert adapted == [{"role": "assistant", "content": "ok"}]


def test_adapted_history_round_trips_through_json():
    adapted = adapt_messages_for_text_mode([{"role": "user", "content": "x"}], TOOLS)
    json.dumps(adapted)
