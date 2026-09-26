"""Token estimation and the text-shortening helpers behind the context manager."""

import re

from anvil.context.tokens import (
    MESSAGE_OVERHEAD_TOKENS,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)
from anvil.context.truncate import first_line, format_call, truncate_output

# ---- token estimate -------------------------------------------------------------------------


def test_tokens_are_chars_over_four_rounded_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 4000) == 1000


def test_a_message_costs_its_content_plus_a_fixed_overhead():
    assert estimate_message_tokens({"role": "user", "content": "x" * 40}) == MESSAGE_OVERHEAD_TOKENS + 10


def test_tool_call_arguments_count_towards_an_assistant_message():
    plain = {"role": "assistant", "content": ""}
    calls = [{"id": "c1", "type": "function", "function": {"name": "edit_file", "arguments": "y" * 400}}]
    assert estimate_message_tokens({**plain, "tool_calls": calls}) > estimate_message_tokens(plain) + 100


def test_a_missing_content_is_free():
    assert estimate_message_tokens({"role": "assistant", "content": None}) == MESSAGE_OVERHEAD_TOKENS


def test_a_prompt_is_the_sum_of_its_messages():
    messages = [{"role": "system", "content": "a" * 8}, {"role": "user", "content": "b" * 12}]
    assert estimate_messages_tokens(messages) == 2 * MESSAGE_OVERHEAD_TOKENS + 2 + 3


# ---- truncate_output ------------------------------------------------------------------------


def test_short_output_is_returned_untouched():
    assert truncate_output("a\nb\nc", 100) == "a\nb\nc"
    assert truncate_output("x" * 100, 100) == "x" * 100


def test_long_output_keeps_head_and_tail_and_counts_the_omitted_lines():
    lines = [f"line {i}" for i in range(1000)]
    result = truncate_output("\n".join(lines), 500)
    assert len(result) <= 500
    assert result.startswith("line 0\nline 1\n") and result.endswith("line 998\nline 999")
    match = re.search(r"\[(\d+) lines omitted\]", result)
    assert match
    kept = [line for line in result.splitlines() if line.startswith("line ")]
    assert len(kept) + int(match.group(1)) == 1000


def test_the_omitted_lines_are_exactly_the_missing_ones():
    lines = [f"row-{i:04d}" for i in range(300)]
    result = truncate_output("\n".join(lines), 400)
    kept = [line for line in result.splitlines() if line.startswith("row-")]
    omitted = int(re.search(r"\[(\d+) lines omitted\]", result).group(1))
    head = [k for k in kept if int(k[4:]) < 150]
    tail = [k for k in kept if int(k[4:]) >= 150]
    assert int(head[-1][4:]) + 1 + omitted == int(tail[0][4:])


def test_a_single_omitted_line_is_not_pluralised():
    text = "a" * 200 + "\n" + "b" * 200 + "\n" + "c" * 200
    result = truncate_output(text, 470)
    assert "[1 line omitted]" in result and "b" * 200 not in result


def test_the_tail_of_a_command_survives_because_that_is_where_failures_are():
    output = "$ pytest\n" + "collecting...\n" * 500 + "FAILED tests/test_x.py::test_y - AssertionError\n1 failed"
    result = truncate_output(output, 300)
    assert result.startswith("$ pytest") and result.endswith("1 failed")
    assert "FAILED tests/test_x.py::test_y" in result


def test_output_made_of_huge_lines_is_cut_by_characters_instead():
    result = truncate_output("z" * 10_000, 400)
    assert len(result) <= 400
    assert re.search(r"\[\d+ chars omitted\]", result)
    assert result.startswith("zzz") and result.endswith("zzz")


def test_a_giant_first_line_falls_back_to_character_cutting_too():
    result = truncate_output("q" * 5000 + "\nsmall\nlines\n", 300)
    assert len(result) <= 300 and "chars omitted" in result


def test_the_marker_states_the_true_number_of_dropped_characters():
    result = truncate_output("z" * 10_000, 400)
    dropped = int(re.search(r"\[(\d+) chars omitted\]", result).group(1))
    assert dropped == 10_000 - result.count("z")


# ---- one-line helpers -----------------------------------------------------------------------


def test_first_line_skips_blank_lines_and_clips_long_ones():
    assert first_line("\n\n  hello world  \nsecond") == "hello world"
    assert first_line("") == ""
    clipped = first_line("x" * 500, 50)
    assert len(clipped) == 50 and clipped.endswith("…")


def test_format_call_renders_a_compact_signature():
    assert format_call("grep", {"pattern": "add", "path": "src"}) == "grep(pattern='add', path='src')"
    assert format_call("git_diff", {}) == "git_diff()"


def test_format_call_shortens_long_values_and_stays_on_one_line():
    call = format_call("write_repro", {"path": "r.py", "content": "line one\nline two\n" * 100})
    assert "\n" not in call and len(call) <= 120 and call.startswith("write_repro(path='r.py', content='line one")


def test_format_call_copes_with_arguments_that_are_not_a_dict():
    assert format_call("run_cmd", '{"cmd": "ls"}').startswith("run_cmd(")
    assert format_call("run_cmd", None) == "run_cmd()"
