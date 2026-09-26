"""A read_file result of more than N lines becomes its first lines plus an outline, with a hint to read a range.

Measured on a real repository: reading all of Flask's blueprints.py cost ~6,400 tokens in every prompt until it aged out.
"""

from __future__ import annotations

import re

import pytest

from anvil.context import ContextManager
from anvil.context.tokens import estimate_tokens
from anvil.context.truncate import cap_read_output

CAP = 4000


def numbered(lines: list[str], first: int = 1) -> str:
    """Text as the read_file tool prints it: ``%6d  line``."""
    return "".join(f"{i:6d}  {line}\n" for i, line in enumerate(lines, start=first))


def python_file(functions: int = 60) -> list[str]:
    lines = ["import os", "", "class Widget:", '    """A widget."""', ""]
    for i in range(functions):
        lines += [f"    def method_{i}(self, value):", f'        """Docs for {i}."""', f"        return value + {i}", ""]
    return lines


def cap(text: str, path: str = "pkg/mod.py", max_lines: int = 150, head: int = 60, budget: int = CAP) -> str:
    return cap_read_output(text, {"path": path}, max_lines, head, budget)


# ---- when it applies -------------------------------------------------------------------------------------------


def test_a_read_of_no_more_than_the_limit_is_returned_untouched():
    text = numbered([f"line {i}" for i in range(150)])
    assert cap(text) == text


def test_one_line_over_the_limit_is_shortened():
    text = numbered([f"line {i}" for i in range(151)])
    assert cap(text) != text and "read_file shortened" in cap(text)


def test_a_long_read_becomes_a_hint_the_head_and_an_outline():
    text = numbered(python_file())
    out = cap(text)

    assert out.startswith("[read_file shortened: 245 lines (numbered 1 to 245) is more than the 150 shown at once.")
    assert "read_file(path='pkg/mod.py', start=<first line>, end=<last line>)" in out and "at most 150 lines" in out
    assert "     1  import os" in out and "     3  class Widget:" in out, "the head is there, numbered as before"
    assert "def method_59(self, value):" in out, "the outline reaches the end of the file"
    assert len(out) <= CAP


def test_a_large_file_costs_a_fraction_of_the_tokens_it_did():
    text = numbered(python_file(200))  # 805 lines
    assert estimate_tokens(cap(text)) < estimate_tokens(text) / 4


def test_the_outline_gives_the_line_number_of_each_definition_to_use_in_a_range_request():
    lines = python_file()
    out = cap(numbered(lines), head=10)
    position = next(i for i, l in enumerate(lines, start=1) if "def method_30(" in l)
    assert f"  {position}: def method_30(self, value):" in out


def test_definitions_already_in_the_head_are_not_listed_again_in_the_outline():
    out = cap(numbered(python_file()), head=30, budget=100_000)
    outline = out.split("[outline of lines")[1]
    assert "class Widget" not in outline and "method_0(" not in outline
    assert "method_20(" in outline


def test_a_range_read_keeps_its_real_line_numbers_in_the_notice_and_the_outline():
    lines = python_file(80)[:400]
    out = cap(numbered(lines, first=300), budget=100_000)
    assert "(numbered 300 to " in out
    assert "[outline of lines " in out and " 300:" not in out.split("[outline of lines")[1].split("\n")[0]
    assert "start=<first line>" in out


def test_no_path_in_the_arguments_still_gives_a_usable_hint():
    out = cap_read_output(numbered(python_file()), {}, 150, 60, CAP)
    assert "read_file(start=<first line>, end=<last line>)" in out
    assert "read_file shortened" in cap_read_output(numbered(python_file()), None, 150, 60, CAP)


# ---- what it leaves alone --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "File not found: src/missing.py",
        "\n".join(f"src/file_{i}.py" for i in range(400)),  # a listing, not numbered file text
        "\n".join(f"just a very long line of output {i}" for i in range(400)),
    ],
    ids=["error", "listing", "unnumbered"],
)
def test_output_that_is_not_numbered_file_text_is_not_touched(text):
    assert cap(text) == text


def test_it_only_applies_to_read_file_results_in_the_context_manager():
    big = numbered(python_file())
    on = ContextManager(read_file_max_lines=150, read_file_head_lines=60, tool_output_char_cap=100_000)

    def call(tool: str) -> str:
        on.add_message("assistant", "", tool_calls=[{"id": tool, "type": "function", "function": {"name": tool, "arguments": "{}"}}])
        on.add_message("tool", big, tool_call_id=tool, ok=True)
        return on.build_messages("goal")[-1]["content"]

    assert "read_file shortened" in call("read_file")
    assert call("grep") == big, "another tool's output is only ever subject to the ordinary output cap"


def test_switched_off_the_manager_keeps_the_read_whole():
    big = numbered(python_file())
    off = ContextManager(tool_output_char_cap=100_000)
    off.add_message("assistant", "", tool_calls=[{"id": "r", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}])
    off.add_message("tool", big, tool_call_id="r", ok=True)
    assert off.build_messages("goal")[-1]["content"] == big


def test_the_manager_passes_the_call_arguments_to_the_hint_and_a_pruned_result_still_reads_as_one_line():
    manager = ContextManager(read_file_max_lines=150, read_file_head_lines=60, tool_output_char_cap=CAP, keep_steps=1)
    manager.add_message(
        "assistant", "", tool_calls=[{"id": "r", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a/b.py"}'}}]
    )
    manager.add_message("tool", numbered(python_file()), tool_call_id="r", ok=True)
    assert "read_file(path='a/b.py', start=" in manager.build_messages("goal")[-1]["content"]
    for step in range(3):  # later steps age the read out
        manager.add_message("assistant", f"step {step}")
    assert any(m["content"].startswith("[output pruned] read_file(") for m in manager.build_messages("goal"))


# ---- the size limits -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [4000, 2500, 1200])
def test_the_result_always_fits_the_character_budget_by_shrinking_the_head_then_the_outline(budget):
    out = cap(numbered(python_file(200)), budget=budget)
    assert len(out) <= budget
    assert "read_file shortened" in out


def test_the_head_never_shrinks_below_ten_lines_before_the_outline_does():
    out = cap(numbered(python_file(200)), head=65, budget=1200)  # (65 is no multiple of the step of 10) too tight for even ten lines plus a full outline
    assert all(f"{n:6d}  " in out for n in range(1, 11)), "lines 1 to 10 survive"
    assert "Here are the first 10 lines" in out, "the head stops at its floor, and the outline is what gives"
    assert len(out) <= 1200


def test_at_most_eighty_definitions_are_listed_and_the_rest_are_counted():
    lines = ["import os"] + [f"def f{i}():\n    pass" for i in range(300)]
    out = cap(numbered("\n".join(lines).split("\n")), head=10, budget=1_000_000)
    listed = re.findall(r"^  \d+: def f\d+", out, re.MULTILINE)
    assert len(listed) == 80
    assert "[215 more definitions not listed]" in out, "300 definitions, 5 of them inside the 10-line head"


# ---- other languages -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "definition",
    [
        "function render(props) {",
        "export default function App() {",
        "export async function load(id) {",
        "func (s *Server) Handle(w http.ResponseWriter) {",
        "pub fn parse(input: &str) -> Result<Ast> {",
        "pub(crate) async fn run() {",
        "impl Display for Point {",
        "struct Point {",
        "enum Kind {",
        "interface Options {",
        "async def fetch(self):",
        "class Foo(Base):",
    ],
)
def test_the_outline_recognises_definitions_in_the_common_languages(definition):
    lines = ["// header"] * 20 + [definition] + ["    body"] * 200
    out = cap(numbered(lines), head=10, budget=1_000_000)
    assert f"  21: {definition}" in out


@pytest.mark.parametrize(
    "prose",
    [
        "function name will be used.",
        "type of the value that is returned",
        "class of objects that share behaviour, described in prose.",
        "object which is passed to the callback",
        "function (described below) is called once, when the phase ends.",
        "type (a str or None) of the value that is returned.",
    ],
)
def test_prose_that_starts_with_a_keyword_is_not_an_outline_entry(prose):
    lines = ["# header"] * 20 + [f"    {prose}"] + ["    body"] * 200
    assert "  21:" not in cap(numbered(lines), head=10, budget=1_000_000)
