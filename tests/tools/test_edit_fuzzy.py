"""edit_file's whitespace-tolerant path: the match, the re-indentation, and the line endings.

Real failure (pytest-dev__pytest-11143, Qwen3-Coder): the model sent `old` indented two spaces deeper than the file. The tolerant
path matched it, but replaced the matched lines (final line break included) with a `new` that had no trailing newline, so the
next line was glued on ("expect_docstring = False            elif (") and the syntax guard reverted the edit with
"invalid syntax at line 684, column 53".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from anvil.sandbox.base import ExecResult
from anvil.tools.edit_file import EditFileTool


class Box:
    """The parts of a sandbox that edit_file touches, over a temporary directory (bytes are kept exactly: no newline translation)."""

    def __init__(self, root: Path):
        self.root = root

    def read_file(self, path: str, start=None, end=None) -> str:
        return (self.root / path).read_bytes().decode()

    def write_file(self, path: str, content: str) -> None:
        (self.root / path).write_bytes(content.encode())

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        return ExecResult(0, "", "", False, 0.0)


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path)


def edit(box: Box, path: str, old: str, new: str):
    return EditFileTool().run({"path": path, "old": old, "new": new}, box)


# The block from pytest's rewrite.py (12-space indent), and the model's text for it (14 spaces), verbatim from the real run.
FILE = (
    "    def run(self, mod):\n"
    "        for item in mod.body:\n"
    "            if (\n"
    "                expect_docstring\n"
    "                and isinstance(item, ast.Expr)\n"
    "                and isinstance(item.value, ast.Constant)\n"
    "            ):\n"
    "                doc = item.value.value\n"
    "                if self.is_rewrite_disabled(doc):\n"
    "                    return\n"
    "                expect_docstring = False\n"
    "            elif (\n"
    "                isinstance(item, ast.ImportFrom)\n"
    "                and item.level == 0\n"
    "                and item.module == \"__future__\"\n"
    "            ):\n"
    "                pass\n"
)
MODEL_OLD = (
    "              if (\n"
    "                  expect_docstring\n"
    "                  and isinstance(item, ast.Expr)\n"
    "                  and isinstance(item.value, ast.Constant)\n"
    "              ):\n"
    "                  doc = item.value.value\n"
    "                  if self.is_rewrite_disabled(doc):\n"
    "                      return\n"
    "                  expect_docstring = False"
)
MODEL_NEW = (
    "              if (\n"
    "                  expect_docstring\n"
    "                  and isinstance(item, ast.Expr)\n"
    "                  and isinstance(item.value, ast.Constant)\n"
    "              ):\n"
    "                  doc = item.value.value\n"
    "                  # Only check docstring for rewrite disabling if it's actually a string\n"
    "                  if isinstance(doc, str) and self.is_rewrite_disabled(doc):\n"
    "                      return\n"
    "                  expect_docstring = False"
)


def dedent_file() -> str:
    return "class Rewriter:\n" + FILE


# ---- the real failure -------------------------------------------------------------------------------------------------------


def test_the_pytest_11143_edit_applies_and_the_file_still_parses(box):
    box.write_file("rewrite.py", dedent_file())
    result = edit(box, "rewrite.py", MODEL_OLD, MODEL_NEW)

    assert result.ok, result.output
    text = box.read_file("rewrite.py")
    ast.parse(text)
    assert "                if isinstance(doc, str) and self.is_rewrite_disabled(doc):\n" in text, "re-indented to the file's 12/16 spaces"
    assert "                expect_docstring = False\n            elif (\n" in text, "and the next line is on its own line"


def test_the_line_after_the_block_is_never_glued_on(box):
    box.write_file("rewrite.py", dedent_file())
    edit(box, "rewrite.py", MODEL_OLD, MODEL_NEW)
    assert "False            elif" not in box.read_file("rewrite.py")


# ---- trailing newlines ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_newline, new_newline",
    [(False, False), (True, True), (False, True), (True, False)],
    ids=["neither has one", "both have one", "only new has one", "only old has one"],
)
def test_the_files_own_line_break_after_the_block_is_kept_whatever_the_model_sent(box, old_newline, new_newline):
    box.write_file("m.py", "def f():\n    x = 1\n    y = 2\n    z = 3\n")
    old = "x = 1\ny = 2" + ("\n" if old_newline else "")
    new = "x = 10\ny = 20" + ("\n" if new_newline else "")
    result = edit(box, "m.py", old, new)
    assert result.ok, result.output
    assert box.read_file("m.py") == "def f():\n    x = 10\n    y = 20\n    z = 3\n"


def test_a_block_at_the_end_of_a_file_with_no_final_newline_stays_without_one(box):
    box.write_file("m.py", "def f():\n    x = 1\n    y = 2")
    assert edit(box, "m.py", "x = 1\ny = 2", "x = 10\ny = 20").ok
    assert box.read_file("m.py") == "def f():\n    x = 10\n    y = 20"


def test_a_block_at_the_end_of_a_file_that_ends_with_a_newline_keeps_it(box):
    box.write_file("m.py", "def f():\n    x = 1\n    y = 2\n")
    assert edit(box, "m.py", "x = 1\ny = 2", "x = 10\ny = 20").ok
    assert box.read_file("m.py") == "def f():\n    x = 10\n    y = 20\n"


# ---- CRLF ---------------------------------------------------------------------------------------------------------------------------


def test_a_crlf_file_stays_crlf_all_the_way_through(box):
    box.write_file("w.py", "def f():\r\n    x = 1\r\n    y = 2\r\n    z = 3\r\n")
    result = edit(box, "w.py", "x = 1\ny = 2", "x = 10\ny = 20")
    assert result.ok, result.output
    raw = box.read_file("w.py")
    assert raw == "def f():\r\n    x = 10\r\n    y = 20\r\n    z = 3\r\n"
    assert "\n" not in raw.replace("\r\n", ""), "no bare line feed was introduced"


def test_a_crlf_file_with_the_model_using_crlf_too(box):
    box.write_file("w.py", "def f():\r\n    x = 1\r\n    z = 3\r\n")
    assert edit(box, "w.py", "x = 1\r\n", "x = 10\r\n").ok
    assert box.read_file("w.py") == "def f():\r\n    x = 10\r\n    z = 3\r\n"


def test_a_file_that_is_lf_stays_lf_when_the_model_sends_crlf(box):
    box.write_file("w.py", "def f():\n    x = 1\n    z = 3\n")
    assert edit(box, "w.py", "x = 1\r\n", "x = 10\r\n").ok
    assert box.read_file("w.py") == "def f():\n    x = 10\n    z = 3\n"


# ---- indentation --------------------------------------------------------------------------------------------------------------------


def test_a_tab_indented_file_gets_tab_indentation_at_every_level(box):
    box.write_file("t.py", "def f():\n\tif x:\n\t\ty = 1\n\tz = 2\n")
    result = edit(box, "t.py", "    if x:\n        y = 1", "    if x:\n        y = 10\n        w = 3")
    assert result.ok, result.output
    assert box.read_file("t.py") == "def f():\n\tif x:\n\t\ty = 10\n\t\tw = 3\n\tz = 2\n"


def test_the_model_indenting_deeper_than_the_file_is_taken_back_to_the_files_indentation(box):
    box.write_file("d.py", "class A:\n    def f(self):\n        return 1\n")
    assert edit(box, "d.py", "            def f(self):\n                return 1", "            def f(self):\n                return 2").ok
    assert box.read_file("d.py") == "class A:\n    def f(self):\n        return 2\n"


def test_the_model_indenting_shallower_than_the_file_is_pushed_out_to_the_files_indentation(box):
    box.write_file("d.py", "class A:\n    def f(self):\n        if True:\n            return 1\n")
    assert edit(box, "d.py", "if True:\n    return 1", "if True:\n    return 2").ok
    assert box.read_file("d.py") == "class A:\n    def f(self):\n        if True:\n            return 2\n"


def test_a_new_line_dedented_relative_to_the_first_line_stays_dedented_relative_to_it(box):
    box.write_file("d.py", "def f():\n    if a:\n        x = 1\n    else:\n        x = 2\n")
    result = edit(box, "d.py", "        if a:\n            x = 1\n        else:\n            x = 2", "        if b:\n            x = 1\n        else:\n            x = 3")
    assert result.ok, result.output
    assert box.read_file("d.py") == "def f():\n    if b:\n        x = 1\n    else:\n        x = 3\n"


def test_blank_lines_inside_the_block_stay_blank_with_no_trailing_spaces(box):
    box.write_file("b.py", "def f():\n    a = 1\n\n    b = 2\n")
    assert edit(box, "b.py", "        a = 1\n\n        b = 2", "        a = 10\n\n        b = 20").ok
    assert box.read_file("b.py") == "def f():\n    a = 10\n\n    b = 20\n"


def test_deleting_a_block_through_the_tolerant_path_removes_whole_lines(box):
    box.write_file("x.py", "def f():\n    a = 1\n    b = 2\n    c = 3\n")
    assert edit(box, "x.py", "        b = 2", "").ok  # (8 spaces: not a substring of the file's 4-space line, so the tolerant path)
    assert box.read_file("x.py") == "def f():\n    a = 1\n    c = 3\n"


# ---- what the model is told ----------------------------------------------------------------------------------------------------------


def test_the_result_says_the_match_was_whitespace_tolerant_and_shows_the_resulting_lines(box):
    box.write_file("m.py", "def f():\n    x = 1\n    y = 2\n    z = 3\n")
    result = edit(box, "m.py", "x = 1\ny = 2", "x = 10\ny = 20")

    assert result.ok and result.meta["tolerant_match"] is True and result.meta["lines"] == [2, 3]
    assert "whitespace-tolerant" in result.output.lower() and "indentation" in result.output
    assert "lines 2-3" in result.output.lower()
    assert "     2      x = 10\n     3      y = 20" in result.output, "the lines as they now are, numbered like read_file, indentation included"
    assert "z = 3" not in result.output, "only the changed lines are shown"


def test_an_exact_match_is_reported_as_before_and_is_not_called_tolerant(box):
    box.write_file("m.py", "def f():\n    x = 1\n")
    result = edit(box, "m.py", "    x = 1", "    x = 2")
    assert result.ok and "whitespace-tolerant" not in result.output
    assert not result.meta.get("tolerant_match")


def test_a_long_result_is_shown_only_up_to_a_limit(box):
    lines = [f"    v{i} = {i}\n" for i in range(60)]
    box.write_file("big.py", "def f():\n" + "".join(lines))
    old = "\n".join(f"v{i} = {i}" for i in range(60))
    new = "\n".join(f"v{i} = {i + 1}" for i in range(60))
    result = edit(box, "big.py", old, new)
    assert result.ok and result.output.count("v") < 200 and "more lines" in result.output


def test_a_tolerant_edit_that_breaks_the_syntax_is_still_reverted(box):
    box.write_file("s.py", "def f():\n    x = 1\n    y = 2\n")
    result = edit(box, "s.py", "x = 1", "x = (1")
    assert not result.ok and "Edit reverted due to syntax error" in result.output
    assert box.read_file("s.py") == "def f():\n    x = 1\n    y = 2\n"


def test_ambiguous_and_missing_matches_are_still_reported_as_before(box):
    box.write_file("a.py", "def f():\n    x = 1\ndef g():\n  x = 1\n")
    result = edit(box, "a.py", "        x = 1", "        x = 2")
    assert not result.ok and "2 times (ignoring indentation)" in result.output
    missing = edit(box, "a.py", "        q = 9", "        q = 8")
    assert not missing.ok and "String not found" in missing.output


def test_a_space_indented_file_gets_spaces_when_the_model_used_tabs(box):
    box.write_file("s.py", "def f():\n    if x:\n        y = 1\n    z = 2\n")
    assert edit(box, "s.py", "\tif x:\n\t\ty = 1", "\tif x:\n\t\ty = 10\n\t\tw = 3").ok
    assert box.read_file("s.py") == "def f():\n    if x:\n        y = 10\n        w = 3\n    z = 2\n"


def test_a_two_space_file_gets_two_space_levels_when_the_model_used_four(box):
    box.write_file("s.py", "def f():\n  if x:\n    y = 1\n  z = 2\n")
    assert edit(box, "s.py", "        if x:\n            y = 1", "        if x:\n            y = 10\n            w = 3").ok
    assert box.read_file("s.py") == "def f():\n  if x:\n    y = 10\n    w = 3\n  z = 2\n"


def test_a_new_line_shallower_than_the_first_line_is_as_much_shallower_in_the_file(box):
    box.write_file("d.py", "def f():\n    if a:\n        x = 1\n")
    result = edit(box, "d.py", "        if a:\n            x = 1", "        if a:\n            x = 1\n    y = 2")
    assert result.ok, result.output
    assert box.read_file("d.py") == "def f():\n    if a:\n        x = 1\ny = 2\n", "4 shallower than the model's first line: 4 shallower than the file's"


def test_an_old_that_starts_with_a_blank_line_is_indented_from_its_first_real_line(box):
    box.write_file("d.py", "def f():\n\n    x = 1\n    y = 2\n")
    result = edit(box, "d.py", "\n        x = 1\n        y = 2", "\n        x = 10\n        y = 20")
    assert result.ok, result.output
    assert box.read_file("d.py") == "def f():\n\n    x = 10\n    y = 20\n"

