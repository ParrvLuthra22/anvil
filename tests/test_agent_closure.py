"""The summary the harness writes when it closes a phase itself (agent/closure.py): deterministic, from the calls made."""

from __future__ import annotations

from anvil.agent.closure import MAX_ITEMS, MAX_SUMMARY_CHARS, close_summary
from anvil.agent.loop import ToolRecord
from anvil.events import Phase


def rec(tool: str, ok: bool = True, meta: dict | None = None, **args) -> ToolRecord:
    return ToolRecord(tool, args, ok, "", meta or {})


def test_the_summary_says_it_was_written_by_the_harness_and_which_phase_and_cap():
    text = close_summary(Phase.REPRODUCE, 10, [])
    assert text.startswith("[Written by the harness: the REPRODUCE phase used its 10 calls and was closed without a verdict from the model.]")


def test_no_calls_and_no_words_is_said_plainly():
    assert "No tool call was made in this phase." in close_summary(Phase.LOCALIZE, 8, [])


def test_the_models_last_note_is_kept_on_one_line_and_clipped():
    text = close_summary(Phase.LOCALIZE, 8, [], "It is in\n   calc.py,   line 2.")
    assert "The model's last note: It is in calc.py, line 2." in text
    long = close_summary(Phase.LOCALIZE, 8, [], "x" * 5000)
    assert "x" * 501 not in long and "more chars]" in long


def test_every_kind_of_call_is_reported_with_its_outcome():
    records = [
        rec("read_file", path="calc.py", start=1, end=40),
        rec("read_file", path="tests/test_calc.py"),
        rec("grep", pattern="def add"),
        rec("list_dir", path="src"),
        rec("write_repro", path=".anvil/repro.py"),
        rec("run_cmd", ok=False, meta={"exit_code": 1}, cmd="python3 .anvil/repro.py"),
        rec("run_cmd", ok=True, meta={"exit_code": 0}, cmd="ls"),
        rec("run_cmd", ok=False, meta={"exit_code": -1, "timed_out": True}, cmd="sleep 999"),
        rec("run_tests", ok=True, targets=["tests/test_calc.py"]),
        rec("run_tests", ok=False),
        rec("edit_file", path="calc.py"),
        rec("edit_file", ok=False, path="other.py"),
        rec("git_diff"),
        rec("git_diff"),
    ]
    text = close_summary(Phase.PATCH, 15, records)
    assert "Files read: calc.py (lines 1-40), tests/test_calc.py" in text
    assert "Searched for: 'def add'" in text
    assert "Directories listed: src" in text
    assert "Repro scripts written: .anvil/repro.py" in text
    assert "Commands run: `python3 .anvil/repro.py` (failed, exit 1), `ls` (exit 0), `sleep 999` (timed out)" in text
    assert "Tests run: tests/test_calc.py (passed), all (failed)" in text
    assert "Edits made: calc.py (ok), other.py (failed)" in text
    assert "The diff was looked at 2 time(s)." in text


def test_repeated_items_are_listed_once_and_a_long_list_is_cut_with_a_count():
    same = [rec("grep", pattern="def add")] * 4
    assert close_summary(Phase.LOCALIZE, 8, same).count("'def add'") == 1
    many = [rec("read_file", path=f"m{i}.py") for i in range(MAX_ITEMS + 3)]
    text = close_summary(Phase.LOCALIZE, 8, many)
    assert f"m{MAX_ITEMS - 1}.py and 3 more" in text and f"m{MAX_ITEMS}.py" not in text


def test_odd_arguments_never_raise():
    weird = [
        ToolRecord("read_file", {}, True, "", {}),
        ToolRecord("grep", {"pattern": None}, True, "", {}),
        ToolRecord("run_cmd", {"cmd": 12345}, False, "", {}),
        ToolRecord("run_tests", {"target": ["a", "b"]}, True, "", {}),
        ToolRecord("edit_file", {"path": ["x"]}, True, "", {}),
        ToolRecord("read_file", "not a dict", True, "", {}),  # type: ignore[arg-type]
    ]
    text = close_summary(Phase.VERIFY, 8, weird, None)  # type: ignore[arg-type]
    assert "12345" in text and "a b (passed)" in text


def test_the_summary_is_never_longer_than_its_limit_even_when_every_kind_of_call_fills_its_list():
    def long(prefix: str, i: int) -> str:
        return f"{prefix}{i}" + "z" * 200

    records = []
    for i in range(20):
        records += [
            rec("read_file", path=long("r", i)),
            rec("grep", pattern=long("g", i)),
            rec("list_dir", path=long("d", i)),
            rec("write_repro", path=long("w", i)),
            rec("run_cmd", ok=False, meta={"exit_code": 1}, cmd=long("c", i)),
            rec("run_tests", targets=[long("t", i)]),
            rec("edit_file", path=long("e", i)),
        ]
    text = close_summary(Phase.PATCH, 15, records, "n" * 900)
    assert text.endswith(" more chars]"), "so much was done that the limit is what cuts the summary, and says so"
    assert 1500 < len(text) < 1540, "the summary that goes into every later prompt is cut at 1500 characters (plus the marker)"
    assert text.startswith("[Written by the harness"), "and it is cut from the end, not the start"
