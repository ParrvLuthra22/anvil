"""The recovery policy on its own: loop detection, edit feedback, argument checks, result review, checkpoints."""

from __future__ import annotations

import pytest

from anvil.agent.emitter import Emitter
from anvil.agent.recovery import (
    LOOP_STRIKE_LIMIT,
    MAX_SILENT_REPLIES,
    NUDGE,
    Checkpointer,
    ErrorClass,
    LoopDetector,
    PhaseGuard,
    call_signature,
    closest_lines,
    edit_failure_feedback,
    failure_digest,
    invalid_arguments_reply,
    llm_failure_advice,
    review_result,
    signature,
    timed_out,
    unknown_tool_reply,
    validate_arguments,
)
from anvil.events import EventBus, Phase
from anvil.llm.errors import LLMConfigError, LLMError
from tests.fakes import BUGGY, FakeSandbox, StashStyleSandbox, project_sandbox

# ---- loop detection -------------------------------------------------------------------------


def feed(detector: LoopDetector, *calls: tuple[str, dict]) -> list[str | None]:
    return [detector.observe(tool, args) for tool, args in calls]


A, B, C = ("grep", {"pattern": "add"}), ("read_file", {"path": "calc.py"}), ("list_dir", {"path": "."})


def test_the_same_call_three_times_in_a_row_is_a_loop_and_two_are_not():
    detector = LoopDetector()
    assert feed(detector, A, A, A) == [None, None, "the same call 3 times in a row"]
    assert detector.strikes == 1


def test_different_arguments_are_a_different_call():
    detector = LoopDetector()
    assert feed(detector, A, ("grep", {"pattern": "sub"}), A, ("grep", {"pattern": "sub"})) == [None] * 2 + [None, "alternating between the same two calls"]


def test_argument_order_does_not_matter():
    assert call_signature("edit_file", {"a": 1, "b": 2}) == call_signature("edit_file", {"b": 2, "a": 1})
    detector = LoopDetector()
    assert feed(detector, ("t", {"a": 1, "b": 2}), ("t", {"b": 2, "a": 1}), ("t", {"a": 1, "b": 2}))[-1]


def test_alternating_a_b_a_b_is_a_loop_on_the_fourth_call():
    detector = LoopDetector()
    assert feed(detector, A, B, A, B) == [None, None, None, "alternating between the same two calls"]


@pytest.mark.parametrize("calls", [[A, A, B, B], [A, B, C, A, B, C], [A, B, A, C], [A, A, B, A, A]])
def test_other_patterns_are_not_loops(calls):
    assert not any(feed(LoopDetector(), *calls))


def test_a_different_call_breaks_the_streak():
    assert not any(feed(LoopDetector(), A, A, B, A, A, B))


def test_a_model_that_ignores_the_warning_gets_a_strike_per_repeat_and_is_then_exhausted():
    detector = LoopDetector()
    feed(detector, A, A, A)
    assert detector.strikes == 1 and not detector.exhausted
    assert detector.observe(*A) == "the same call 4 times in a row" and detector.strikes == 2
    detector.observe(*A)
    assert detector.strikes == LOOP_STRIKE_LIMIT and detector.exhausted


def test_alternation_that_goes_on_also_runs_out_of_strikes():
    detector = LoopDetector()
    feed(detector, A, B, A, B, A, B)
    assert detector.exhausted


def test_strikes_stay_when_the_model_changes_course_for_a_while():
    detector = LoopDetector()
    feed(detector, A, A, A, B, C, B, C)  # ...and a new pattern later
    assert detector.strikes == 2


# ---- edit feedback --------------------------------------------------------------------------

SOURCE = "class Calc:\n    def add(self, a, b):\n        return a - b\n\n    def sub(self, a, b):\n        return a - b\n"


def sandbox_with(text: str, path: str = "calc.py") -> FakeSandbox:
    return FakeSandbox({path: text})


def test_a_failed_edit_gets_the_closest_lines_with_numbers_and_the_reread_reminder():
    args = {"path": "calc.py", "old": "def add(self, x, y):", "new": "def add(self, x, y): pass"}
    feedback = edit_failure_feedback(args, "String not found exactly once.", sandbox_with(SOURCE))
    assert "Closest matching lines in calc.py:" in feedback
    assert "  2:     def add(self, a, b):" in feedback
    assert "Re-read the file with read_file before editing again" in feedback
    assert feedback.index("Closest") < feedback.index("Re-read")


def test_the_closest_lines_are_the_most_similar_ones_best_first():
    matches = closest_lines(SOURCE, "def sub(self, a, c):")
    assert matches[0] == (5, "    def sub(self, a, b):")
    assert len(matches) <= 3


def test_indentation_only_mismatches_are_called_out():
    args = {"path": "calc.py", "old": "return a - b\nreturn 0", "new": "x"}
    feedback = edit_failure_feedback(args, "String not found.", sandbox_with("def f(a, b):\n        return a - b\n"))
    assert "whitespace differs" in feedback and "  2:         return a - b" in feedback


def test_a_multi_line_edit_is_matched_as_a_block_showing_the_real_indentation():
    content = "import os\n\nclass Calc:\n    def add(self, a, b):\n        total = a - b\n        return total\n\n    def other(self):\n        pass\n"
    old = "def add(self, a, b):\n    total = a - b\n    return total"
    assert closest_lines(content, old) == [
        (4, "    def add(self, a, b):"),
        (5, "        total = a - b"),
        (6, "        return total"),
    ]
    feedback = edit_failure_feedback({"path": "calc.py", "old": old}, "String not found.", sandbox_with(content))
    assert "  5:         total = a - b" in feedback and "whitespace differs" in feedback


def test_a_block_that_agrees_on_fewer_than_two_lines_falls_back_to_similar_lines():
    content = "def add(a, b):\n    return a - b\n"
    matches = closest_lines(content, "def add(x, y):\nreturn x + y")
    assert matches and matches[0][0] == 1


def test_a_correctly_indented_block_with_a_typo_gets_no_whitespace_hint():
    content = "def add(a, b):\n    return a - b\n"
    old = "def add(a, b):\n    return a - c"
    feedback = edit_failure_feedback({"path": "calc.py", "old": old}, "String not found.", sandbox_with(content))
    assert "Closest matching lines" in feedback and "whitespace differs" not in feedback


def test_lines_the_tool_already_listed_are_not_repeated():
    output = "String not found in 'calc.py'.\nClosest existing lines (for reference):\n    def add(self, a, b):"
    feedback = edit_failure_feedback({"path": "calc.py", "old": "def add(x):"}, output, sandbox_with(SOURCE))
    assert "Closest matching lines" not in feedback and "Re-read the file" in feedback


def test_nothing_similar_still_yields_the_reminder():
    feedback = edit_failure_feedback({"path": "calc.py", "old": "zzzzzzzzzzzzqqqq"}, "String not found.", sandbox_with(SOURCE))
    assert "Closest" not in feedback and feedback.endswith("Do not retry the same edit.")


def test_a_missing_file_points_to_list_dir_and_grep():
    feedback = edit_failure_feedback({"path": "nope.py", "old": "x"}, "File not found: 'nope.py'", sandbox_with(SOURCE))
    assert "list_dir or grep" in feedback and "Re-read the file" in feedback


def test_an_unreadable_file_does_not_break_the_feedback():
    class Broken(FakeSandbox):
        def read_file(self, path, start=None, end=None):
            raise OSError("disk on fire")

    feedback = edit_failure_feedback({"path": "calc.py", "old": "def add"}, "String not found.", Broken({}))
    assert "Re-read the file" in feedback


def test_missing_arguments_do_not_break_the_feedback():
    assert "Re-read the file" in edit_failure_feedback({}, "'path' argument is required.", sandbox_with(SOURCE))


def test_blank_old_strings_have_no_closest_lines():
    assert closest_lines(SOURCE, "\n  \n") == []


# ---- argument checks and schemas ------------------------------------------------------------

READ_FILE = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
    "required": ["path"],
}


def test_valid_arguments_pass():
    assert validate_arguments(READ_FILE, {"path": "a.py", "start": 1, "end": 10}) == []
    assert validate_arguments(READ_FILE, {"path": "a.py", "start": 3.0}) == []
    assert validate_arguments(READ_FILE, {"path": "a.py", "extra": True}) == []


def test_a_missing_required_argument_is_named():
    assert validate_arguments(READ_FILE, {}) == ["missing required argument 'path'"]


def test_wrong_types_are_named():
    problems = validate_arguments(READ_FILE, {"path": 3, "start": "ten", "end": True})
    assert len(problems) == 3
    assert "'path' must be of type string, got int" in problems[0]
    assert any("'start'" in p for p in problems) and any("'end'" in p for p in problems), "a bool is not an integer"


def test_schemas_without_properties_accept_anything():
    assert validate_arguments({"type": "object", "properties": {}}, {"whatever": 1}) == []
    assert validate_arguments({}, {}) == []


def test_signature_marks_optional_arguments():
    assert signature("read_file", READ_FILE) == "read_file(path, start?, end?)"
    assert signature("git_diff", {"type": "object", "properties": {}}) == "git_diff()"


def test_invalid_arguments_reply_carries_the_schema():
    reply = invalid_arguments_reply("read_file", "missing required argument 'path'", READ_FILE)
    assert reply.startswith("Invalid arguments for 'read_file': missing required argument 'path'.")
    assert '"required":["path"]' in reply and "Call it again" in reply


def test_invalid_arguments_reply_without_a_known_schema_still_explains():
    assert "Invalid arguments for 'x': bad." in invalid_arguments_reply("x", "bad", None)


def test_unknown_tool_reply_lists_what_is_available_and_suggests_the_closest():
    offered = {"grep": {"properties": {"pattern": {}}, "required": ["pattern"]}, "read_file": READ_FILE}
    reply = unknown_tool_reply("read_fil", Phase.LOCALIZE, offered)
    assert "not available in the localize phase" in reply
    assert "Did you mean 'read_file'?" in reply and '"required":["path"]' in reply
    assert "grep(pattern), read_file(path, start?, end?)" in reply


def test_unknown_tool_reply_without_a_close_match_only_lists_tools():
    reply = unknown_tool_reply("frobnicate", Phase.PATCH, {"grep": READ_FILE})
    assert "Did you mean" not in reply and "Tools you can call here: grep(path, start?, end?)" in reply


# ---- reviewing results ----------------------------------------------------------------------

SB = project_sandbox()


def review(tool, ok, output, args=None, meta=None):
    return review_result(tool, args or {}, ok, output, meta or {}, SB)


def test_successful_results_are_left_alone():
    result = review("grep", True, "calc.py:1:def add")
    assert result.output == "calc.py:1:def add" and result.error_class is None


def test_a_failed_edit_is_classified_and_gets_feedback():
    result = review(
        "edit_file", False, "String not found exactly once in 'calc.py'.", {"path": "calc.py", "old": "return a * b"}
    )
    assert result.error_class is ErrorClass.EDIT_FAILED
    assert result.output.startswith("String not found exactly once")
    assert "Closest matching lines in calc.py" in result.output and "Re-read the file" in result.output


def test_a_failed_test_run_gets_its_key_lines_first_and_guidance():
    output = "collected 1 item\ntests/test_calc.py F\nFAILED tests/test_calc.py::test_add - AssertionError\nE   assert -1 == 5\n1 failed"
    result = review("run_tests", False, output)
    assert result.error_class is ErrorClass.TEST_FAILURE
    head = result.output.split("collected")[0]
    assert "[test run failed]" in head and "FAILED tests/test_calc.py::test_add" in head and "E   assert -1 == 5" in head
    assert "Never weaken, skip or delete a test" in head
    assert result.output.endswith(output), "the raw output is kept in full"


def test_a_failed_test_run_without_recognisable_lines_still_gets_guidance():
    result = review("run_tests", False, "something odd happened")
    assert result.error_class is ErrorClass.TEST_FAILURE and "Key failure lines" not in result.output
    assert "pre-existing" in result.output


def test_a_command_that_exits_non_zero_is_information_not_an_error():
    result = review("run_cmd", False, "[exit code 1]\nAssertionError: add(2, 3) == -1")
    assert result.error_class is None and result.output.startswith("[exit code 1]")


@pytest.mark.parametrize("tool", ["run_cmd", "run_tests"])
def test_a_timeout_is_classified_whatever_the_tool(tool):
    result = review(tool, False, "[TIMED OUT after 120s]", meta={"timed_out": True})
    assert result.error_class is ErrorClass.TIMEOUT and "run something narrower" in result.output


def test_a_timeout_is_recognised_from_the_text_alone_and_even_when_ok():
    assert timed_out("... [TIMED OUT after 5s]", {}) and timed_out("", {"timed_out": True}) and not timed_out("fine", {})
    assert review("run_cmd", True, "[TIMED OUT after 5s]").error_class is ErrorClass.TIMEOUT


def test_any_other_failing_tool_is_a_tool_error_with_advice():
    result = review("read_file", False, "File not found: 'nope.py'")
    assert result.error_class is ErrorClass.TOOL_ERROR
    assert "list_dir or grep" in result.output and "do not repeat the same call" in result.output
    assert "list_dir" not in review("git_diff", False, "boom").output


@pytest.mark.parametrize(
    "output, expected",
    [
        ("FAILED tests/a.py::t - assert 1 == 2\nE   assert 1 == 2\n=== 1 failed ===", ["FAILED tests/a.py::t - assert 1 == 2", "E   assert 1 == 2"]),
        ("--- FAIL: TestAdd (0.00s)\n    add_test.go:9: got -1\nFAIL\nFAIL\texample.com/calc\t0.01s", ["--- FAIL: TestAdd (0.00s)", "FAIL\texample.com/calc\t0.01s"]),
        ("  ● add › adds\n    expect(received).toBe(expected)\nFAIL src/add.test.js", ["● add › adds", "FAIL src/add.test.js"]),
        ("test add::t ... FAILED\nthread 'add::t' panicked at src/lib.rs:4:9:", ["test add::t ... FAILED", "thread 'add::t' panicked at src/lib.rs:4:9:"]),
        ("[ERROR] Tests run: 3, Failures: 1\njava.lang.AssertionError: expected:<5> but was:<-1>", ["[ERROR] Tests run: 3, Failures: 1", "java.lang.AssertionError: expected:<5> but was:<-1>"]),
        ("FAIL: test_add (tests.test_calc.T)\nAssertionError: -1 != 5", ["FAIL: test_add (tests.test_calc.T)", "AssertionError: -1 != 5"]),
        ("all good\nnothing to see", []),
    ],
)
def test_failure_digest_finds_the_failing_lines_of_the_common_test_runners(output, expected):
    assert failure_digest(output) == expected


def test_failure_digest_is_capped_deduplicated_and_clipped():
    output = "\n".join(["FAILED a"] * 5 + [f"FAILED test_{i}" for i in range(50)] + ["FAILED " + "x" * 500])
    digest = failure_digest(output, limit=4)
    assert digest == ["FAILED a", "FAILED test_0", "FAILED test_1", "FAILED test_2"]
    assert all(len(line) < 260 for line in failure_digest("FAILED " + "x" * 500))


# ---- per-phase guard ------------------------------------------------------------------------


def make_guard(phase=Phase.LOCALIZE, tools=None):
    bus = EventBus()
    queue = bus.subscribe()
    guard = PhaseGuard(Emitter(bus), phase, tools if tools is not None else {"read_file": READ_FILE, "phase_done": {"required": ["summary"], "properties": {"summary": {"type": "string"}}}})

    def events():
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return [(e.data["kind"], e.data["message"]) for e in out if e.type == "error"]

    return guard, events


def test_the_guard_answers_a_repeat_with_a_corrective_message_and_an_event():
    guard, events = make_guard(Phase.LOCALIZE)
    assert guard.repeated("grep", {"pattern": "x"}) is None and guard.repeated("grep", {"pattern": "x"}) is None
    verdict = guard.repeated("grep", {"pattern": "x"})
    assert verdict and not verdict.exhausted
    assert "you are repeating yourself" in verdict.reply and "Try a different approach" in verdict.reply
    assert f"Strike 1 of {LOOP_STRIKE_LIMIT}" in verdict.reply and "search for a different identifier" in verdict.reply
    ((kind, message),) = events()
    assert kind == "loop" and "grep(pattern='x')" in message and "strike 1/3" in message


def test_the_guard_ends_the_phase_at_the_third_strike():
    guard, events = make_guard(Phase.PATCH)
    verdicts = [guard.repeated("run_cmd", {"cmd": "pytest"}) for _ in range(5)]
    assert [v is not None for v in verdicts] == [False, False, True, True, True]
    assert [v.exhausted for v in verdicts[2:]] == [False, False, True]
    assert "the model kept repeating itself" in verdicts[-1].reason and "run_cmd(cmd='pytest')" in verdicts[-1].reason
    assert events()[-1][1].endswith("ending the phase")


def test_each_phase_has_its_own_hint_and_unknown_phases_get_a_generic_one():
    for phase, expected in [(Phase.REPRODUCE, "repro script"), (Phase.VERIFY, "diagnose the failure"), (Phase.REVIEW, "approve")]:
        guard, _ = make_guard(phase)
        for _ in range(3):
            verdict = guard.repeated("t", {})
        assert expected in verdict.reply
    guard, _ = make_guard(Phase.FINALIZE)
    for _ in range(3):
        verdict = guard.repeated("t", {})
    assert "different arguments or a different tool" in verdict.reply


def test_bad_arguments_get_the_schema_and_an_event():
    guard, events = make_guard()
    reply = guard.check_arguments("read_file", {"start": 1})
    assert "missing required argument 'path'" in reply and '"required":["path"]' in reply
    assert events() == [("invalid_call", "read_file: missing required argument 'path'; sent the schema")]
    assert guard.check_arguments("read_file", {"path": "a.py"}) is None
    assert guard.check_arguments("no_schema_tool", {"anything": 1}) is None


def test_an_unknown_tool_gets_the_tool_list_and_an_event():
    guard, events = make_guard()
    reply = guard.unknown_tool("read_fil")
    assert "Did you mean 'read_file'?" in reply
    assert events()[0][0] == "invalid_call"


def test_the_guard_nudges_once_then_ends_the_phase():
    assert MAX_SILENT_REPLIES == 1
    guard, events = make_guard()
    assert guard.silent_reply() == NUDGE
    assert guard.silent_reply() is None
    assert [kind for kind, _ in events()] == ["no_tool_call", "no_tool_call"]


def test_a_tool_call_resets_the_nudge():
    guard, _ = make_guard()
    assert guard.silent_reply() == NUDGE
    guard.tool_used()
    assert guard.silent_reply() == NUDGE


def test_reviewing_announces_only_real_problems():
    guard, events = make_guard()
    assert guard.review("grep", {}, True, "ok", {}, SB) == "ok" and events() == []
    text = guard.review("read_file", {}, False, "File not found: 'x'", {}, SB)
    assert "list_dir or grep" in text
    assert [kind for kind, _ in events()] == ["tool"]


# ---- LLM failures ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected",
    [
        (LLMConfigError("AI_API_KEY is not set"), "AI_API_KEY"),
        (LLMError("HTTP 401", status_code=401), "rejected the credentials"),
        (LLMError("HTTP 403", status_code=403), "rejected the credentials"),
        (LLMError("HTTP 404", status_code=404), "model or endpoint was not found"),
        (LLMError("HTTP 413", status_code=413), "max_context_tokens"),
        (LLMError("HTTP 400", status_code=400, detail="This model's maximum context length is 8192 tokens"), "max_context_tokens"),
        (LLMError("HTTP 429", status_code=429, retryable=True, attempts=5), "rate limit or quota"),
        (LLMError("HTTP 503", status_code=503, retryable=True, attempts=5), "kept failing after 5 attempts"),
        (LLMError("connection reset", retryable=True, attempts=3), "network or provider kept failing after 3 attempts"),
        (LLMError("HTTP 400", status_code=400), "rejected"),
    ],
)
def test_llm_failures_get_advice_by_cause(exc, expected):
    assert expected in llm_failure_advice(exc)


# ---- checkpoint and rollback ----------------------------------------------------------------


def make_checkpointer(sandbox, protected=lambda: ()):
    bus = EventBus()
    queue = bus.subscribe()

    def events():
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    return Checkpointer(sandbox, Emitter(bus), protected), events


def test_checkpoint_and_rollback_reset_the_tree_through_the_sandbox_interface():
    sandbox = project_sandbox()
    checkpointer, events = make_checkpointer(sandbox)
    ref = checkpointer.checkpoint("patch-start")
    sandbox.write_file("calc.py", "def add(a, b):\n    return a * b\n")
    assert checkpointer.rollback(ref, ["multiplied"]) is True
    assert sandbox.files["calc.py"] == BUGGY
    assert [e.type for e in events()] == ["tool_call", "tool_result", "tool_call", "tool_result", "error"]


def test_a_rollback_announces_itself_and_what_the_model_will_be_told():
    sandbox = project_sandbox()
    checkpointer, events = make_checkpointer(sandbox)
    ref = checkpointer.checkpoint("s")
    checkpointer.rollback(ref, ["a", "b"])
    (error,) = [e for e in events() if e.type == "error"]
    assert error.data["kind"] == "rollback" and ref in error.data["message"] and "2 approach(es)" in error.data["message"]


def test_the_protected_files_survive_a_sandbox_that_sweeps_untracked_files():
    sandbox = StashStyleSandbox({"calc.py": BUGGY}, on_exec=None)
    sandbox.write_file(".anvil/repro.py", "print('repro')")
    checkpointer, _ = make_checkpointer(sandbox, lambda: [".anvil/repro.py", ".anvil/gone.py"])
    ref = checkpointer.checkpoint("s")
    assert sandbox.files[".anvil/repro.py"] == "print('repro')", "checkpoint swept the repro away"
    sandbox.write_file("calc.py", "changed")
    assert checkpointer.rollback(ref) and sandbox.files[".anvil/repro.py"] == "print('repro')"
    assert ".anvil/gone.py" not in sandbox.files


def test_a_failing_checkpoint_is_reported_and_returns_none():
    class Broken(FakeSandbox):
        def checkpoint(self, label):
            raise RuntimeError("no git")

    checkpointer, events = make_checkpointer(Broken({}))
    assert checkpointer.checkpoint("s") is None
    errors = [e for e in events() if e.type == "error"]
    assert errors[0].data["kind"] == "sandbox" and "no git" in errors[0].data["message"]


def test_a_failing_rollback_is_reported_and_returns_false():
    class Broken(FakeSandbox):
        def rollback(self, ref):
            raise RuntimeError("stash lost")

    sandbox = Broken({"a": "1"})
    checkpointer, events = make_checkpointer(sandbox)
    ref = checkpointer.checkpoint("s")
    assert checkpointer.rollback(ref) is False
    assert [e.data["kind"] for e in events() if e.type == "error"] == ["sandbox"]


def test_rolling_back_to_no_checkpoint_does_nothing():
    checkpointer, events = make_checkpointer(project_sandbox())
    assert checkpointer.rollback(None) is False and events() == []
