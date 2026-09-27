"""features.patch_sanity: a patch that introduces a bare assert in non-test code is flagged, and the flag earns the one retry.

Seen with a real model on pallets__flask-4045, whose issue asks for a ValueError: it wrote an assert, which is removed under
python -O and raises AssertionError. The patch works, so this is a warning (not a failed check): one retry, then delivery with
the warning if the assert is still there.
"""

from __future__ import annotations

import pytest

from anvil.agent.sanity import MAX_LISTED_ASSERTS, added_asserts, inspect_patch
from tests.fakes import REPRO_CMD, call, done, give_up, reply
from tests.test_agent_patch_sanity import Stub, git_world, kickoffs, sanity_line
from tests.test_orchestrator import execute, finalize, good_patch, localize, reproduce, review_ok, understand, verify


def patch_with(path: str, *added: str, removed: tuple[str, ...] = (), context: tuple[str, ...] = ("def add(a, b):",), start: int = 1) -> str:
    body = "".join(f" {c}\n" for c in context) + "".join(f"-{r}\n" for r in removed) + "".join(f"+{a}\n" for a in added)
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -{start},{len(context) + len(removed)} +{start},{len(context) + len(added)} @@\n{body}"


# ---- what counts -----------------------------------------------------------------------------------------------------


def test_an_added_assert_in_source_is_found_with_its_file_and_line():
    patch = patch_with("calc.py", '    assert isinstance(a, int), "a must be an int"', "    return a + b", removed=("    return a - b",))
    assert added_asserts(patch) == [("calc.py", 2, 'assert isinstance(a, int), "a must be an int"')]


@pytest.mark.parametrize("code", ["    assert x > 0", "    assert(x > 0)", '    assert x > 0, "message"', "\tassert x", "assert x"])
def test_the_forms_of_assert_that_are_flagged(code):
    assert [n for _, n, _ in added_asserts(patch_with("m.py", code))] == [2]


@pytest.mark.parametrize(
    "code",
    [
        "    assert_called_with(1)", "    assertEqual(a, b)", "    assert.equal(a, b)", "    assert = 1", "    assert[0] = 1",
        "    assertions = 3", "    # assert x > 0", '    """assert x > 0"""', "    x = 'assert x'", "    reassert(x)", "    self.assert_ok()",
    ],
)
def test_lines_that_only_look_like_an_assert_are_not_flagged(code):
    assert added_asserts(patch_with("m.py", code)) == []


@pytest.mark.parametrize(
    "path", ["tests/test_calc.py", "test_calc.py", "conftest.py", "calc_test.py", "pkg/tests/helpers.py", "src/FooTest.java"]
)
def test_asserts_in_test_files_are_fine(path):
    assert added_asserts(patch_with(path, "    assert x > 0")) == []


@pytest.mark.parametrize("path", ["web/app.js", "main.go", "lib.rs", "notes.md", "config.yaml", "src/app.ts"])
def test_only_languages_where_assert_is_a_statement_are_checked(path):
    assert added_asserts(patch_with(path, "    assert x > 0")) == []


def test_java_source_is_checked_too():
    assert added_asserts(patch_with("src/main/java/Calc.java", "        assert a > 0 : \"a must be positive\";")) == [
        ("src/main/java/Calc.java", 2, 'assert a > 0 : "a must be positive";')
    ]


def test_an_assert_that_was_already_there_is_not_introduced_by_the_patch():
    context_only = patch_with("m.py", "    y = 2", context=("def f(x):", "    assert x > 0"))
    assert added_asserts(context_only) == []
    deleted = patch_with("m.py", "    y = 2", removed=("    assert x > 0",))
    assert added_asserts(deleted) == []


def test_moving_or_reindenting_an_assert_is_not_introducing_one_but_changing_its_message_is():
    same = patch_with("m.py", "        assert x > 0", removed=("    assert x > 0",))
    assert added_asserts(same) == []
    reworded = patch_with("m.py", '    assert x > 0, "must be positive"', removed=("    assert x > 0",))
    assert [code for _, _, code in added_asserts(reworded)] == ['assert x > 0, "must be positive"']


def test_line_numbers_follow_the_new_file_across_hunks_and_files():
    first = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n import os\n+assert os\n import sys\n"
        "@@ -20,2 +21,3 @@\n def f():\n     pass\n+    assert f\n"
    )
    second = patch_with("pkg/b.py", "    assert g", context=("x = 1", "y = 2", "z = 3"), start=10)
    assert added_asserts(first + second) == [("a.py", 2, "assert os"), ("a.py", 23, "assert f"), ("pkg/b.py", 13, "assert g")]


def test_a_new_file_is_read_from_its_first_line():
    new = "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,3 @@\n+import os\n+\n+assert os\n"
    assert added_asserts(new) == [("n.py", 3, "assert os")]


def test_a_no_newline_marker_is_not_a_line_even_between_a_removed_and_an_added_line():
    at_the_end = patch_with("m.py", "assert a", "x = 1") + "\\ No newline at end of file\n"
    assert [n for _, n, _ in added_asserts(at_the_end)] == [2]
    between = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,3 @@\n def f():\n-    return 1\n"
        "\\ No newline at end of file\n+    assert a\n+    return 1\n"
    )
    assert added_asserts(between) == [("m.py", 2, "assert a")], "the marker is not a context line, so it does not push the line down"


def test_a_one_line_hunk_whose_header_omits_the_counts_is_read_too():
    """git writes ``@@ -1 +1 @@`` for a change of a single line."""
    patch = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+assert x\n"
    assert added_asserts(patch) == [("m.py", 1, "assert x")]
    new = "diff --git a/n.py b/n.py\nnew file mode 100644\n--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+assert x\n"
    assert added_asserts(new) == [("n.py", 1, "assert x")]


def test_nothing_is_nothing():
    assert added_asserts("") == []


# ---- what the check reports ----------------------------------------------------------------------------------------


def test_the_check_still_passes_but_carries_a_warning_naming_the_file_and_line(tmp_path):
    check = inspect_patch(Stub(tmp_path), patch_with("calc.py", "    assert a > 0"), allow_tests=False)
    assert check.ok and check.apply_checked, "the patch applies: this is not a problem"
    (warning,) = check.warnings
    assert "calc.py:2 (`assert a > 0`)" in warning and "python -O" in warning and "AssertionError" in warning
    assert "ValueError, TypeError, or the type the issue names" in warning
    assert check.warning_text == warning


def test_a_clean_patch_has_no_warnings(tmp_path):
    assert inspect_patch(Stub(tmp_path), patch_with("calc.py", "    return a + b"), allow_tests=False).warnings == ()


def test_a_long_list_is_cut_with_a_count(tmp_path):
    lines = [f"    assert v{i}" for i in range(MAX_LISTED_ASSERTS + 2)]
    (warning,) = inspect_patch(Stub(tmp_path), patch_with("m.py", *lines), allow_tests=False).warnings
    assert f"m.py:{MAX_LISTED_ASSERTS + 1}" in warning and "and 2 more" in warning and f"m.py:{MAX_LISTED_ASSERTS + 2}" not in warning


def test_asserts_in_test_files_that_are_taken_out_of_the_patch_are_not_counted(tmp_path):
    patch = patch_with("calc.py", "    return a + b") + patch_with("tests/test_calc.py", "    assert add(1, 1) == 2")
    check = inspect_patch(Stub(tmp_path), patch, allow_tests=False)
    assert check.ok and check.warnings == () and check.removed_tests == ("tests/test_calc.py",)


def test_a_patch_that_does_not_apply_reports_that_problem_and_no_warning(tmp_path):
    from anvil.sandbox.base import ExecResult

    stub = Stub(tmp_path, {" apply --check": ExecResult(1, "", "error: patch does not apply\n", False, 0.0)})
    check = inspect_patch(stub, patch_with("calc.py", "    assert a"), allow_tests=False)
    assert not check.ok and check.warnings == ()


# ---- in a run ------------------------------------------------------------------------------------------------------------

GUARD = 'assert isinstance(a, int), "a must be an int"'
RAISE = 'if not isinstance(a, int):\n        raise TypeError("a must be an int")'


def asserting_patch():
    return [
        reply(call("edit_file", path="calc.py", old="return a - b", new=f"{GUARD}\n    return a + b")),
        reply(call("run_cmd", cmd=REPRO_CMD)),
        done("fixed, with a guard on the input"),
    ]


def solve_with_the_assert():
    return understand() + localize() + reproduce() + asserting_patch() + verify() + review_ok()


def replace_the_assert():
    return [reply(call("edit_file", path="calc.py", old=GUARD, new=RAISE)), reply(call("run_cmd", cmd=REPRO_CMD)), done("raises TypeError now")]


def test_a_patch_with_an_added_assert_gets_the_one_retry_and_it_is_told_what_to_do(tmp_path):
    script = solve_with_the_assert() + replace_the_assert() + verify() + finalize()
    run = execute(script, tmp_path, pipeline=git_world())

    (told,) = kickoffs(run)
    assert "It works, but:" in told and "calc.py:" in told and "python -O" in told
    assert "raise" in told.lower() and "This is your last attempt" in told and "Change nothing else" in told
    assert "raise TypeError" in run.patch and "assert" not in run.patch
    line = sanity_line(run)
    assert line.startswith("- Patch sanity: passed after one forced-fix retry (the first patch was flagged: The patch adds a bare assert")
    assert "WARNING" not in line and "## Warnings" not in run.report, "the retry replaced it, so nothing is left to warn about"
    assert "Verified after patching: yes" in run.report and "Confidence: **high**" in run.report
    assert run.llm.remaining == 0


def test_an_assert_that_survives_the_retry_is_delivered_with_a_warning_and_the_confidence_is_not_lowered(tmp_path):
    keep = [reply(call("edit_file", path="calc.py", old="return a + b", new="return a + b  # kept")), reply(call("run_cmd", cmd=REPRO_CMD)), done("left it")]
    script = solve_with_the_assert() + keep + verify() + finalize()
    run = execute(script, tmp_path, pipeline=git_world())

    assert len(kickoffs(run)) == 1, "one retry, and no second"
    assert "assert isinstance" in run.patch
    assert "WARNING: The patch adds a bare assert in non-test code: calc.py:" in sanity_line(run)
    assert "## Warnings" in run.report and "- **WARNING:** The patch adds a bare assert" in run.report
    assert "Confidence: **high**" in run.report, "a warning is not a failed check"
    assert run.llm.remaining == 0


def test_a_retry_that_breaks_verification_is_undone_and_the_earlier_patch_is_delivered_with_the_warning(tmp_path):
    breaks = [reply(call("edit_file", path="calc.py", old=GUARD, new=RAISE)), reply(call("edit_file", path="calc.py", old="return a + b", new="return a - b")), done("oops")]
    script = solve_with_the_assert() + breaks + finalize()
    fake = git_world()
    run = execute(script, tmp_path, pipeline=fake)

    assert "The retry to replace the assert failed verification; the earlier patch was restored." in run.report
    assert "return a + b" in run.patch and "assert isinstance" in run.patch, "the verified patch, as it was"
    assert "Verified after patching: yes" in run.report
    assert any(e[0] == "rollback" for e in fake.sandbox.events)
    assert "WARNING: The patch adds a bare assert" in sanity_line(run)


def test_a_run_that_had_already_stopped_is_looked_at_without_a_retry_and_carries_the_warning(tmp_path):
    run = execute(solve_with_the_assert(), tmp_path, pipeline=git_world(), max_total_steps=14)
    assert kickoffs(run) == []
    assert "WARNING: The patch adds a bare assert" in sanity_line(run)
    assert "## Warnings" in run.report


def test_with_patch_sanity_off_an_assert_is_not_looked_for(tmp_path):
    script = solve_with_the_assert() + finalize()
    run = execute(script, tmp_path, pipeline=git_world(), features={"patch_sanity": False})
    assert "Patch sanity" not in run.report and kickoffs(run) == [] and "assert isinstance" in run.patch


def test_a_retry_for_an_empty_patch_keeps_its_own_wording_and_is_not_the_warning_retry(tmp_path):
    """The two retries share one path: the problem's kickoff says the patch cannot be handed in, the warning's says it works."""
    script = understand() + localize() + reproduce() + [give_up("no idea")] + good_patch() + verify() + finalize()
    run = execute(script, tmp_path, pipeline=git_world(), max_patch_attempts=1, max_rollbacks=0)
    (told,) = kickoffs(run)
    assert "cannot be handed in as it is" in told and "It works, but" not in told
    assert not any(e[0] == "checkpoint" and e[1] == "before-sanity-retry" for e in run.pipeline.sandbox.events), "no safety net: there was nothing to lose"


def test_a_patch_without_an_assert_is_untouched_by_all_this(tmp_path):
    script = understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, pipeline=git_world())
    assert sanity_line(run) == "- Patch sanity: passed" and kickoffs(run) == []
