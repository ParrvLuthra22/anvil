"""Patch filtering, confidence scoring and report rendering."""

import pytest

from anvil.agent.outputs import changed_files, filter_diff, render_report, write_outputs
from anvil.agent.state import CheckRun, RunState
from anvil.repo.ingest import IssueRef

SRC_DIFF = (
    "diff --git a/calc.py b/calc.py\n"
    "index 111..222 100644\n"
    "--- a/calc.py\n"
    "+++ b/calc.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n"
    "-    return a - b\n"
    "+    return a + b\n"
)
REPRO_DIFF = (
    "diff --git a/.anvil/repro.py b/.anvil/repro.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/.anvil/repro.py\n"
    "@@ -0,0 +1 @@\n"
    "+assert add(2, 3) == 5\n"
)
NEW_FILE_DIFF = (
    "diff --git a/docs/notes.md b/docs/notes.md\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/docs/notes.md\n"
    "@@ -0,0 +1,2 @@\n"
    "+one\n"
    "+two\n"
)


# ---- filter_diff ----------------------------------------------------------------------------


def test_scratch_hunks_are_removed_and_source_hunks_kept():
    assert filter_diff(REPRO_DIFF + SRC_DIFF) == SRC_DIFF
    assert filter_diff(SRC_DIFF + REPRO_DIFF) == SRC_DIFF


def test_diff_with_only_scratch_files_becomes_empty():
    assert filter_diff(REPRO_DIFF) == ""


def test_scratch_files_in_nested_scratch_dirs_are_removed():
    nested = REPRO_DIFF.replace(".anvil/repro.py", ".anvil/helpers/util.py")
    assert filter_diff(nested + SRC_DIFF) == SRC_DIFF


def test_lookalike_paths_are_not_treated_as_scratch():
    lookalike = SRC_DIFF.replace("calc.py", "src/.anvil_config/calc.py")
    other = SRC_DIFF.replace("calc.py", "docs.anvil/calc.py")
    assert filter_diff(lookalike + other) == lookalike + other


def test_non_diff_text_from_a_sandbox_is_dropped():
    assert filter_diff("(no baseline - diff unavailable)") == ""
    assert filter_diff("") == ""
    assert filter_diff("warning: something\n" + SRC_DIFF) == SRC_DIFF


def test_result_always_ends_with_a_newline():
    assert filter_diff(SRC_DIFF.rstrip("\n")).endswith("\n")


def test_plain_unified_diffs_without_git_headers_are_supported():
    plain_src = "--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-x\n+y\n"
    plain_scratch = "--- /dev/null\n+++ b/.anvil/repro.py\n@@ -0,0 +1 @@\n+z\n"
    assert filter_diff(plain_scratch + plain_src) == plain_src


def test_deleting_a_scratch_file_is_filtered_too():
    deletion = (
        "diff --git a/.anvil/old.py b/.anvil/old.py\ndeleted file mode 100644\n"
        "--- a/.anvil/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    assert filter_diff(deletion + SRC_DIFF) == SRC_DIFF


# ---- changed_files --------------------------------------------------------------------------


def test_changed_files_counts_added_and_removed_lines():
    files = changed_files(SRC_DIFF + NEW_FILE_DIFF)
    assert [(f.path, f.added, f.removed) for f in files] == [("calc.py", 1, 1), ("docs/notes.md", 2, 0)]


def test_changed_files_does_not_miscount_headers_or_content_that_looks_like_them():
    tricky = (
        "diff --git a/x.txt b/x.txt\n--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n"
        "--- removed line that starts with dashes\n+++ added line that starts with pluses\n"
    )
    (change,) = changed_files(tricky)
    assert (change.added, change.removed) == (1, 1)


def test_changed_files_of_a_deleted_file_reports_its_old_path():
    deletion = "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    assert changed_files(deletion)[0].path == "gone.py"


def test_changed_files_of_empty_patch_is_empty():
    assert changed_files("") == []


# ---- confidence -----------------------------------------------------------------------------


def _state(**overrides) -> RunState:
    base = dict(repro_confirmed=True, verified=True, review="approved", checks=[CheckRun("pytest", True)])
    return RunState(issue_url="u", **{**base, **overrides})


def test_confidence_high_needs_repro_verification_clean_tests_and_approval():
    assert _state().confidence(True) == "high"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"review": "changes requested"}, "medium"),
        ({"review": "not run"}, "medium"),
        ({"checks": [CheckRun("pytest", False)]}, "medium"),
        ({"repro_confirmed": False}, "low"),
        ({"verified": False}, "low"),
    ],
)
def test_confidence_downgrades_on_each_caveat(overrides, expected):
    assert _state(**overrides).confidence(True) == expected


def test_no_patch_means_no_confidence_whatever_else_is_true():
    assert _state().confidence(False) == "none"
    assert _state().confidence_score(False) == 0.0


def test_confidence_score_is_a_fraction_between_zero_and_one():
    scores = [_state(**o).confidence_score(True) for o in ({}, {"review": "not run"}, {"verified": False})]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1 for s in scores)


# ---- report ---------------------------------------------------------------------------------


def _report(state: RunState, patch: str = SRC_DIFF) -> str:
    return render_report(state, patch, steps=14, tokens=1234, seconds=12.34)


def test_report_has_every_required_section():
    state = _state(
        issue=IssueRef("acme", "calc", 7, "https://github.com/acme/calc/issues/7", title="add() is wrong"),
        understanding="add subtracts",
        repro_cmd="python .anvil/repro.py",
        final_summary="Fixed the sign.",
    )
    state.limit("something to know")
    report = _report(state)
    for expected in (
        "add() is wrong (acme/calc#7)",
        "add subtracts",
        "Confidence: **high**",
        "yes (`python .anvil/repro.py`)",
        "`calc.py` (+1 / -1)",
        "pass: pytest",
        "Fixed the sign.",
        "LLM calls: 14",
        "Tokens: 1234",
        "Wall clock: 12.3s",
        "- something to know",
    ):
        assert expected in report, expected


def test_report_for_a_failed_run_is_still_complete():
    report = _report(RunState(issue_url="https://github.com/a/b/issues/1"), patch="")
    assert "https://github.com/a/b/issues/1" in report
    assert "Confidence: **none** (0.00)" in report
    assert "None: no patch was produced" in report
    assert "None recorded." in report
    assert "None noted." in report


def test_limitations_are_deduplicated():
    state = RunState(issue_url="u")
    state.limit("a")
    state.limit("a")
    assert state.limitations == ["a"]


# ---- write_outputs --------------------------------------------------------------------------


def test_write_outputs_creates_the_directory_and_both_files(tmp_path):
    patch_path, report_path = write_outputs(tmp_path / "out" / "nested", SRC_DIFF, "# r\n")
    assert patch_path.read_text() == SRC_DIFF
    assert report_path.read_text() == "# r\n"
    assert patch_path.name == "patch.diff" and report_path.name == "report.md"
