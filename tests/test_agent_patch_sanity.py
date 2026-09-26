"""features.patch_sanity: the patch is looked at before it is delivered.

Empty, not applying to the original code, or editing the tests: the first two get ONE forced-fix retry with the reason
in the prompt, the third has the test changes taken out. Whatever happens is written to report.md.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from anvil.agent.outputs import split_patch
from anvil.agent.prompts import patch_kickoff
from anvil.agent.sanity import inspect_patch, is_test_path, issue_is_about_tests
from anvil.agent.state import RunState
from anvil.sandbox.base import ExecResult
from tests.fakes import FIX, REPRO_CMD, FakePipeline, FakeSandbox, call, done, give_up, project_exec, project_files, reply
from tests.test_orchestrator import execute, finalize, good_patch, happy, localize, reproduce, review_ok, understand, verify

OK = ExecResult(0, "", "", False, 0.0)
SOURCE_PATCH = "diff --git a/calc.py b/calc.py\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
TEST_PATCH = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -1,2 +1,3 @@\n from calc import add\n+# touched\n \n"


# ---- which files are tests, which issues are about tests --------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_calc.py", "test_calc.py", "src/pkg/tests/helpers.py", "pkg/test/util.py", "conftest.py", "tests/conftest.py",
        "calc_test.py", "calc_tests.py", "web/app.test.ts", "web/app.spec.jsx", "web/app.test.mjs", "pkg/util_test.go",
        "src/main/java/FooTest.java", "src/FooTests.kt", "__tests__/a.js", "spec/foo_spec.rb",
    ],
)
def test_test_files_and_files_in_test_directories_are_recognised(path):
    assert is_test_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "calc.py", "src/flask/testing.py", "src/latest.py", "contest.py", "attestation.py", "docs/testing_guide.md",
        "src/pkg/protest.py", "pkg/test_data/example.json", "src/greatest/core.py", "Makefile", "setup.py",
    ],
)
def test_source_files_with_test_like_names_are_not_taken_for_tests(path):
    assert not is_test_path(path)


@pytest.mark.parametrize(
    "title, body",
    [
        ("Add tests for the parser", ""), ("Flaky test in CI", ""), ("Improve test coverage of utils", ""),
        ("Testing helpers are missing", ""), ("Parser", "Please add a test for the empty case."),
        ("Parser", "We should write unit tests for this."), ("Parser", "It has missing tests."),
    ],
)
def test_an_issue_that_asks_for_tests_is_about_tests(title, body):
    assert issue_is_about_tests(title, body)


@pytest.mark.parametrize(
    "title, body",
    [
        ("add() returns the wrong sum", "add(2, 3) returns -1 instead of 5."),
        ("test_client drops cookies", "Repro: client.get('/')"),
        ("Crash on empty input", "The test suite passes but running `pytest` shows the traceback below."),
        ("Raise error when blueprint name contains a dot", "This is required since every dot is now significant."),
    ],
)
def test_a_bug_report_that_only_mentions_a_test_is_not_about_tests(title, body):
    assert not issue_is_about_tests(title, body)


# ---- taking a patch apart ---------------------------------------------------------------------------------------


def test_split_patch_gives_the_path_and_text_of_each_file_section():
    sections = split_patch(SOURCE_PATCH + TEST_PATCH)
    assert [path for path, _ in sections] == ["calc.py", "tests/test_calc.py"]
    assert sections[0][1] == SOURCE_PATCH and sections[1][1] == TEST_PATCH


def test_split_patch_uses_the_old_path_for_a_deleted_file_and_the_new_one_for_an_added_file():
    deleted = "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
    added = "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    assert [path for path, _ in split_patch(deleted + added)] == ["old.py", "new.py"]


def test_split_patch_reads_paths_from_a_plain_unified_diff_too_including_a_deletion_and_an_addition():
    deleted = "--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n"
    added = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    assert [path for path, _ in split_patch(deleted + added)] == ["old.py", "new.py"]


def test_split_patch_of_nothing_is_nothing():
    assert split_patch("") == []


# ---- inspect_patch against a stub sandbox -----------------------------------------------------------------------


class Stub:
    """Just enough sandbox for inspect_patch: records commands, answers by prefix, keeps written files."""

    def __init__(self, tmp_path: Path, answers: dict[str, ExecResult] | None = None):
        self.root = tmp_path
        self.written: dict[str, str] = {}
        self.log: list[str] = []
        self.answers = {"git rev-parse HEAD": ExecResult(0, "abc123def\n", "", False, 0.0), **(answers or {})}

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        self.log.append(cmd)
        for prefix, result in self.answers.items():
            if cmd.startswith(prefix) or prefix in cmd:
                return result
        return OK

    def write_file(self, path: str, content: str) -> None:
        self.written[path] = content


def test_an_empty_patch_is_a_problem_and_git_is_never_asked(tmp_path):
    stub = Stub(tmp_path)
    check = inspect_patch(stub, "", allow_tests=False)
    assert not check.ok and check.problem == "The patch is empty: no source file was changed."
    assert stub.log == [] and check.patch == ""


def test_a_patch_that_only_touches_tests_is_a_different_problem_and_says_which_files(tmp_path):
    check = inspect_patch(Stub(tmp_path), TEST_PATCH, allow_tests=False)
    assert not check.ok and check.removed_tests == ("tests/test_calc.py",) and check.patch == ""
    assert "only changes test files (tests/test_calc.py)" in check.problem and "Fix the source instead" in check.problem


def test_test_files_are_taken_out_of_a_patch_that_also_has_source_changes(tmp_path):
    stub = Stub(tmp_path)
    check = inspect_patch(stub, SOURCE_PATCH + TEST_PATCH, allow_tests=False)
    assert check.ok and check.patch == SOURCE_PATCH and check.removed_tests == ("tests/test_calc.py",)
    assert stub.written[".anvil/sanity.diff"] == SOURCE_PATCH, "what is checked is what will be delivered"


def test_when_the_issue_is_about_tests_they_stay_in_the_patch(tmp_path):
    check = inspect_patch(Stub(tmp_path), SOURCE_PATCH + TEST_PATCH, allow_tests=True)
    assert check.ok and check.patch == SOURCE_PATCH + TEST_PATCH and check.removed_tests == ()


def test_a_clean_patch_is_checked_in_a_scratch_worktree_at_the_base_commit_and_the_worktree_is_removed(tmp_path):
    stub = Stub(tmp_path)
    check = inspect_patch(stub, SOURCE_PATCH, allow_tests=False)

    assert check.ok and check.apply_checked and check.skipped == ""
    add, apply_, remove, prune = [next(c for c in stub.log if c.startswith(p)) for p in ("git worktree add", "git -C", "git worktree remove", "git worktree prune")]
    assert "--detach" in add and add.endswith("abc123def"), "at the commit HEAD is at"
    scratch = shlex.split(add)[-2]
    assert Path(scratch).name == "base" and "anvil-sanity-" in scratch and not str(scratch).startswith(str(tmp_path))
    assert f"git -C {scratch} apply --check" in apply_ and str(tmp_path / ".anvil" / "sanity.diff") in apply_
    assert remove == f"git worktree remove --force {scratch}" and not Path(scratch).parent.exists(), "removed again"
    assert stub.log.index(add) < stub.log.index(apply_) < stub.log.index(remove) < stub.log.index(prune)


def test_a_patch_that_does_not_apply_is_a_problem_with_gits_own_words(tmp_path):
    stub = Stub(tmp_path, {" apply --check": ExecResult(1, "", "error: patch failed: calc.py:2\nerror: calc.py: patch does not apply\n", False, 0.0)})
    check = inspect_patch(stub, SOURCE_PATCH, allow_tests=False)
    assert not check.ok and check.apply_checked and check.patch == SOURCE_PATCH, "still returned: it is delivered, flagged"
    assert check.problem == (
        "The patch does not apply to the original code (git apply --check): "
        "error: patch failed: calc.py:2 error: calc.py: patch does not apply"
    )
    assert any(c.startswith("git worktree remove") for c in stub.log), "cleaned up after a failure too"


def test_outside_a_git_checkout_the_apply_check_is_skipped_and_the_patch_is_accepted(tmp_path):
    stub = Stub(tmp_path, {"git rev-parse HEAD": ExecResult(128, "", "fatal: not a git repository", False, 0.0)})
    check = inspect_patch(stub, SOURCE_PATCH, allow_tests=False)
    assert check.ok and not check.apply_checked and check.skipped == "the working directory is not a git checkout"
    assert stub.log == ["git rev-parse HEAD"] and stub.written == {}


def test_a_scratch_worktree_that_cannot_be_made_skips_the_check_with_the_reason(tmp_path):
    stub = Stub(tmp_path, {"git worktree add": ExecResult(128, "", "fatal: 'x' already exists", False, 0.0)})
    check = inspect_patch(stub, SOURCE_PATCH, allow_tests=False)
    assert check.ok and not check.apply_checked
    assert "could not create a scratch worktree at the base commit: fatal: 'x' already exists" in check.skipped
    assert not any(c.startswith("git worktree remove") for c in stub.log), "nothing was added, so nothing is removed"


def test_a_file_that_cannot_be_written_only_skips_the_check(tmp_path):
    class ReadOnly(Stub):
        def write_file(self, path, content):
            raise OSError("read-only file system")

    stub = ReadOnly(tmp_path)
    check = inspect_patch(stub, SOURCE_PATCH, allow_tests=False)
    assert check.ok and not check.apply_checked and "OSError: read-only file system" in check.skipped
    assert any(c.startswith("git worktree remove") for c in stub.log), "the scratch worktree it had made is still removed"


def test_a_timeout_or_a_crash_in_git_only_skips_the_check(tmp_path):
    slow = Stub(tmp_path, {" apply --check": ExecResult(-1, "", "", True, 60.0)})
    assert inspect_patch(slow, SOURCE_PATCH, allow_tests=False).ok
    assert "could not be run" in inspect_patch(slow, SOURCE_PATCH, allow_tests=False).skipped

    class Crashing(Stub):
        def exec(self, cmd, timeout=120):
            raise OSError("no shell")

    check = inspect_patch(Crashing(tmp_path), SOURCE_PATCH, allow_tests=False)
    assert check.ok and not check.apply_checked


# ---- against real git ------------------------------------------------------------------------------------------


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", "-c", "user.email=a@b.c", "-c", "user.name=t", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


class ShellSandbox:
    """A real directory and a real shell: what WorktreeSandbox provides to inspect_patch."""

    def __init__(self, root: Path):
        self.root = root

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        done_ = subprocess.run(cmd, shell=True, cwd=self.root, capture_output=True, text=True, timeout=timeout)
        return ExecResult(done_.returncode, done_.stdout, done_.stderr, False, 0.0)

    def write_file(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)


@pytest.fixture(params=["plain repository", "linked worktree"])
def real_sandbox(request, tmp_path):
    main = tmp_path / "clone"
    main.mkdir()
    git("init", "-q", "-b", "main", cwd=main)
    (main / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (main / "tests").mkdir()
    (main / "tests" / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    git("add", "-A", cwd=main)
    git("commit", "-qm", "init", cwd=main)
    if request.param == "plain repository":
        return ShellSandbox(main), main
    work = tmp_path / "clone_sandbox"  # like the sandbox the harness uses: a linked worktree of the clone
    git("worktree", "add", "--detach", str(work), "HEAD", cwd=main)
    return ShellSandbox(work), main


def test_real_git_accepts_the_patch_the_diff_produces_and_leaves_no_scratch_worktree(real_sandbox):
    sandbox, main = real_sandbox
    (sandbox.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    patch = sandbox.exec("git diff").stdout
    check = inspect_patch(sandbox, patch, allow_tests=False)
    assert check.ok and check.apply_checked, check
    listing = git("worktree", "list", "--porcelain", cwd=main)
    assert "anvil-sanity" not in listing, "the scratch worktree was removed and pruned"


def test_real_git_refuses_a_patch_that_does_not_fit_the_base_commit(real_sandbox):
    sandbox, _ = real_sandbox
    stale = SOURCE_PATCH.replace("-    return a - b", "-    return a * b")  # its context is not what the base has
    check = inspect_patch(sandbox, stale, allow_tests=False)
    assert not check.ok and check.apply_checked
    assert "does not apply to the original code" in check.problem and "calc.py" in check.problem


def test_real_git_takes_the_test_file_out_of_a_real_diff(real_sandbox):
    sandbox, _ = real_sandbox
    (sandbox.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (sandbox.root / "tests" / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5  # ok\n")
    check = inspect_patch(sandbox, sandbox.exec("git diff").stdout, allow_tests=False)
    assert check.ok and check.removed_tests == ("tests/test_calc.py",)
    assert "diff --git a/calc.py" in check.patch and "test_calc" not in check.patch


# ---- the state -------------------------------------------------------------------------------------------------


def test_a_patch_that_failed_its_sanity_check_is_never_more_than_low_confidence():
    state = RunState("u")
    state.repro_confirmed = state.verified = True
    assert state.confidence(True) == "medium" or state.confidence(True) == "high"
    state.sanity_failed = True
    assert state.confidence(True) == "low"
    assert state.confidence(False) == "none", "and still none when there is no patch at all"


def test_the_sanity_kickoff_carries_the_reason_and_says_it_is_the_last_attempt():
    text = patch_kickoff(attempt=4, repro_cmd=None, repro_output="", feedback="The patch is empty: no source file was changed.", kind="sanity")
    assert "The patch is empty: no source file was changed." in text
    assert "checked before delivery and cannot be handed in as it is" in text
    assert "never the tests" in text and "This is your last attempt." in text


# ---- whole runs ------------------------------------------------------------------------------------------------


def git_world(check=lambda patch, n: True, rev=ExecResult(0, "abc123\n", "", False, 0.0)):
    """A fake sandbox whose shell also answers the few git commands the sanity check runs."""
    base = project_files()
    calls = {"apply": 0}

    def on_exec(cmd: str, files: dict[str, str]) -> ExecResult:
        if cmd == "git rev-parse HEAD":
            return rev
        if cmd.startswith(("git worktree add", "git worktree remove", "git worktree prune")):
            return OK
        if " apply --check " in cmd:
            calls["apply"] += 1
            if check(files.get(".anvil/sanity.diff", ""), calls["apply"]):
                return OK
            return ExecResult(1, "", "error: patch failed: calc.py:2\nerror: calc.py: patch does not apply\n", False, 0.0)
        if cmd.startswith("git checkout HEAD -- "):
            path = shlex.split(cmd)[4]
            files[path] = base[path]
            return OK
        return project_exec(cmd, files)

    return FakePipeline(FakeSandbox(project_files(), on_exec=on_exec))


def kickoffs(run) -> list[str]:
    """The distinct sanity kickoff messages the model was sent (prompt_texts repeats a message once per call that saw it)."""
    return list(dict.fromkeys(t for t in run.prompt_texts() if "checked before delivery" in t))


def sanity_line(run) -> str:
    return next(line for line in run.report.splitlines() if line.startswith("- Patch sanity:"))


def shell_log(pipeline) -> list[str]:
    return pipeline.sandbox.exec_log


def test_a_clean_patch_passes_and_the_check_ran_in_a_scratch_worktree(tmp_path):
    world = git_world()
    run = execute(happy(), tmp_path, pipeline=world)
    assert sanity_line(run) == "- Patch sanity: passed"
    assert "+    return a + b" in run.patch and run.llm.remaining == 0 and len(run.llm.calls) == 15, "no retry"
    log = shell_log(world)
    assert log.count("git rev-parse HEAD") == 1 and sum(c.startswith("git worktree add --detach") for c in log) == 1
    assert sum(" apply --check " in c for c in log) == 1 and sum(c.startswith("git worktree remove") for c in log) == 1
    assert "Confidence: **high**" in run.report


def test_the_closing_summary_is_written_knowing_the_outcome(tmp_path):
    run = execute(happy(), tmp_path, pipeline=git_world())
    last_kickoff = run.llm.calls[-1][0][-1]["content"]
    assert "- Patch sanity: passed" in last_kickoff


def test_edits_to_test_files_are_taken_out_of_the_patch_and_the_report_says_so(tmp_path):
    touch_tests = [reply(call("edit_file", path="tests/test_calc.py", old="def test_add():", new="def test_add():\n    # touched"))]
    script = understand() + localize() + reproduce() + [reply(call("edit_file", **FIX))] + touch_tests + [reply(call("run_cmd", cmd=REPRO_CMD)), done("fixed")] + verify() + review_ok() + finalize()
    world = git_world()
    run = execute(script, tmp_path, pipeline=world)

    assert "diff --git a/calc.py" in run.patch and "test_calc" not in run.patch
    assert sanity_line(run) == "- Patch sanity: passed. Changes to test files were taken out of the patch: tests/test_calc.py"
    assert "# touched" in world.sandbox.files["tests/test_calc.py"], "only the delivered patch is cleaned; the tree is the model's"
    assert len(run.llm.calls) == 16, "no retry: the source change is what matters"


def test_when_the_issue_is_about_tests_their_changes_stay(tmp_path):
    class AboutTests(FakePipeline):
        def ingest(self, *args, **kwargs):
            result = super().ingest(*args, **kwargs)
            result.issue.title = "Add tests for add()"
            return result

    world = git_world()
    tests_pipeline = AboutTests(world.sandbox)
    touch_tests = [reply(call("edit_file", path="tests/test_calc.py", old="def test_add():", new="def test_add():\n    # touched"))]
    script = understand() + localize() + reproduce() + [reply(call("edit_file", **FIX))] + touch_tests + [reply(call("run_cmd", cmd=REPRO_CMD)), done("fixed")] + verify() + review_ok() + finalize()
    run = execute(script, tmp_path, pipeline=tests_pipeline)
    assert "diff --git a/tests/test_calc.py" in run.patch
    assert sanity_line(run) == "- Patch sanity: passed"


def test_an_empty_patch_gets_one_forced_fix_retry_with_the_reason_in_the_prompt(tmp_path):
    script = understand() + localize() + reproduce() + [give_up("no idea")] + good_patch() + verify() + finalize()
    run = execute(script, tmp_path, pipeline=git_world(), max_patch_attempts=1, max_rollbacks=0)

    (kickoff,) = kickoffs(run)
    assert "The patch is empty: no source file was changed." in kickoff and "last attempt" in kickoff
    assert "+    return a + b" in run.patch and "Verified after patching: yes" in run.report
    assert sanity_line(run) == (
        "- Patch sanity: passed after one forced-fix retry (the first patch was refused: The patch is empty: no source file was changed.)"
    )
    assert run.llm.remaining == 0 and len(run.llm.calls) == 14
    assert "Patch attempts: 2" in run.report, "the retry is a patch attempt like the others"


def test_the_retry_is_the_only_one_a_patch_that_is_still_empty_ends_the_run_at_none(tmp_path):
    script = understand() + localize() + reproduce() + [give_up("no idea"), give_up("still no idea")] + finalize()
    run = execute(script, tmp_path, pipeline=git_world(), max_patch_attempts=1, max_rollbacks=0)

    assert len(kickoffs(run)) == 1, "one retry, not more"
    assert sanity_line(run) == "- Patch sanity: FAILED after one forced-fix retry: The patch is empty: no source file was changed."
    assert run.patch == "" and run.done.data["resolved_confidence"] == 0.0
    assert "The patch failed its sanity check: The patch is empty" in run.report
    assert run.done.type == "done"


def test_a_patch_that_does_not_apply_gets_the_retry_with_gits_error_and_then_passes(tmp_path):
    world = git_world(check=lambda patch, n: n >= 2)
    script = understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + [reply(call("read_file", path="calc.py")), done("adjusted")] + verify() + finalize()
    run = execute(script, tmp_path, pipeline=world)

    (kickoff,) = kickoffs(run)
    assert "does not apply to the original code (git apply --check): error: patch failed: calc.py:2" in kickoff
    assert sanity_line(run).startswith("- Patch sanity: passed after one forced-fix retry (the first patch was refused: The patch does not apply")
    assert sum(" apply --check " in c for c in shell_log(world)) == 2 and run.llm.remaining == 0


def test_a_patch_that_still_does_not_apply_after_the_retry_is_delivered_flagged_and_capped_at_low(tmp_path):
    world = git_world(check=lambda patch, n: False)
    script = understand() + localize() + reproduce() + good_patch() + verify() + review_ok() + [reply(call("read_file", path="calc.py")), done("adjusted")] + verify() + finalize()
    run = execute(script, tmp_path, pipeline=world)

    assert sanity_line(run).startswith("- Patch sanity: FAILED after one forced-fix retry: The patch does not apply to the original code")
    assert "+    return a + b" in run.patch, "it is still handed in: the reader decides"
    assert "Confidence: **low**" in run.report and run.done.data["resolved_confidence"] == pytest.approx(0.3)
    assert sum(" apply --check " in c for c in shell_log(world)) == 2, "checked before and after the one retry, never a third time"


def test_a_patch_that_only_edits_tests_is_reverted_and_gets_the_retry(tmp_path):
    tests_only = [reply(call("edit_file", path="tests/test_calc.py", old="def test_add():", new="def test_add():\n    pass")), done("tests only")]
    script = understand() + localize() + reproduce() + tests_only + good_patch() + verify() + finalize()
    world = git_world()
    run = execute(script, tmp_path, pipeline=world, max_patch_attempts=1, max_rollbacks=0)

    (kickoff,) = kickoffs(run)
    assert "only changes test files (tests/test_calc.py)" in kickoff
    assert any(c.startswith("git checkout HEAD -- tests/test_calc.py") for c in shell_log(world)), "the test edit was undone first"
    assert "pass" not in world.sandbox.files["tests/test_calc.py"] and "+    return a + b" in run.patch
    assert sanity_line(run).startswith("- Patch sanity: passed after one forced-fix retry (the first patch was refused: The patch only changes test files")


def test_outside_a_git_checkout_the_patch_is_delivered_and_the_report_says_the_check_was_skipped(tmp_path):
    world = git_world(rev=ExecResult(128, "", "fatal: not a git repository", False, 0.0))
    run = execute(happy(), tmp_path, pipeline=world)
    assert sanity_line(run) == "- Patch sanity: passed. git apply --check was skipped: the working directory is not a git checkout"
    assert "+    return a + b" in run.patch and len(run.llm.calls) == 15


def test_a_run_the_budget_stopped_is_checked_without_a_retry(tmp_path):
    run = execute(understand() + localize() + reproduce(), tmp_path, pipeline=git_world(), max_total_steps=5)
    assert run.patch == ""
    assert sanity_line(run).startswith("- Patch sanity: FAILED: The patch is empty")
    assert "(no retry: the run had already stopped: budget)" in sanity_line(run)
    assert kickoffs(run) == []


def test_a_budget_that_runs_out_during_the_retry_ends_the_run_as_a_budget_stop_and_the_report_says_the_retry_was_cut_short(tmp_path):
    script = understand() + localize() + reproduce() + [give_up("no idea")]
    run = execute(script, tmp_path, pipeline=git_world(), max_patch_attempts=1, max_rollbacks=0, max_total_steps=8)

    assert "(the retry was cut short: the run stopped: budget)" in sanity_line(run)
    assert sanity_line(run).startswith("- Patch sanity: FAILED after one forced-fix retry: The patch is empty")
    assert "step budget exhausted" in run.report, "the run stopped for the budget, as it would anywhere else"
    assert run.done.type == "done" and run.patch == ""


def test_the_check_is_only_ever_a_safeguard_an_unexpected_error_in_it_does_not_lose_the_run(tmp_path, monkeypatch):
    import anvil.agent.orchestrator as orchestrator

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "inspect_patch", boom)
    run = execute(happy(), tmp_path, pipeline=git_world())
    assert sanity_line(run) == "- Patch sanity: not run (RuntimeError: boom)"
    assert "+    return a + b" in run.patch and run.done.type == "done"


def test_with_the_switch_off_nothing_is_checked_and_test_edits_stay(tmp_path):
    touch_tests = [reply(call("edit_file", path="tests/test_calc.py", old="def test_add():", new="def test_add():\n    # touched"))]
    script = understand() + localize() + reproduce() + [reply(call("edit_file", **FIX))] + touch_tests + [reply(call("run_cmd", cmd=REPRO_CMD)), done("fixed")] + verify() + review_ok() + finalize()
    world = git_world()
    run = execute(script, tmp_path, pipeline=world, features={"patch_sanity": False})

    assert "Patch sanity" not in run.report
    assert "diff --git a/tests/test_calc.py" in run.patch, "the patch is exactly what it was before the feature"
    assert not [c for c in shell_log(world) if c.startswith("git ")], "and git was never asked anything"
