"""The harness against a real git repository, the real worktree sandbox and the real tools, with a relative output_dir.

``make run`` uses the default ``output_dir: output``, which is relative. These tests reproduce that setup:
the audit found that with it the sandbox was an empty directory, so the model could see no code at all.
Every test here failed on the code before ``RepoPipeline`` resolved its paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.agent.pipeline import RepoPipeline
from anvil.sandbox import make_sandbox
from tests import real_repo as rr


@pytest.fixture
def toy(tmp_path, monkeypatch):
    return rr.prepare(tmp_path, monkeypatch)


# ---- the pipeline itself --------------------------------------------------------------------


def test_a_relative_output_dir_yields_a_sandbox_that_can_see_the_repository(toy, tmp_path):
    """The audit's minimal reproduction: relative output_dir -> read_file('calc.py') raised FileNotFoundError."""
    pipeline = RepoPipeline({"sandbox": "worktree", "install_dependencies": False}, Path("out"))
    ingested = pipeline.ingest(rr.ISSUE_URL)
    workspace = pipeline.profile(ingested)
    try:
        assert ingested.repo_root.is_absolute() and workspace.sandbox.root.is_absolute()
        assert workspace.sandbox.read_file("calc.py") == rr.BUGGY
        assert "calc.py" in workspace.repo_map
        assert workspace.sandbox.diff() == ""
        ref = workspace.sandbox.checkpoint("x")
        assert ref and not ref.startswith("error"), "git must work in the worktree: it has to be a real repository"
    finally:
        workspace.sandbox.close()
    assert (tmp_path / "out" / "workspace").is_dir(), "the clone lives under the relative output_dir, resolved against the cwd"


def test_the_clone_and_the_sandbox_are_addressed_by_absolute_paths_whatever_the_cwd(toy, tmp_path):
    pipeline = RepoPipeline({"sandbox": "worktree"}, Path("out"))
    ingested = pipeline.ingest(rr.ISSUE_URL)
    assert str(ingested.repo_root).startswith(str((tmp_path / "out").resolve()))


def test_a_relative_repo_root_handed_to_profile_is_resolved_too(toy, tmp_path, monkeypatch):
    pipeline = RepoPipeline({"sandbox": "worktree"}, tmp_path / "out")
    ingested = pipeline.ingest(rr.ISSUE_URL)
    from dataclasses import replace

    relative = replace(ingested, repo_root=ingested.repo_root.relative_to(tmp_path))
    workspace = pipeline.profile(relative)
    try:
        assert workspace.sandbox.read_file("calc.py") == rr.BUGGY
    finally:
        workspace.sandbox.close()


# ---- whole runs -----------------------------------------------------------------------------


def test_happy_path_with_a_relative_output_dir(toy):
    run = rr.run_real(rr.understand() + rr.localize() + rr.reproduce() + rr.good_patch() + rr.verify() + rr.review_ok() + rr.finalize())

    assert run.errors() == []
    assert run.llm.remaining == 0
    assert "+    return a + b" in run.patch and "-    return a - b" in run.patch and ".anvil" not in run.patch
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)
    assert "Verified after patching: yes" in run.report and "Review: approved" in run.report
    texts = " ".join(run.prompt_texts())
    assert "File not found" not in texts and "no baseline" not in texts, "the model must see the code, not an empty tree"
    assert "def add(a, b):" in texts
    assert Path("out/patch.diff").exists() and Path("out/report.md").exists(), "outputs land under the relative dir"


def test_rollback_with_a_relative_output_dir_really_resets_the_tree(toy):
    script = (
        rr.understand() + rr.localize() + rr.reproduce()
        + rr.wrong_patch("return a * b")
        + rr.good_patch()  # this edit only applies if the rollback restored `return a - b`
        + rr.verify() + rr.review_ok() + rr.finalize()
    )
    run = rr.run_real(script, max_patch_attempts=1, max_rollbacks=1)

    assert run.errors() == ["rollback"]
    assert "+    return a + b" in run.patch and "a * b" not in run.patch
    assert "rollbacks: 1" in run.report and "Patch attempts: 2" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.9)
    rethink = next(t for t in run.prompt_texts() if "DIFFERENT hypothesis" in t)
    assert "used 'return a * b'" in rethink


def test_rework_with_a_relative_output_dir_keeps_the_verified_patch_and_adds_the_requested_change(toy):
    """Regression (audit): the pre-rework checkpoint emptied the tree, so the rework ran without the fix and failed."""
    script = (
        rr.understand() + rr.localize() + rr.reproduce() + rr.good_patch() + rr.verify()
        + rr.review_changes() + rr.rework() + rr.verify() + rr.finalize()
    )
    run = rr.run_real(script)

    assert "+    return a + b" in run.patch, "the verified fix must survive the rework round"
    assert '"""Add two numbers."""' in run.patch, "and the reviewer's requested change must be in it"
    assert run.llm.remaining == 0
    assert "reviewer's requested rework failed verification" not in run.report
    assert "Verified after patching: yes" in run.report and "Review: changes requested" in run.report
    assert run.done.data["resolved_confidence"] == pytest.approx(0.6)
    assert not [line for line in run.report.splitlines() if line.startswith("- FAIL")], run.report
    assert set(run.errors()) <= {"sandbox"}, run.errors()
