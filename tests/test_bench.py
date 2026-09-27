import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from bench.score import _score_instance, _test_argv
from bench.run_bench import _load_completed, _load_instances
from bench.run_bench import _should_retry_rate_limit
from tests.mock_llm import MockLLM
from anvil.agent.orchestrator import run_harness
from anvil.events import EventBus
from anvil.llm.client import LLMResponse
from anvil.repo.deps import DepsResult


def _stub_score_venv(sandbox, profile, *, as_of=None):
    """Create an offline test venv with access to the test suite's pytest."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(sandbox.root / ".anvil_venv")],
        check=True,
        capture_output=True,
        text=True,
    )
    site_packages = sandbox.root / ".anvil_venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site_packages / "anvil-test-site.pth").write_text(
        str(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"),
        encoding="utf-8",
    )
    return DepsResult(ok=True, report="offline test venv", venv_python=".anvil_venv/bin/python")


def test_test_argv_uses_the_instance_interpreter(tmp_path: Path):
    interpreter = tmp_path / "instance" / ".anvil_venv" / "bin" / "python"
    argv = _test_argv(
        ".anvil_venv/bin/python -m pytest -x -q --tb=short tests/test_x.py::test_a",
        ["tests/test_x.py::test_b"],
        interpreter,
    )
    assert argv == [str(interpreter), "-m", "pytest", "-x", "-q", "--tb=short", "tests/test_x.py::test_b"]


@pytest.mark.parametrize("runner", ["python", "python3", "python3.13", "python3.14", "/opt/python3.14/bin/python3.14"])
def test_test_argv_replaces_recorded_python_with_ensure_deps_interpreter(tmp_path: Path, runner: str):
    """Run oracle tests with ensure_deps' selected interpreter, not the recorded host Python."""
    interpreter = tmp_path / "instance" / ".anvil_venv" / "bin" / "python"
    argv = _test_argv(f"{runner} -m pytest -q tests/test_x.py::test_a", ["tests/test_x.py::test_b"], interpreter)

    assert argv == [str(interpreter), "-m", "pytest", "-q", "tests/test_x.py::test_b"]


def test_score_instance_resolved(tmp_path: Path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    (repo_dir / "main.py").write_text("def is_true(): return False\n", encoding="utf-8")
    (repo_dir / "test_main.py").write_text(
        "from main import is_true\ndef test_is_true(): assert is_true() == False\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
    sha = proc.stdout.strip()

    instance = {
        "instance_id": "test_fixture_001",
        "repo": f"file://{repo_dir.resolve()}",
        "base_commit": sha,
        "problem_statement": "Make is_true return True.",
        "test_patch": "--- a/test_main.py\n+++ b/test_main.py\n@@ -1,2 +1,2 @@\n from main import is_true\n-def test_is_true(): assert is_true() == False\n+def test_is_true(): assert is_true() == True\n",
        "FAIL_TO_PASS": ["test_main.py::test_is_true"],
        "PASS_TO_PASS": [],
    }

    patch_path = tmp_path / "patch.diff"
    patch_path.write_text("--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-def is_true(): return False\n+def is_true(): return True\n", encoding="utf-8")

    run_result = {
        "instance_id": "test_fixture_001",
        "patch_path": str(patch_path),
        "trace_path": "fake",
        "error": "",
    }

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    monkeypatch.setattr("anvil.repo.deps.ensure_deps", _stub_score_venv)
    result = _score_instance(instance, run_result, work_dir)
    assert result["resolved"] is True, (result.get("category"), result.get("error"), result.get("test_output"))
    assert result["category"] == "resolved"


def test_score_instance_empty_patch(tmp_path: Path):
    instance = {
        "instance_id": "test_empty",
        "repo": "file:///nonexistent",
        "base_commit": "HEAD",
    }
    run_result = {"instance_id": "test_empty", "patch_path": str(tmp_path / "missing.diff")}
    result = _score_instance(instance, run_result, tmp_path)
    assert result["resolved"] is False
    assert result["category"] == "no_patch"


def test_score_instance_patch_failed(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    (repo_dir / "main.py").write_text("def a(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    instance = {
        "instance_id": "test_bad_patch",
        "repo": f"file://{repo_dir.resolve()}",
        "base_commit": sha,
    }
    # Corrupt patch
    patch_path = tmp_path / "corrupt.diff"
    patch_path.write_text("--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-xxx\n+yyy\n", encoding="utf-8")

    run_result = {"instance_id": "test_bad_patch", "patch_path": str(patch_path)}
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    result = _score_instance(instance, run_result, work_dir)
    assert result["resolved"] is False
    assert result["category"] == "patch_does_not_apply"


def test_score_instance_syntax_error(tmp_path: Path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    (repo_dir / "main.py").write_text("def a(): pass\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    instance = {
        "instance_id": "test_syntax",
        "repo": f"file://{repo_dir.resolve()}",
        "base_commit": sha,
    }
    patch_path = tmp_path / "syntax.diff"
    patch_path.write_text("--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-def a(): pass\n+def a() syntax error here\n", encoding="utf-8")

    run_result = {"instance_id": "test_syntax", "patch_path": str(patch_path)}
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setattr("anvil.repo.deps.ensure_deps", _stub_score_venv)
    result = _score_instance(instance, run_result, work_dir)
    assert result["resolved"] is False
    assert result["category"] == "patch_does_not_apply"


def test_score_instance_regression(tmp_path: Path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    (repo_dir / "main.py").write_text("def a(): return 1\ndef b(): return 2\n", encoding="utf-8")
    (repo_dir / "test_main.py").write_text(
        "from main import a, b\ndef test_a(): assert a() == 1\ndef test_b(): assert b() == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True).stdout.strip()

    instance = {
        "instance_id": "test_reg",
        "repo": f"file://{repo_dir.resolve()}",
        "base_commit": sha,
        "PASS_TO_PASS": ["test_main.py::test_b"],
        "FAIL_TO_PASS": ["test_main.py::test_a"],
    }
    # Patch breaks b()
    patch_path = tmp_path / "break_b.diff"
    patch_path.write_text("--- a/main.py\n+++ b/main.py\n@@ -1,2 +1,2 @@\n def a(): return 1\n-def b(): return 2\n+def b(): return 0\n", encoding="utf-8")

    run_result = {"instance_id": "test_reg", "patch_path": str(patch_path)}
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setattr("anvil.repo.deps.ensure_deps", _stub_score_venv)
    result = _score_instance(instance, run_result, work_dir)
    assert result["resolved"] is False
    assert result["category"] == "p2p_regression"


def test_run_bench_helpers(tmp_path: Path):
    instances_file = tmp_path / "inst.json"
    instances_file.write_text(json.dumps([{"instance_id": "inst_1"}]), encoding="utf-8")
    loaded = _load_instances(instances_file)
    assert len(loaded) == 1
    assert loaded[0]["instance_id"] == "inst_1"

    results_file = tmp_path / "res.jsonl"
    results_file.write_text(json.dumps({"instance_id": "inst_1"}) + "\n", encoding="utf-8")
    completed = _load_completed(results_file)
    assert "inst_1" in completed


def test_curated_benchmark_instances_include_checkout_dates_and_oracles():
    """Curated SWE-bench records retain revisions, date, and official test oracle."""
    instances = json.loads((Path(__file__).parents[1] / "bench/instances.json").read_text())
    by_id = {row["instance_id"]: row for row in instances}

    assert {"pallets__flask-4045", "pallets__flask-4992"} <= set(by_id)
    for instance_id in ("pallets__flask-4045", "pallets__flask-4992"):
        row = by_id[instance_id]
        assert len(row["base_commit"]) == 40
        datetime.fromisoformat(row["base_commit_date"].replace("Z", "+00:00"))
        assert row["environment_setup_commit"]
        assert row["test_patch"].startswith("diff --git ")
        assert row["FAIL_TO_PASS"]
        assert row["PASS_TO_PASS"]


def test_run_bench_does_not_retry_healthy_summary_containing_429(tmp_path: Path):
    """Token totals may contain 429; healthy process exit must never retry."""
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"ts":1,"type":"done","phase":"finalize","data":{"tokens":142930}}\n',
        encoding="utf-8",
    )
    summary = "Finished instance in 4.2s. Error: \ntokens=142930"

    assert "429" in summary
    assert _should_retry_rate_limit(0, trace) is False


def test_run_bench_retries_failed_run_with_trace_error(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"ts":1,"type":"error","phase":null,"data":{"kind":"llm_error","message":"limited"}}\n',
        encoding="utf-8",
    )

    assert _should_retry_rate_limit(1, trace) is True


def test_run_bench_does_not_retry_failed_run_without_trace_error(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"ts":1,"type":"error","phase":null,"data":{"kind":"internal","message":"failed"}}\n',
        encoding="utf-8",
    )

    assert _should_retry_rate_limit(1, trace) is False


def test_run_bench_retries_llm_error_with_http_429(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"ts":1,"type":"error","phase":"understand","data":{"kind":"llm","message":"HTTP 429: Rate limit exceeded"}}\n',
        encoding="utf-8",
    )

    # The harness catches provider failures and still finalizes with exit 0.
    assert _should_retry_rate_limit(0, trace) is True


def test_mock_llm_and_fake_fixture_integration(tmp_path: Path, monkeypatch):
    """Test MockLLM running the full pipeline on a fake fixture and getting scored correctly."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True)
    (repo_dir / "main.py").write_text("def add(a, b): return a - b\n", encoding="utf-8")
    (repo_dir / "test_main.py").write_text("from main import add\ndef test_add(): assert add(2, 3) == 5\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)
    sha = proc.stdout.strip()

    def _msg(text="", tool=None, args=None):
        return LLMResponse(
            text=text,
            tool_calls=[{"id": "call_1", "tool": tool, "args": args}] if tool else [],
            usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
        )

    # 1 ingest -> 2 profile -> 3 understand -> 4 localize -> 5 reproduce -> 6 patch -> 7 verify -> 8 review -> 9 finalize
    script = [
        _msg("issue read", "phase_done", {"name": "ingest"}),
        _msg("profiled", "phase_done", {"name": "profile"}),
        _msg("understood", "phase_done", {"name": "understand"}),
        _msg("localized", "phase_done", {"name": "localize"}),
        _msg("writing repro", "write_repro", {"path": "repro.py", "content": "import sys; sys.path.insert(0, '.')\nfrom main import add\nassert add(2,3)==5\n"}),
        _msg("running repro", "run_cmd", {"cmd": "python3 .anvil/repro.py"}),
        _msg("reproduced", "phase_done", {"name": "reproduce", "repro_cmd": "python3 .anvil/repro.py"}),
        _msg("editing", "edit_file", {"path": "main.py", "old": "def add(a, b): return a - b", "new": "def add(a, b): return a + b"}),
        _msg("running repro", "run_cmd", {"cmd": "python3 .anvil/repro.py"}),
        _msg("patched", "phase_done", {"name": "patch"}),
        _msg("testing", "run_tests", {"target": "test_main.py"}),
        _msg("verified", "phase_done", {"name": "verify"}),
        _msg("diffing", "git_diff", {}),
        _msg("reviewed", "phase_done", {"name": "review"}),
        _msg("all done", "phase_done", {"name": "finalize"}),
    ]

    mock_llm = MockLLM(script)
    bus = EventBus()
    out_dir = tmp_path / "out"

    config = {"output_dir": str(out_dir), "sandbox": "worktree", "max_total_steps": 20}
    with patch("anvil.agent.pipeline._owner_and_repo", return_value=("fake", "repo")), \
         patch("anvil.agent.pipeline.clone_repo", return_value=repo_dir):
        run_harness(
            "",
            config,
            bus,
            llm=mock_llm,
            repo_url=f"file://{repo_dir.resolve()}",
            issue_text="add() subtracts",
            git_ref=sha,
        )

    # Check outputs
    assert (out_dir / "patch.diff").exists()
    assert (out_dir / "report.md").exists()

    # Score it!
    instance = {
        "instance_id": "test_fixture_001",
        "repo": f"file://{repo_dir.resolve()}",
        "base_commit": sha,
        "problem_statement": "add() subtracts",
        "test_patch": "",
        "FAIL_TO_PASS": ["test_main.py::test_add"],
        "PASS_TO_PASS": [],
    }

    run_result = {
        "instance_id": "test_fixture_001",
        "patch_path": str(out_dir / "patch.diff"),
        "trace_path": "fake",
        "error": "",
    }

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    monkeypatch.setattr("anvil.repo.deps.ensure_deps", _stub_score_venv)
    result = _score_instance(instance, run_result, work_dir)
    assert result["resolved"] is True, result
    assert result["category"] == "resolved"
