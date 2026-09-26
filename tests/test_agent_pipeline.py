"""RepoPipeline wiring: the repo/sandbox/tool functions are faked, so nothing is cloned or fetched."""

import sys
import types
from pathlib import Path

import pytest

import anvil.agent.pipeline as pipeline_module
from anvil.agent.prepared_sandbox import PreparedSandbox
from anvil.agent.pipeline import Ingested, RepoPipeline
from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile

URL = "https://github.com/acme/calc/issues/7"


def no_deps(sandbox, profile):
    """A dependency installer that does nothing, so no test here ever runs pip."""
    return types.SimpleNamespace(ok=True, report="", venv_python=None)

PROFILE = RepoProfile(["python"], "python", "pip install -e .", "pytest", "pytest")


@pytest.fixture
def repo_functions(monkeypatch):
    """Replace the anvil.repo contract functions with recording fakes; returns the call log."""
    calls: list[tuple] = []

    def parse(url):
        calls.append(("parse", url))
        return IssueRef("acme", "calc", 7, url)

    def fetch(ref):
        calls.append(("fetch", ref.number))
        return IssueRef(ref.owner, ref.repo, ref.number, ref.url, title="add is wrong", body="details")

    def clone(ref, dest):
        calls.append(("clone", dest))
        dest.mkdir(parents=True)
        return dest / "repo"

    monkeypatch.setattr(pipeline_module, "parse_issue_url", parse)
    monkeypatch.setattr(pipeline_module, "fetch_issue", fetch)
    monkeypatch.setattr(pipeline_module, "clone_repo", clone)
    monkeypatch.setattr(pipeline_module, "profile_repo", lambda root: calls.append(("profile", root)) or PROFILE)
    monkeypatch.setattr(pipeline_module, "repo_map", lambda root: f"map of {root.name}")
    return calls


def test_ingest_parses_fetches_then_clones_into_a_unique_workspace_dir(tmp_path, repo_functions):
    ingested = RepoPipeline({}, tmp_path).ingest(URL)

    assert [c[0] for c in repo_functions] == ["parse", "fetch", "clone"]
    assert ingested.issue.title == "add is wrong", "the fetched issue, not the parsed one, is returned"
    dest = repo_functions[2][1]
    assert dest.parent == tmp_path / "workspace"
    assert dest.name.startswith("acme__calc__7__")
    assert ingested.repo_root == dest / "repo"


def test_two_ingests_never_share_a_clone_directory(tmp_path, repo_functions, monkeypatch):
    times = iter([1000, 1001])
    monkeypatch.setattr(pipeline_module.time, "time", lambda: next(times))
    pipeline = RepoPipeline({}, tmp_path)
    first, second = pipeline.ingest(URL), pipeline.ingest(URL)
    assert first.repo_root != second.repo_root


def test_profile_builds_the_workspace_from_the_injected_factories(tmp_path, repo_functions):
    seen = {}
    sandbox, registry = object(), object()

    def make_sandbox(config, root, profile):
        seen["sandbox"] = (config, root, profile)
        return sandbox

    def make_registry(profile):
        seen["registry"] = profile
        return registry

    config = {"sandbox": "worktree"}
    pipeline = RepoPipeline(config, tmp_path, sandbox_factory=make_sandbox, registry_factory=make_registry, deps_installer=no_deps)
    issue = IssueRef("acme", "calc", 7, URL, title="t")
    workspace = pipeline.profile(Ingested(issue, tmp_path / "repo"))

    assert workspace.issue is issue and workspace.profile is PROFILE
    assert workspace.repo_map == "map of repo"
    assert isinstance(workspace.sandbox, PreparedSandbox) and workspace.sandbox.inner is sandbox
    assert workspace.tools is registry
    assert seen["sandbox"] == (config, tmp_path / "repo", PROFILE)
    assert seen["registry"] is PROFILE


def test_default_factories_delegate_to_the_sandbox_and_tools_packages(tmp_path, repo_functions, monkeypatch):
    sandbox_pkg = types.ModuleType("anvil.sandbox")
    sandbox_pkg.make_sandbox = lambda config, root, profile: ("sandbox", config, root, profile)
    monkeypatch.setitem(sys.modules, "anvil.sandbox", sandbox_pkg)
    monkeypatch.setattr("anvil.tools.registry.make_default_registry", lambda profile: ("registry", profile), raising=False)

    workspace = RepoPipeline({"k": 1}, tmp_path, deps_installer=no_deps).profile(Ingested(IssueRef("a", "b", 1, URL), Path("/r")))

    assert workspace.sandbox.inner == ("sandbox", {"k": 1}, Path("/r"), PROFILE)
    assert workspace.tools == ("registry", PROFILE)


def test_a_missing_sandbox_backend_is_a_clear_error(tmp_path, repo_functions, monkeypatch):
    monkeypatch.setitem(sys.modules, "anvil.sandbox", types.ModuleType("anvil.sandbox"))  # no make_sandbox
    monkeypatch.setattr("anvil.tools.registry.make_default_registry", lambda profile: object(), raising=False)
    with pytest.raises(RuntimeError, match="no sandbox backend"):
        RepoPipeline({}, tmp_path, deps_installer=no_deps).profile(Ingested(IssueRef("a", "b", 1, URL), Path("/r")))


def test_a_missing_default_tool_set_is_a_clear_error(tmp_path, repo_functions, monkeypatch):
    monkeypatch.delattr("anvil.tools.registry.make_default_registry", raising=False)
    pipeline = RepoPipeline({}, tmp_path, sandbox_factory=lambda config, root, profile: object(), deps_installer=no_deps)
    with pytest.raises(RuntimeError, match="no tools are available"):
        pipeline.profile(Ingested(IssueRef("a", "b", 1, URL), Path("/r")))
