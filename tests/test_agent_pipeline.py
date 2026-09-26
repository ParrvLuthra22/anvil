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
from tests.offline import no_base_ref_lookup  # noqa: F401 (autouse: no GitHub lookup)

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
    monkeypatch.setattr(pipeline_module, "repo_map", lambda root, max_chars=6000, label=None: f"map of {label} at {root.name}")
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
    assert workspace.repo_map == "map of acme/calc at repo"
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


def test_the_repo_map_header_names_the_repository_not_the_clone_directory(tmp_path):
    """The real profile_repo and repo_map, on a small repository in an ugly temp directory name."""
    root = tmp_path / "acme__calc__7__1790000000" / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    pipeline = RepoPipeline(
        {}, tmp_path, sandbox_factory=lambda config, root, profile: object(), registry_factory=lambda profile: object(),
        deps_installer=no_deps,
    )
    workspace = pipeline.profile(Ingested(IssueRef("acme", "calc", 7, URL), root))

    assert workspace.repo_map.splitlines()[0] == "# repo: acme/calc"
    assert "acme__calc" not in workspace.repo_map and "1790000000" not in workspace.repo_map
    assert "calc.py" in workspace.repo_map


# ---- the size of the repository map (features.token_budgets) ------------------------------------------------


def workspace_for(tmp_path, config, monkeypatch=None, recorded=None):
    """PROFILE with the real profile_repo on a tiny repository; ``recorded`` (a list) captures repo_map's arguments."""
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    if recorded is not None:
        real = pipeline_module.repo_map

        def spy(root, max_chars=6000, label=None):
            recorded.append({"max_chars": max_chars, "label": label})
            return real(root, max_chars=max_chars, label=label)

        monkeypatch.setattr(pipeline_module, "repo_map", spy)
    pipeline = RepoPipeline(
        config, tmp_path, sandbox_factory=lambda config, root, profile: object(), registry_factory=lambda profile: object(),
        deps_installer=no_deps,
    )
    return pipeline.profile(Ingested(IssueRef("acme", "calc", 7, URL), root))


def test_the_repo_map_is_3000_characters_by_default(tmp_path, monkeypatch):
    seen: list = []
    workspace_for(tmp_path, {}, monkeypatch, seen)
    assert seen == [{"max_chars": 3000, "label": "acme/calc"}]


def test_with_token_budgets_off_the_map_keeps_its_own_default_of_6000(tmp_path, monkeypatch):
    seen: list = []
    workspace_for(tmp_path, {"features": {"token_budgets": False}}, monkeypatch, seen)
    assert seen[0]["max_chars"] == 6000


def test_the_size_can_be_set_in_the_config(tmp_path, monkeypatch):
    seen: list = []
    workspace_for(tmp_path, {"token_saving": {"repo_map_chars": 1500}}, monkeypatch, seen)
    assert seen[0]["max_chars"] == 1500


def test_a_config_that_does_not_validate_still_gives_a_map_of_the_default_size(tmp_path, monkeypatch):
    seen: list = []
    workspace_for(tmp_path, {"features": "nonsense", "token_saving": {"repo_map_chars": "big"}}, monkeypatch, seen)
    assert seen[0]["max_chars"] == 3000, "the run must not fail in PROFILE over a bad optional setting"


def test_the_real_map_of_a_big_repository_fits_the_size(tmp_path):
    root = tmp_path / "repo"
    for i in range(400):
        package = root / "src" / f"pkg_{i % 20}"
        package.mkdir(parents=True, exist_ok=True)
        (package / f"module_{i}.py").write_text("".join(f"def function_{i}_{j}(x):\n    return x\n" for j in range(6)))
    small = workspace_for(tmp_path, {}).repo_map
    big = workspace_for(tmp_path, {"features": {"token_budgets": False}}).repo_map
    assert len(small) <= 3000 and small.startswith("# repo: acme/calc")
    assert 3000 < len(big) <= 6000, "the full map is the larger one"
