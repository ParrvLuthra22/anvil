"""INGEST and PROFILE: everything that touches the network, git or the disk before the model starts.

The orchestrator only sees the ``Pipeline`` protocol, so tests substitute a fake and
never clone anything. ``RepoPipeline`` is the real one, built from the contract
functions in ``anvil.repo``; the sandbox backend and the default tool set come from
factories that are imported lazily, so this module loads even where those parts of
the codebase are not merged yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from anvil.repo.ingest import IssueRef, clone_repo, fetch_issue, parse_issue_url
from anvil.repo.profile import RepoProfile, profile_repo, repo_map
from anvil.sandbox.base import Sandbox
from anvil.tools.registry import ToolRegistry

SandboxFactory = Callable[[dict, Path, RepoProfile], Sandbox]
RegistryFactory = Callable[[RepoProfile], ToolRegistry]


@dataclass
class Ingested:
    """Result of INGEST: the issue with its text, and the local clone of its repository."""

    issue: IssueRef
    repo_root: Path


@dataclass
class Workspace:
    """Result of PROFILE: everything the LLM phases work with."""

    issue: IssueRef
    profile: RepoProfile
    repo_map: str
    sandbox: Sandbox
    tools: ToolRegistry


class Pipeline(Protocol):
    """The two setup phases, as the orchestrator sees them. Either may raise; the orchestrator copes."""

    def ingest(self, issue_url: str) -> Ingested:
        """Resolve ``issue_url`` to the issue text and a local clone of the repository."""
        ...

    def profile(self, ingested: Ingested) -> Workspace:
        """Profile the clone and open the sandbox and tool set the LLM phases will use."""
        ...


class RepoPipeline:
    """The real pipeline: GitHub API, ``git clone``, repo profiling, then sandbox and tools.

    Clones go to ``<output_dir>/workspace/<owner>__<repo>__<number>__<timestamp>``, so
    repeated runs never collide and ``make clean`` removes them with the rest of ``output/``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: Path,
        *,
        sandbox_factory: SandboxFactory | None = None,
        registry_factory: RegistryFactory | None = None,
    ) -> None:
        self._config = config
        self._workspace_dir = output_dir / "workspace"
        self._sandbox_factory = sandbox_factory or _default_sandbox
        self._registry_factory = registry_factory or _default_registry

    def ingest(self, issue_url: str) -> Ingested:
        """Parse the URL, fetch the issue over the GitHub API and shallow-clone its repository."""
        issue = fetch_issue(parse_issue_url(issue_url))
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        dest = self._workspace_dir / f"{issue.owner}__{issue.repo}__{issue.number}__{int(time.time())}"
        return Ingested(issue, clone_repo(issue, dest))

    def profile(self, ingested: Ingested) -> Workspace:
        """Detect languages and commands, map the repo, then open the sandbox and the tool registry."""
        root = ingested.repo_root
        profile = profile_repo(root)
        return Workspace(
            issue=ingested.issue,
            profile=profile,
            repo_map=repo_map(root),
            sandbox=self._sandbox_factory(self._config, root, profile),
            tools=self._registry_factory(profile),
        )


def _default_sandbox(config: dict, repo_root: Path, profile: RepoProfile) -> Sandbox:
    try:
        from anvil.sandbox import make_sandbox
    except ImportError:
        raise RuntimeError("no sandbox backend is available: anvil.sandbox.make_sandbox is missing") from None
    return make_sandbox(config, repo_root, profile)


def _default_registry(profile: RepoProfile) -> ToolRegistry:
    try:
        from anvil.tools.registry import make_default_registry
    except ImportError:
        raise RuntimeError("no tools are available: anvil.tools.registry.make_default_registry is missing") from None
    return make_default_registry(profile)
