"""INGEST and PROFILE: everything that touches the network, git or the disk before the model starts.

The orchestrator only sees the ``Pipeline`` protocol, so tests substitute a fake and
never clone anything. ``RepoPipeline`` is the real one, built from the contract
functions in ``anvil.repo``; the sandbox backend and the default tool set come from
factories that are imported lazily, so this module loads even where those parts of
the codebase are not merged yet.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from anvil.agent.prepared_sandbox import PreparedSandbox
from anvil.repo.ingest import IssueRef, clone_repo, fetch_issue, parse_issue_url
from anvil.repo.profile import RepoProfile, profile_repo, repo_map
from anvil.sandbox.base import Sandbox
from anvil.tools.registry import ToolRegistry

_REPO_URL = re.compile(r"^https?://(?:www\.)?github\.com/(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$", re.IGNORECASE)
_TITLE_CHARS = 100

SandboxFactory = Callable[[dict, Path, RepoProfile], Sandbox]
RegistryFactory = Callable[[RepoProfile], ToolRegistry]


class IssueFetchError(RuntimeError):
    """GitHub did not give us the issue (rate limit, not found, network), so there is nothing to work from."""


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

    def ingest(self, issue_url: str, *, repo_url: str | None = None, issue_text: str | None = None) -> Ingested:
        """Resolve ``issue_url`` to the issue text and a local clone of the repository.

        ``issue_text`` (with ``repo_url``, or an ``issue_url`` that names the repository) replaces the
        GitHub fetch: it is used as the issue body.
        """
        ...

    def profile(self, ingested: Ingested) -> Workspace:
        """Profile the clone and open the sandbox and tool set the LLM phases will use."""
        ...


class RepoPipeline:
    """The real pipeline: GitHub API, ``git clone``, repo profiling, then sandbox and tools.

    Clones go to ``<output_dir>/workspace/<owner>__<repo>__<number>__<timestamp>``, so
    repeated runs never collide and ``make clean`` removes them with the rest of ``output/``.

    Every path handed on is absolute. ``output_dir`` defaults to the relative ``output``, and a
    sandbox that runs ``git worktree add <relative>`` from inside the clone puts the worktree
    somewhere other than where it later looks for it, leaving the model an empty repository.
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
        self._workspace_dir = Path(output_dir).resolve() / "workspace"
        self._sandbox_factory = sandbox_factory or _default_sandbox
        self._registry_factory = registry_factory or _default_registry

    def ingest(self, issue_url: str, *, repo_url: str | None = None, issue_text: str | None = None) -> Ingested:
        """Get the issue and shallow-clone its repository.

        With ``issue_text`` the issue is the text given: the GitHub API is not asked, and the repository comes
        from ``repo_url`` (or from ``issue_url``, which may be an issue URL or a repository URL). Otherwise the
        issue is fetched, and a fetch that failed raises ``IssueFetchError`` rather than passing GitHub's error
        note on as if it were the issue. Raises ``ValueError`` for a URL that is not a GitHub one.
        """
        issue = self._issue(issue_url, repo_url, issue_text)
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        dest = (self._workspace_dir / f"{issue.owner}__{issue.repo}__{issue.number}__{int(time.time())}").resolve()
        return Ingested(issue, Path(clone_repo(issue, dest)).resolve())

    @staticmethod
    def _issue(issue_url: str, repo_url: str | None, issue_text: str | None) -> IssueRef:
        text = (issue_text or "").strip()
        if text:
            return _supplied_issue(issue_url, repo_url, text)
        issue = fetch_issue(parse_issue_url(issue_url))
        if not issue.title.strip():
            # fetch_issue reports every failure as a note in ``body`` and leaves the title empty; a real issue always has one.
            note = issue.body.strip().strip("[]") or "no reason given"
            raise IssueFetchError(
                f"Could not fetch the issue from GitHub: {note}. To work without the API, supply the issue text "
                f'yourself: --repo <repository url> --issue-text "<the issue>".'
            )
        return issue

    def profile(self, ingested: Ingested) -> Workspace:
        """Detect languages and commands, map the repo, then open the sandbox and the tool registry."""
        root = Path(ingested.repo_root).resolve()
        profile = profile_repo(root)
        return Workspace(
            issue=ingested.issue,
            profile=profile,
            repo_map=repo_map(root),
            sandbox=PreparedSandbox(self._sandbox_factory(self._config, root, profile)),
            tools=self._registry_factory(profile),
        )


def _supplied_issue(issue_url: str, repo_url: str | None, text: str) -> IssueRef:
    """An ``IssueRef`` for text the user supplied. The repository is ``repo_url``'s, else the one ``issue_url`` names."""
    owner, repo = _owner_and_repo(repo_url or issue_url)
    number, url = 0, repo_url or issue_url
    try:
        named = parse_issue_url(issue_url)
    except ValueError:
        pass  # issue_url is a repository URL: there is no issue number
    else:
        number, url = named.number, named.url
    title = next((line.strip() for line in text.splitlines() if line.strip()), "")[:_TITLE_CHARS]
    return IssueRef(owner, repo, number, url, title=title, body=text)


def _owner_and_repo(url: str) -> tuple[str, str]:
    """``(owner, repo)`` of a GitHub repository or issue URL; ``ValueError`` for anything else."""
    url = url.strip()
    match = _REPO_URL.match(url)
    if match:
        return match["owner"], match["repo"]
    try:
        issue = parse_issue_url(url)
    except ValueError:
        raise ValueError(
            f"Not a GitHub repository URL: {url!r}. Expected https://github.com/<owner>/<repo>"
        ) from None
    return issue.owner, issue.repo


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
