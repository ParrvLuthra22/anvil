"""Turn a GitHub issue URL into an issue description and a local clone of its repo."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class IssueRef:
    """A GitHub issue: identity plus (once fetched) its title, body and comments."""

    owner: str
    repo: str
    number: int
    url: str
    title: str = ""
    body: str = ""
    comments: list[str] = field(default_factory=list)
    # Structured error set when fetch_issue fails (body is never the raw error)
    fetch_error: str = ""


# ---------------------------------------------------------------------------
# URL parsing (no network)
# ---------------------------------------------------------------------------

_GITHUB_PATTERN = re.compile(
    r"^https?://github\.com"
    r"/(?P<owner>[^/]+)"
    r"/(?P<repo>[^/]+)"
    r"/(?:issues|pull)"
    r"/(?P<number>\d+)"
    r"[/?#]?",
    re.IGNORECASE,
)


def parse_issue_url(url: str) -> IssueRef:
    """Parse ``https://github.com/<owner>/<repo>/issues/<n>`` without touching the network.

    Also accepts ``/pull/<n>`` (PRs share numbering with issues on GitHub).
    Trailing slashes, query strings, and fragments are silently ignored.
    Raises ``ValueError`` on anything that does not match the expected pattern.
    """
    url = url.strip()

    m = _GITHUB_PATTERN.match(url)
    if not m:
        parsed = urlparse(url)
        if parsed.netloc not in ("github.com", "www.github.com"):
            raise ValueError(
                f"Not a GitHub URL: {url!r}. "
                "Expected https://github.com/<owner>/<repo>/issues/<number>"
            )
        raise ValueError(
            f"Unrecognised GitHub URL format: {url!r}. "
            "Expected https://github.com/<owner>/<repo>/issues/<number> "
            "(or /pull/<number>)"
        )

    owner = m.group("owner")
    repo = m.group("repo")
    number = int(m.group("number"))

    canonical = f"https://github.com/{owner}/{repo}/issues/{number}"
    return IssueRef(owner=owner, repo=repo, number=number, url=canonical)


# ---------------------------------------------------------------------------
# GitHub REST API fetch
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 30.0
_MAX_COMMENTS = 10


def _http_client() -> httpx.Client:
    """Return a pre-configured httpx client with sensible defaults.

    If ``GITHUB_TOKEN`` is set in the environment it is attached as a
    Bearer token for higher rate limits.  The value is *never* logged.
    """
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return httpx.Client(
        timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT,
                              write=10.0, pool=10.0),
        headers=headers,
        follow_redirects=True,
    )


def fetch_issue(ref: IssueRef) -> IssueRef:
    """Fill in title, body and comments via the GitHub REST API.

    Supports an optional ``GITHUB_TOKEN`` env var for authenticated requests
    (higher rate limits).  The token is read-only and never logged.

    On 403 (rate-limit), 404, or any network error the function returns the
    ``IssueRef`` with ``fetch_error`` set to a structured message and
    ``body`` left empty, so the agent can still work from the URL alone.
    """
    issue_url = f"{_GITHUB_API}/repos/{ref.owner}/{ref.repo}/issues/{ref.number}"
    comments_url = f"{issue_url}/comments"

    try:
        with _http_client() as client:
            resp = client.get(issue_url)

            if resp.status_code == 403:
                ref.fetch_error = (
                    f"GitHub API rate-limited (HTTP 403). "
                    f"Set GITHUB_TOKEN for higher limits. URL: {ref.url}"
                )
                return ref

            if resp.status_code == 404:
                ref.fetch_error = (
                    f"Issue not found (HTTP 404) — may be private or deleted. "
                    f"URL: {ref.url}"
                )
                return ref

            resp.raise_for_status()
            data = resp.json()
            ref.title = data.get("title", "")
            ref.body = data.get("body", "") or ""

            c_resp = client.get(comments_url, params={"per_page": _MAX_COMMENTS, "page": 1})
            if c_resp.status_code == 200:
                for comment in c_resp.json():
                    body_text = comment.get("body", "") or ""
                    if body_text.strip():
                        ref.comments.append(body_text)

    except httpx.TimeoutException as exc:
        ref.fetch_error = f"Network timeout fetching issue: {exc}. URL: {ref.url}"
    except httpx.HTTPStatusError as exc:
        ref.fetch_error = (
            f"HTTP error {exc.response.status_code} fetching issue. "
            f"URL: {ref.url}"
        )
    except httpx.HTTPError as exc:
        ref.fetch_error = f"Network error fetching issue: {type(exc).__name__}. URL: {ref.url}"

    return ref


# ---------------------------------------------------------------------------
# Repository clone
# ---------------------------------------------------------------------------

_CLONE_TIMEOUT = 60   # seconds — tight so missing/private repos fail fast


def clone_repo(ref: IssueRef, dest: Path, git_ref: str | None = None) -> Path:
    """Shallow-clone the issue's repository into ``dest`` and return the repo root.

    Uses ``git clone --depth 1`` for speed.

    ``GIT_TERMINAL_PROMPT=0`` is set in the subprocess env so that git never
    hangs waiting for a username/password prompt on private repos.

    Args:
        ref:     The parsed issue reference (owner/repo must be set).
        dest:    Destination directory for the clone (created if absent).
        git_ref: Optional branch/tag/SHA to clone.  If *None* the remote's
                 default branch (usually ``main`` or ``master``) is used.
                 Example: ``"v2.31.0"`` or ``"fix/my-branch"``.

    Raises:
        RuntimeError: If ``git clone`` exits with a non-zero code or times out.
    """
    clone_url = f"https://github.com/{ref.owner}/{ref.repo}.git"
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1"]
    if git_ref:
        cmd += ["--branch", git_ref]
    cmd += [clone_url, str(dest)]

    # Disable interactive prompts so private/missing repos fail immediately.
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Never pass secrets into git subprocess
    for key in list(env):
        if key in ("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            del env[key]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_CLONE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"git clone timed out after {_CLONE_TIMEOUT}s for {clone_url}. "
            "The repo may be large or unreachable."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}) for {clone_url}:\n"
            f"stderr: {result.stderr[:500]}"
        )

    return dest
