"""Turn a GitHub issue URL into an issue description and a local clone of its repo."""

from __future__ import annotations

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
    # Strip leading/trailing whitespace that users sometimes paste
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

    # Normalise to the canonical issues URL regardless of /pull/ input
    canonical = f"https://github.com/{owner}/{repo}/issues/{number}"
    return IssueRef(owner=owner, repo=repo, number=number, url=canonical)


# ---------------------------------------------------------------------------
# GitHub REST API fetch (unauthenticated)
# ---------------------------------------------------------------------------

_GITHUB_API = "https://api.github.com"
_CONNECT_TIMEOUT = 10.0   # seconds
_READ_TIMEOUT = 30.0       # seconds
_MAX_COMMENTS = 10


def _http_client() -> httpx.Client:
    """Return a pre-configured httpx client with sensible defaults."""
    return httpx.Client(
        timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT,
                              write=10.0, pool=10.0),
        headers={"Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28"},
        follow_redirects=True,
    )


def fetch_issue(ref: IssueRef) -> IssueRef:
    """Fill in title, body and comments via the unauthenticated GitHub REST API.

    On 403 (rate-limit), 404 or any network error the function returns the
    ``IssueRef`` with a descriptive error note in ``body`` instead of raising,
    so the agent can still attempt to work from the URL alone.
    """
    issue_url = f"{_GITHUB_API}/repos/{ref.owner}/{ref.repo}/issues/{ref.number}"
    comments_url = f"{issue_url}/comments"

    try:
        with _http_client() as client:
            # --- Fetch issue metadata ---
            resp = client.get(issue_url)

            if resp.status_code == 403:
                ref.body = (
                    f"[GitHub API rate-limited (HTTP 403). "
                    f"URL: {ref.url}. Retry after the rate-limit window resets "
                    f"or authenticate with a token.]"
                )
                return ref

            if resp.status_code == 404:
                ref.body = (
                    f"[Issue not found (HTTP 404). "
                    f"URL: {ref.url}. The issue may be private or deleted.]"
                )
                return ref

            resp.raise_for_status()
            data = resp.json()
            ref.title = data.get("title", "")
            ref.body = data.get("body", "") or ""

            # --- Fetch first N comments ---
            c_resp = client.get(comments_url, params={"per_page": _MAX_COMMENTS, "page": 1})
            if c_resp.status_code == 200:
                for comment in c_resp.json():
                    body_text = comment.get("body", "") or ""
                    if body_text.strip():
                        ref.comments.append(body_text)

    except httpx.TimeoutException as exc:
        ref.body = f"[Network timeout fetching issue: {exc}. URL: {ref.url}]"
    except httpx.HTTPStatusError as exc:
        ref.body = (
            f"[HTTP error {exc.response.status_code} fetching issue: {exc}. "
            f"URL: {ref.url}]"
        )
    except httpx.HTTPError as exc:
        ref.body = f"[Network error fetching issue: {exc}. URL: {ref.url}]"

    return ref


# ---------------------------------------------------------------------------
# Repository clone
# ---------------------------------------------------------------------------

_CLONE_TIMEOUT = 180  # seconds — generous for large-ish repos


def clone_repo(ref: IssueRef, dest: Path) -> Path:
    """Shallow-clone the issue's repository into ``dest`` and return the repo root.

    Uses ``git clone --depth 1`` for speed.  Raises ``RuntimeError`` if git
    exits with a non-zero code.
    """
    clone_url = f"https://github.com/{ref.owner}/{ref.repo}.git"
    dest.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(dest)],
        capture_output=True,
        text=True,
        timeout=_CLONE_TIMEOUT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    return dest
