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


@dataclass
class BaseRefResult:
    """The result of resolving an issue's pre-fix commit."""
    ref: str | None
    reason: str
    already_fixed: bool


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


def resolve_base_ref(ref: IssueRef) -> BaseRefResult:
    """Determine the pre-fix base commit for a given issue.
    
    If the issue is already fixed by a merged PR or a direct commit, returns
    the parent commit of the fix (or the PR's base sha).
    """
    timeline_url = f"{_GITHUB_API}/repos/{ref.owner}/{ref.repo}/issues/{ref.number}/timeline"
    issue_url = f"{_GITHUB_API}/repos/{ref.owner}/{ref.repo}/issues/{ref.number}"
    
    try:
        with _http_client() as client:
            resp = client.get(issue_url)
            if resp.status_code == 403:
                return BaseRefResult(None, "GitHub API rate-limited (HTTP 403)", False)
            if resp.status_code == 404:
                return BaseRefResult(None, "Issue not found", False)
            resp.raise_for_status()
            issue_data = resp.json()
            
            if issue_data.get("state") == "open":
                return BaseRefResult(None, "Issue is still open", False)
            
            # Fetch timeline
            t_resp = client.get(timeline_url)
            if t_resp.status_code == 403:
                return BaseRefResult(None, "GitHub API rate-limited on timeline", True)
            t_resp.raise_for_status()
            timeline = t_resp.json()
            
            fix_commit = None
            fix_pr = None
            
            for event in timeline:
                ev_type = event.get("event")
                if ev_type == "closed" and event.get("commit_id"):
                    fix_commit = event["commit_id"]
                    break
                elif ev_type == "cross-referenced":
                    issue_src = event.get("source", {}).get("issue", {})
                    if issue_src.get("pull_request") and issue_src.get("state") == "closed":
                        if issue_src.get("pull_request", {}).get("merged_at"):
                            fix_pr = issue_src.get("number")
                elif ev_type == "connected":
                    subj = event.get("subject", {})
                    if subj.get("type") == "pull_request" and subj.get("state") == "merged":
                        fix_pr = subj.get("number")
            
            if fix_commit:
                return BaseRefResult(f"{fix_commit}^", f"Closed by commit {fix_commit[:7]}", True)
            
            if fix_pr:
                pr_url = f"{_GITHUB_API}/repos/{ref.owner}/{ref.repo}/pulls/{fix_pr}"
                pr_resp = client.get(pr_url)
                if pr_resp.status_code == 200:
                    pr_data = pr_resp.json()
                    if pr_data.get("merged"):
                        base_sha = pr_data.get("base", {}).get("sha")
                        if base_sha:
                            return BaseRefResult(base_sha, f"Closed by merged PR #{fix_pr}", True)
                        
            return BaseRefResult(None, "Issue is closed but no fix commit or PR found", True)
            
    except httpx.HTTPError as exc:
        return BaseRefResult(None, f"Network error resolving base ref: {exc}", False)

# ---------------------------------------------------------------------------
# Repository clone
# ---------------------------------------------------------------------------

_CLONE_TIMEOUT = 60   # seconds — tight so missing/private repos fail fast


def clone_repo(ref: IssueRef, dest: Path, git_ref: str | None = None) -> Path:
    """Shallow-clone the issue's repository into ``dest`` and return the repo root.

    When *git_ref* is **None** a plain ``git clone --depth 1`` of the remote
    default branch is performed.

    When *git_ref* is provided (e.g. a full 40-character SHA or a branch name)
    the function attempts three strategies in order:

    1. ``git fetch --depth 1 origin <sha>`` — fastest, works when the server
       advertises that commit (most GitHub repos allow single-SHA fetches).
    2. ``git fetch --depth 50 origin <sha>`` — deeper fetch for repos that
       require some history before serving the commit.
    3. **Raise** :class:`RuntimeError` with a clear message.  The caller
       (agent cycle) can then fall back to ``HEAD`` if desired.

    The subprocess environment always has ``GIT_TERMINAL_PROMPT=0`` to prevent
    interactive prompts from hanging the process, and all secret env vars
    are stripped.

    Args:
        ref:     The parsed issue reference (owner/repo must be set).
        dest:    Destination directory for the clone (created if absent).
        git_ref: Optional full SHA or branch/tag.  If *None* the remote default
                 branch is used.

    Returns:
        The repo root path (*dest*).

    Raises:
        RuntimeError: If any git operation fails or times out.
    """
    import os
    import subprocess
    
    clone_url = f"https://github.com/{ref.owner}/{ref.repo}.git"
    dest.mkdir(parents=True, exist_ok=True)

    # Sanitize environment: no secrets, no interactive prompts.
    env = {k: v for k, v in os.environ.items()
           if k not in ("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    env["GIT_TERMINAL_PROMPT"] = "0"

    if not git_ref:
        cmd = ["git", "clone", "--depth", "1", clone_url, str(dest)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=_CLONE_TIMEOUT, stdin=subprocess.DEVNULL, env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"git clone timed out for {clone_url}")
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone failed (exit {result.returncode}) for {clone_url}:\n"
                f"stderr: {result.stderr[:500]}"
            )
        return dest

    try:
        subprocess.run(
            ["git", "init", str(dest)], check=True, capture_output=True,
            timeout=15, env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git init timed out for {dest}")

    def _fetch(depth: int) -> subprocess.CompletedProcess:
        fetch_args = ["git", "fetch", f"--depth={depth}", clone_url, git_ref]
        try:
            return subprocess.run(
                fetch_args, cwd=dest, capture_output=True, text=True,
                timeout=_CLONE_TIMEOUT, stdin=subprocess.DEVNULL, env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"git fetch --depth={depth} timed out for {clone_url} ref={git_ref!r}"
            )

    # Step 1: shallow fetch
    res = _fetch(1)
    if res.returncode != 0:
        # Step 2: deeper fetch (catches repos that need some history)
        res = _fetch(50)

    if res.returncode != 0:
        # Step 3: clear error — let the caller fall back to HEAD
        raise RuntimeError(
            f"git fetch failed for {clone_url} ref={git_ref!r} "
            f"(both --depth=1 and --depth=50 failed).\n"
            f"Caller should fall back to HEAD.\n"
            f"stderr: {res.stderr[:500]}"
        )

    # Checkout the fetched commit
    try:
        co = subprocess.run(
            ["git", "checkout", "FETCH_HEAD"],
            cwd=dest, capture_output=True, text=True,
            timeout=_CLONE_TIMEOUT, stdin=subprocess.DEVNULL, env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git checkout FETCH_HEAD timed out for {git_ref!r}")

    if co.returncode != 0:
        raise RuntimeError(
            f"git checkout FETCH_HEAD failed for {git_ref!r}:\n"
            f"stderr: {co.stderr[:500]}"
        )

    return dest
        
    # If git_ref is provided, init an empty repo and fetch that specific ref
    subprocess.run(["git", "init", str(dest)], check=True, capture_output=True)
    
    # Try shallow fetch first
    fetch_cmd = ["git", "fetch", "--depth", "1", clone_url, git_ref]
    res = subprocess.run(fetch_cmd, cwd=dest, capture_output=True, text=True, timeout=_CLONE_TIMEOUT, stdin=subprocess.DEVNULL, env=env)
    
    if res.returncode != 0:
        # Fall back to a deeper fetch
        fetch_cmd = ["git", "fetch", "--deepen", "100", clone_url, git_ref]
        res = subprocess.run(fetch_cmd, cwd=dest, capture_output=True, text=True, timeout=_CLONE_TIMEOUT, stdin=subprocess.DEVNULL, env=env)
        
    if res.returncode != 0:
        # If it still fails, we might just be failing to fetch a raw SHA directly if the server forbids it.
        # Let's try a full clone and checkout.
        import shutil
        shutil.rmtree(dest)
        subprocess.run(["git", "clone", clone_url, str(dest)], check=True, capture_output=True, env=env, timeout=_CLONE_TIMEOUT)
        res = subprocess.run(["git", "checkout", git_ref], cwd=dest, capture_output=True, text=True, env=env, timeout=_CLONE_TIMEOUT)
        if res.returncode != 0:
            raise RuntimeError(f"git checkout {git_ref} failed:\nstderr: {res.stderr[:500]}")
        return dest
        
    checkout_res = subprocess.run(["git", "checkout", "FETCH_HEAD"], cwd=dest, capture_output=True, text=True, env=env, timeout=_CLONE_TIMEOUT)
    if checkout_res.returncode != 0:
        raise RuntimeError(f"git checkout FETCH_HEAD failed for {git_ref}:\nstderr: {checkout_res.stderr[:500]}")
    
    return dest
