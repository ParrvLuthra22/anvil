"""Turn a GitHub issue URL into an issue description and a local clone of its repo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


def parse_issue_url(url: str) -> IssueRef:
    """Parse ``https://github.com/<owner>/<repo>/issues/<n>`` without touching the network."""
    raise NotImplementedError


def fetch_issue(ref: IssueRef) -> IssueRef:
    """Fill in title, body and comments via the unauthenticated GitHub REST API."""
    raise NotImplementedError


def clone_repo(ref: IssueRef, dest: Path) -> Path:
    """Shallow-clone the issue's repository into ``dest`` and return the repo root."""
    raise NotImplementedError
