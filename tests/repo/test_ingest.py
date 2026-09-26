"""Offline tests for repo/ingest.py — no network, no git clones."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from anvil.repo.ingest import IssueRef, clone_repo, fetch_issue, parse_issue_url


# ---------------------------------------------------------------------------
# parse_issue_url
# ---------------------------------------------------------------------------

class TestParseIssueUrl:
    """parse_issue_url must be pure (no I/O) and handle edge cases."""

    def test_canonical_issues_url(self):
        ref = parse_issue_url("https://github.com/psf/requests/issues/42")
        assert ref.owner == "psf"
        assert ref.repo == "requests"
        assert ref.number == 42
        assert ref.url == "https://github.com/psf/requests/issues/42"

    def test_trailing_slash(self):
        ref = parse_issue_url("https://github.com/psf/requests/issues/42/")
        assert ref.number == 42

    def test_query_string_ignored(self):
        ref = parse_issue_url("https://github.com/psf/requests/issues/42?ref=email")
        assert ref.number == 42

    def test_fragment_ignored(self):
        ref = parse_issue_url("https://github.com/psf/requests/issues/42#issuecomment-999")
        assert ref.number == 42

    def test_pull_url_accepted(self):
        """PRs share numbering with issues on GitHub — /pull/ must be accepted."""
        ref = parse_issue_url("https://github.com/psf/requests/pull/99")
        assert ref.number == 99
        # Canonical URL normalised to /issues/
        assert "/issues/" in ref.url

    def test_http_url_accepted(self):
        ref = parse_issue_url("http://github.com/psf/requests/issues/1")
        assert ref.owner == "psf"

    def test_whitespace_stripped(self):
        ref = parse_issue_url("  https://github.com/psf/requests/issues/1  ")
        assert ref.number == 1

    def test_non_github_domain_raises(self):
        with pytest.raises(ValueError, match="Not a GitHub URL"):
            parse_issue_url("https://gitlab.com/psf/requests/issues/1")

    def test_github_wrong_path_raises(self):
        with pytest.raises(ValueError):
            parse_issue_url("https://github.com/psf/requests/blob/main/README.md")

    def test_bare_github_raises(self):
        with pytest.raises(ValueError):
            parse_issue_url("https://github.com/psf/requests")

    def test_missing_number_raises(self):
        with pytest.raises(ValueError):
            parse_issue_url("https://github.com/psf/requests/issues/")

    def test_non_url_raises(self):
        with pytest.raises(ValueError):
            parse_issue_url("not-a-url-at-all")

    def test_dataclass_defaults(self):
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        assert ref.title == ""
        assert ref.body == ""
        assert ref.comments == []

    def test_independent_comments_list(self):
        """Two different IssueRef instances must not share the comments list."""
        ref1 = parse_issue_url("https://github.com/a/b/issues/1")
        ref2 = parse_issue_url("https://github.com/a/b/issues/2")
        ref1.comments.append("hello")
        assert ref2.comments == []


# ---------------------------------------------------------------------------
# fetch_issue  (all network calls mocked via httpx)
# ---------------------------------------------------------------------------

def _make_mock_response(status_code: int, json_data=None, raise_exc=None):
    """Build a mock httpx.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data or {}
    if raise_exc:
        mock.raise_for_status.side_effect = raise_exc
    else:
        mock.raise_for_status.return_value = None
    return mock


class TestFetchIssue:
    """fetch_issue must always return an IssueRef — never raise on API failures."""

    def _ref(self) -> IssueRef:
        return parse_issue_url("https://github.com/psf/requests/issues/1")

    @patch("anvil.repo.ingest._http_client")
    def test_successful_fetch(self, mock_client_factory):
        issue_resp = _make_mock_response(200, {"title": "Bug report", "body": "Steps to repro"})
        comments_resp = _make_mock_response(200, [{"body": "Me too!"}, {"body": "Fixed in v3"}])

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = [issue_resp, comments_resp]
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert ref.title == "Bug report"
        assert ref.body == "Steps to repro"
        assert ref.comments == ["Me too!", "Fixed in v3"]

    @patch("anvil.repo.ingest._http_client")
    def test_rate_limited_403_returns_ref(self, mock_client_factory):
        resp = _make_mock_response(403)
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.return_value = resp
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert "rate-limit" in ref.body.lower() or "403" in ref.body
        assert ref.title == ""  # not filled in

    @patch("anvil.repo.ingest._http_client")
    def test_not_found_404_returns_ref(self, mock_client_factory):
        resp = _make_mock_response(404)
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.return_value = resp
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert "404" in ref.body or "not found" in ref.body.lower()

    @patch("anvil.repo.ingest._http_client")
    def test_network_error_returns_ref(self, mock_client_factory):
        import httpx
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = httpx.ConnectError("connection refused")
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert "network error" in ref.body.lower() or "connection" in ref.body.lower()

    @patch("anvil.repo.ingest._http_client")
    def test_timeout_returns_ref(self, mock_client_factory):
        import httpx
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert "timeout" in ref.body.lower()

    @patch("anvil.repo.ingest._http_client")
    def test_empty_body_normalised(self, mock_client_factory):
        """A null body from GitHub (body=None) should be stored as empty string."""
        issue_resp = _make_mock_response(200, {"title": "Title", "body": None})
        comments_resp = _make_mock_response(200, [])
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = [issue_resp, comments_resp]
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert ref.body == ""

    @patch("anvil.repo.ingest._http_client")
    def test_blank_comments_skipped(self, mock_client_factory):
        """Comments with blank/whitespace body should not be added."""
        issue_resp = _make_mock_response(200, {"title": "T", "body": "B"})
        comments_resp = _make_mock_response(200, [{"body": "  "}, {"body": ""}, {"body": None}])
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = [issue_resp, comments_resp]
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert ref.comments == []


# ---------------------------------------------------------------------------
# clone_repo  (subprocess mocked)
# ---------------------------------------------------------------------------

class TestCloneRepo:
    """clone_repo must call git with the right args and raise on failure."""

    def test_success(self, tmp_path):
        dest = tmp_path / "cloned"
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = clone_repo(
                parse_issue_url("https://github.com/psf/requests/issues/1"), dest
            )
        assert result == dest
        call_args = mock_run.call_args[0][0]
        assert "git" in call_args
        assert "--depth" in call_args
        assert "1" in call_args
        assert "https://github.com/psf/requests.git" in call_args

    def test_git_failure_raises(self, tmp_path):
        dest = tmp_path / "cloned"
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=128, stdout="", stderr="repository not found"
            )
            with pytest.raises(RuntimeError, match="git clone failed"):
                clone_repo(
                    parse_issue_url("https://github.com/psf/requests/issues/1"), dest
                )

    def test_dest_created(self, tmp_path):
        """clone_repo must create dest even if it does not exist yet."""
        dest = tmp_path / "a" / "b" / "c"
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            clone_repo(parse_issue_url("https://github.com/psf/requests/issues/1"), dest)
        assert dest.exists()
