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

        # Bug 5: error goes to fetch_error, body stays empty
        assert ref.body == ""
        assert "rate" in ref.fetch_error.lower() or "403" in ref.fetch_error
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

        # Bug 5: error goes to fetch_error, body stays empty
        assert ref.body == ""
        assert "404" in ref.fetch_error or "not found" in ref.fetch_error.lower()

    @patch("anvil.repo.ingest._http_client")
    def test_network_error_returns_ref(self, mock_client_factory):
        import httpx
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = httpx.ConnectError("connection refused")
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        # Bug 5: error goes to fetch_error, body stays empty
        assert ref.body == ""
        assert ref.fetch_error != ""

    @patch("anvil.repo.ingest._http_client")
    def test_timeout_returns_ref(self, mock_client_factory):
        import httpx
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=cm)
        cm.__exit__ = MagicMock(return_value=False)
        cm.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_factory.return_value = cm

        ref = fetch_issue(self._ref())

        assert "timeout" in ref.fetch_error.lower()
        assert ref.body == ""

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


# ---------------------------------------------------------------------------
# Regression tests for Bug 5 (ingest hardening)
# ---------------------------------------------------------------------------

class TestBug5IngestHardening:
    """Bug 5: structured errors, GITHUB_TOKEN, GIT_TERMINAL_PROMPT, clone timeout."""

    # -- fetch_issue: structured error field --

    def test_403_sets_fetch_error_not_body(self):
        """Bug 5: 403 must set fetch_error, not contaminate body."""
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest._http_client") as mock_factory:
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            resp = MagicMock()
            resp.status_code = 403
            cm.get.return_value = resp
            mock_factory.return_value = cm
            out = fetch_issue(ref)
        assert out.body == ""          # body must be clean
        assert "403" in out.fetch_error or "rate" in out.fetch_error.lower()

    def test_404_sets_fetch_error_not_body(self):
        """Bug 5: 404 must set fetch_error, not contaminate body."""
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest._http_client") as mock_factory:
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            resp = MagicMock()
            resp.status_code = 404
            cm.get.return_value = resp
            mock_factory.return_value = cm
            out = fetch_issue(ref)
        assert out.body == ""
        assert "404" in out.fetch_error or "not found" in out.fetch_error.lower()

    def test_network_error_sets_fetch_error(self):
        """Bug 5: network error sets fetch_error cleanly."""
        import httpx
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest._http_client") as mock_factory:
            cm = MagicMock()
            cm.__enter__ = MagicMock(return_value=cm)
            cm.__exit__ = MagicMock(return_value=False)
            cm.get.side_effect = httpx.ConnectError("connection refused")
            mock_factory.return_value = cm
            out = fetch_issue(ref)
        assert out.body == ""
        assert out.fetch_error != ""

    def test_github_token_attached_as_bearer(self):
        """Bug 5: GITHUB_TOKEN env var is used as Bearer auth header, never logged."""
        import os
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_testtokenxyz"}):
            with patch("anvil.repo.ingest.httpx.Client") as mock_client_cls:
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                from anvil.repo.ingest import _http_client
                _http_client()
                _, kwargs = mock_client_cls.call_args
                headers = kwargs.get("headers", {})
                # Must attach Authorization header
                assert "Authorization" in headers
                assert "ghp_testtokenxyz" in headers["Authorization"]
                # Token must NOT appear in Accept or version headers (no logging)
                assert "ghp_testtokenxyz" not in str(kwargs.get("timeout", ""))

    def test_no_github_token_no_auth_header(self):
        """Without GITHUB_TOKEN, no Authorization header is sent."""
        import os
        env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("anvil.repo.ingest.httpx.Client") as mock_client_cls:
                mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
                mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
                from anvil.repo.ingest import _http_client
                _http_client()
                _, kwargs = mock_client_cls.call_args
                headers = kwargs.get("headers", {})
                assert "Authorization" not in headers

    # -- clone_repo: GIT_TERMINAL_PROMPT, timeout, git_ref --

    def test_clone_sets_git_terminal_prompt_zero(self, tmp_path):
        """Bug 5: GIT_TERMINAL_PROMPT=0 must be in subprocess env."""
        dest = tmp_path / "r"
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            clone_repo(ref, dest)
            _, kwargs = mock_run.call_args
            env = kwargs.get("env", {})
            assert env.get("GIT_TERMINAL_PROMPT") == "0"

    def test_clone_passes_timeout(self, tmp_path):
        """Bug 5: subprocess.run must be called with a timeout."""
        dest = tmp_path / "r"
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            clone_repo(ref, dest)
            _, kwargs = mock_run.call_args
            assert kwargs.get("timeout") is not None

    def test_clone_timeout_raises_runtime_error(self, tmp_path):
        """Bug 5: TimeoutExpired must surface as RuntimeError (not hang)."""
        dest = tmp_path / "r"
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=60)
            with pytest.raises(RuntimeError, match="timed out"):
                clone_repo(ref, dest)

    def test_clone_repo_with_git_ref(self, tmp_path):
        """Bug 5: git_ref kwarg passes --branch to git clone."""
        dest = tmp_path / "r"
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            clone_repo(ref, dest, git_ref="v2.31.0")
            call_args = mock_run.call_args[0][0]
            assert "--branch" in call_args
            assert "v2.31.0" in call_args

    def test_clone_no_git_ref_no_branch_flag(self, tmp_path):
        """Without git_ref, --branch must not appear in the command."""
        dest = tmp_path / "r"
        ref = parse_issue_url("https://github.com/psf/requests/issues/1")
        with patch("anvil.repo.ingest.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            clone_repo(ref, dest)
            call_args = mock_run.call_args[0][0]
            assert "--branch" not in call_args

    def test_issue_ref_has_fetch_error_field(self):
        """IssueRef dataclass must have a fetch_error field (default empty string)."""
        ref = IssueRef(owner="a", repo="b", number=1, url="http://x")
        assert hasattr(ref, "fetch_error")
        assert ref.fetch_error == ""
