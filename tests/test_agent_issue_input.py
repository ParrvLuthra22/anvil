"""Where the issue comes from: the GitHub fetch, or text the user supplied; and what happens when the fetch fails.

The URL parser is the real one (it is pure); only the network calls are stubbed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import anvil.agent.pipeline as pipeline_module
from anvil.agent.pipeline import IssueFetchError, RepoPipeline
from anvil.repo import ingest as ingest_module
from anvil.repo.ingest import IssueRef
from tests import real_repo as rr
from tests.fakes import FakePipeline, RecordingLLM
from tests.offline import no_base_ref_lookup  # noqa: F401 (autouse: no GitHub lookup)
from tests.test_orchestrator import execute, finalize, happy

ISSUE = "https://github.com/acme/calc/issues/7"
REPO = "https://github.com/acme/calc"
NOTE_404 = "[Issue not found (HTTP 404). URL: https://github.com/acme/calc/issues/7. The issue may be private or deleted.]"


@pytest.fixture
def net(monkeypatch):
    """Stub fetch_issue and clone_repo; ``net.fetched`` gets a ref per fetch, ``net.cloned`` the ref cloned."""

    class Net:
        fetched: list[IssueRef] = []
        cloned: list[IssueRef] = []
        result = staticmethod(lambda ref: replace(ref, title="add is wrong", body="details from GitHub"))

    net = Net()
    net.fetched, net.cloned = [], []

    def fetch(ref):
        net.fetched.append(ref)
        return net.result(ref)

    def clone(ref, dest):
        net.cloned.append(ref)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr(pipeline_module, "fetch_issue", fetch)
    monkeypatch.setattr(pipeline_module, "clone_repo", clone)
    return net


def ingest(tmp_path, url=ISSUE, **kwargs):
    return RepoPipeline({}, tmp_path).ingest(url, **kwargs)


# ---- supplied text (item 6) -----------------------------------------------------------------


def test_supplied_text_replaces_the_fetch_and_becomes_the_issue_body(tmp_path, net):
    got = ingest(tmp_path, ISSUE, issue_text="add() returns -1\nfor add(2, 3)").issue
    assert net.fetched == [], "GitHub must not be asked when the user supplied the text"
    assert got.body == "add() returns -1\nfor add(2, 3)"
    assert got.title == "add() returns -1"
    assert (got.owner, got.repo, got.number, got.url) == ("acme", "calc", 7, ISSUE)
    assert [(r.owner, r.repo) for r in net.cloned] == [("acme", "calc")]


def test_the_repository_comes_from_repo_url_when_it_is_given(tmp_path, net):
    got = ingest(tmp_path, ISSUE, repo_url="https://github.com/other/project.git", issue_text="broken").issue
    assert (got.owner, got.repo) == ("other", "project")
    assert got.number == 7, "the issue number still comes from the issue url"
    assert [(r.owner, r.repo) for r in net.cloned] == [("other", "project")]


def test_a_repository_url_in_place_of_the_issue_url_works_with_supplied_text(tmp_path, net):
    """`--repo <url> --issue-text ...` without --issue: the headless entry point passes the repo url as the issue url."""
    got = ingest(tmp_path, REPO, issue_text="add() is wrong").issue
    assert (got.owner, got.repo, got.number, got.url) == ("acme", "calc", 0, REPO)
    got = ingest(tmp_path, "unrelated", repo_url=REPO + "/", issue_text="add() is wrong").issue
    assert (got.owner, got.repo, got.number) == ("acme", "calc", 0)
    assert net.fetched == []


@pytest.mark.parametrize("repo", ["https://github.com/acme/calc", "https://github.com/acme/calc.git", "http://www.github.com/acme/calc/"])
def test_repository_urls_are_accepted_in_their_usual_forms(tmp_path, net, repo):
    got = ingest(tmp_path, repo, issue_text="x").issue
    assert (got.owner, got.repo) == ("acme", "calc")


@pytest.mark.parametrize("bad", ["https://gitlab.com/acme/calc", "not a url", "https://github.com/acme", ""])
def test_a_repository_that_is_not_a_github_one_is_refused_before_cloning(tmp_path, net, bad):
    with pytest.raises(ValueError, match="GitHub"):
        ingest(tmp_path, bad, issue_text="x")
    assert net.cloned == []


def test_blank_supplied_text_counts_as_no_text(tmp_path, net):
    got = ingest(tmp_path, ISSUE, issue_text="  \n ").issue
    assert len(net.fetched) == 1 and got.body == "details from GitHub"


def test_a_very_long_first_line_gives_a_short_title(tmp_path, net):
    assert len(ingest(tmp_path, ISSUE, issue_text="x" * 500).issue.title) == 100


# ---- a failed fetch (item 7) ----------------------------------------------------------------


ERROR_404 = "Issue not found (HTTP 404) — may be private or deleted. URL: https://github.com/acme/calc/issues/7"


def failed_fetch(ref):
    """How fetch_issue reports a failure now: ``fetch_error`` set, title and body left empty."""
    return replace(ref, title="", body="", fetch_error=ERROR_404)


def test_a_failed_fetch_is_an_error_not_an_issue_and_nothing_is_cloned(tmp_path, net):
    net.result = staticmethod(failed_fetch)
    with pytest.raises(IssueFetchError) as info:
        ingest(tmp_path, ISSUE)
    message = str(info.value)
    assert "Issue not found (HTTP 404)" in message and "--issue-text" in message and "--repo" in message
    assert net.cloned == [], "no point cloning a repository for an issue nobody can read"


def test_the_note_is_the_fetch_error_itself_not_a_placeholder(tmp_path, net):
    """Regression: with the structured field the reason was read from the (now empty) body: 'no reason given'."""
    net.result = staticmethod(failed_fetch)
    with pytest.raises(IssueFetchError) as info:
        ingest(tmp_path, ISSUE)
    assert f"Could not fetch the issue from GitHub: {ERROR_404}." in str(info.value)
    assert "no reason given" not in str(info.value)


def test_a_fetch_error_wins_even_when_a_title_came_back(tmp_path, net):
    net.result = staticmethod(lambda ref: replace(ref, title="add is wrong", body="half a body", fetch_error="HTTP error 502"))
    with pytest.raises(IssueFetchError, match="HTTP error 502"):
        ingest(tmp_path, ISSUE)
    assert net.cloned == []


def test_a_fetch_that_returns_nothing_at_all_still_ends_the_run(tmp_path, net):
    net.result = staticmethod(lambda ref: replace(ref, title="", body="", fetch_error=""))
    with pytest.raises(IssueFetchError, match="no reason given"):
        ingest(tmp_path, ISSUE)


def test_the_older_contract_a_note_in_the_body_with_no_title_is_still_recognised(tmp_path, net):
    net.result = staticmethod(lambda ref: replace(ref, title="", body=NOTE_404))
    with pytest.raises(IssueFetchError, match="Issue not found"):
        ingest(tmp_path, ISSUE)


def test_supplied_text_is_the_way_out_of_a_failed_fetch(tmp_path, net):
    net.result = staticmethod(failed_fetch)
    assert ingest(tmp_path, ISSUE, issue_text="add() is wrong").issue.body == "add() is wrong"
    assert net.fetched == []


def test_an_issue_with_a_title_but_an_empty_body_is_a_real_issue(tmp_path, net):
    net.result = staticmethod(lambda ref: replace(ref, title="add is wrong", body=""))
    assert ingest(tmp_path, ISSUE).issue.title == "add is wrong"


# ---- through the real fetch_issue: the way it actually reports failures ---------------------


def _client(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "status, expected",
    [(404, "Issue not found"), (403, "rate-limited")],
)
def test_the_real_fetch_issue_failures_are_recognised(tmp_path, monkeypatch, status, expected):
    monkeypatch.setattr(pipeline_module, "clone_repo", lambda ref, dest: pytest.fail("must not clone"))
    monkeypatch.setattr(ingest_module, "_http_client", _client(lambda request: httpx.Response(status, json={})))
    with pytest.raises(IssueFetchError, match=expected):
        ingest(tmp_path, ISSUE)


def test_a_network_failure_in_the_real_fetch_issue_is_recognised(tmp_path, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(pipeline_module, "clone_repo", lambda ref, dest: pytest.fail("must not clone"))
    monkeypatch.setattr(ingest_module, "_http_client", _client(refuse))
    with pytest.raises(IssueFetchError, match="Network error"):
        ingest(tmp_path, ISSUE)


def test_a_successful_real_fetch_issue_passes_through(tmp_path, monkeypatch, net):
    def api(request):
        if request.url.path.endswith("/comments"):
            return httpx.Response(200, json=[{"body": "me too"}])
        return httpx.Response(200, json={"title": "add is wrong", "body": "add(2, 3) == -1"})

    monkeypatch.setattr(pipeline_module, "fetch_issue", ingest_module.fetch_issue)
    monkeypatch.setattr(ingest_module, "_http_client", _client(api))
    issue = ingest(tmp_path, ISSUE).issue
    assert (issue.title, issue.body, issue.comments) == ("add is wrong", "add(2, 3) == -1", ["me too"])


# ---- through the orchestrator ---------------------------------------------------------------


def test_the_orchestrator_passes_issue_text_and_repo_url_to_ingest_and_says_so(tmp_path):
    from anvil.agent.orchestrator import run_harness
    from anvil.events import EventBus

    pipeline = FakePipeline()
    bus = EventBus()
    queue = bus.subscribe()
    run_harness(
        ISSUE, {"output_dir": str(tmp_path / "out")}, bus, llm=RecordingLLM(happy()), pipeline=pipeline,
        repo_url=REPO, issue_text="add() is wrong",
    )
    assert pipeline.ingest_calls == [{"issue_url": ISSUE, "repo_url": REPO, "issue_text": "add() is wrong", "git_ref": None}]
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    said = [e.data["text"] for e in events if e.type == "message" and e.data["role"] == "system"]
    assert any(text.startswith("Using the issue text you supplied") for text in said)
    assert not any(text.startswith("Fetched issue") for text in said)


def test_without_supplied_text_ingest_is_called_as_before(tmp_path):
    pipeline = FakePipeline()
    execute(happy(), tmp_path, pipeline=pipeline)
    assert pipeline.ingest_calls[0]["repo_url"] is None and pipeline.ingest_calls[0]["issue_text"] is None


def test_the_report_of_a_run_without_an_issue_number_does_not_say_number_zero(tmp_path):
    from anvil.agent.orchestrator import run_harness
    from anvil.events import EventBus

    run_harness(
        REPO, {"output_dir": str(tmp_path / "out")}, EventBus(), llm=RecordingLLM(happy()), pipeline=FakePipeline(),
        issue_text="add() is wrong",
    )
    report = (tmp_path / "out" / "report.md").read_text()
    assert "#0" not in report


# ---- the whole run, real repository ---------------------------------------------------------


def test_a_failed_fetch_ends_the_run_cleanly_and_the_model_is_never_shown_the_error_note(tmp_path, monkeypatch):
    """Regression (audit): the 404 note went to the model as 'the issue', and the run carried on blindly."""
    rr.prepare(tmp_path, monkeypatch, fetch=False)
    monkeypatch.setattr(ingest_module, "_http_client", _client(lambda request: httpx.Response(404, json={})))
    llm = RecordingLLM([])
    run = rr.run_real([], llm=llm)

    assert llm.calls == [], "no LLM call may be made for an issue nobody could read"
    (error,) = [e for e in run.events if e.type == "error"]
    assert error.data["kind"] == "ingest"
    assert "Issue not found (HTTP 404)" in error.data["message"] and "--issue-text" in error.data["message"]
    assert (run.out / "report.md").exists() and (run.out / "patch.diff").read_text() == ""
    assert "Stopped early by an error (ingest)" in run.report
    assert run.events[-1].type == "done" and run.done.data["resolved_confidence"] == 0.0
    assert not (tmp_path / "out" / "workspace").exists() or not any((tmp_path / "out" / "workspace").iterdir())


def test_supplied_text_fixes_the_issue_when_github_cannot_be_asked(tmp_path, monkeypatch):
    rr.prepare(tmp_path, monkeypatch, fetch=False)
    monkeypatch.setattr(pipeline_module, "fetch_issue", lambda ref: pytest.fail("GitHub must not be asked"))
    script = (
        rr.understand() + rr.localize() + rr.reproduce() + rr.good_patch() + rr.verify() + rr.review_ok() + rr.finalize()
    )
    run = rr.run_real(script, repo_url=rr.REPO_URL, issue_text="add(2, 3) returns -1 instead of 5.")

    assert run.errors() == []
    assert "+    return a + b" in run.patch and run.done.data["resolved_confidence"] == pytest.approx(0.9)
    brief = run.llm.calls[0][0][1]["content"]
    assert "add(2, 3) returns -1 instead of 5." in brief and "<issue>" in brief
    assert "acme/calc" in brief and "Issue not found" not in " ".join(run.prompt_texts())


def test_a_repository_url_alone_plus_text_needs_no_issue_number(tmp_path, monkeypatch):
    rr.prepare(tmp_path, monkeypatch, fetch=False)
    script = (
        rr.understand() + rr.localize() + rr.reproduce() + rr.good_patch() + rr.verify() + rr.review_ok() + rr.finalize()
    )
    run = rr.run_real(script, issue_url=rr.REPO_URL, issue_text="add(2, 3) returns -1 instead of 5.")
    assert run.errors() == [] and "#0" not in run.report
    assert "+    return a + b" in run.patch
