"""Which revision the pipeline checks out: explicit ref, then resolve_base_ref, then the default branch."""

from dataclasses import dataclass

import pytest

import anvil.agent.pipeline as pipeline_module
from anvil.agent.pipeline import Checkout, RepoPipeline
from anvil.repo.ingest import IssueRef

URL = "https://github.com/acme/calc/issues/7"
REPO_URL = "https://github.com/acme/calc"


@dataclass
class Found:
    """What a resolver may return: the fields the pipeline reads."""

    ref: str | None = None
    reason: str = ""
    already_fixed: bool = False


class Git:
    """A stand-in for ``clone_repo`` that records the ref of every call and can refuse chosen refs."""

    def __init__(self, refuse=()):
        self.calls: list[tuple] = []
        self.refuse = set(refuse)

    def __call__(self, ref, dest, **kwargs):
        git_ref = kwargs.get("git_ref")
        self.calls.append((dest, kwargs))
        if git_ref in self.refuse:
            dest.mkdir(parents=True)  # git leaves a half-made directory behind on some failures
            (dest / "partial").write_text("x")
            raise RuntimeError(f"git clone failed (exit 128) for {git_ref}:\nRemote branch {git_ref} not found")
        dest.mkdir(parents=True)
        return dest / "repo"

    @property
    def refs(self) -> list:
        return [kwargs.get("git_ref") for _, kwargs in self.calls]


@pytest.fixture
def git(monkeypatch):
    clone = Git()
    monkeypatch.setattr(pipeline_module, "clone_repo", clone)

    def parse(url):
        if "/issues/" not in url:
            raise ValueError(f"not an issue URL: {url}")  # like the real parse_issue_url, for a repository URL
        return IssueRef("acme", "calc", 7, url)

    monkeypatch.setattr(pipeline_module, "parse_issue_url", parse)
    monkeypatch.setattr(
        pipeline_module, "fetch_issue", lambda ref: IssueRef(ref.owner, ref.repo, ref.number, ref.url, title="add is wrong", body="b")
    )
    return clone


class Resolver:
    """A ``resolve_base_ref`` that returns (or raises) what it is told and records the issues it was asked about."""

    def __init__(self, result=None, error: Exception | None = None):
        self.result, self.error, self.asked = result, error, []

    def __call__(self, issue):
        self.asked.append(issue)
        if self.error:
            raise self.error
        return self.result


def ingest(tmp_path, resolver=None, notes=None, **kwargs):
    notify = (lambda level, message: notes.append((level, message))) if notes is not None else None
    pipeline = RepoPipeline({}, tmp_path, base_ref_resolver=resolver, notify=notify)
    return pipeline.ingest(kwargs.pop("issue_url", URL), **kwargs)


# ---- 1. the explicit ref ---------------------------------------------------------------------------------


def test_an_explicit_ref_is_checked_out_and_the_resolver_is_never_asked(tmp_path, git):
    resolver = Resolver(Found("abc123"))
    ingested = ingest(tmp_path, resolver, git_ref="v2.31.0")
    assert git.refs == ["v2.31.0"]
    assert resolver.asked == []
    assert ingested.checkout == Checkout("v2.31.0", "the ref you asked for (git_ref)", already_fixed=False)


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_explicit_ref_counts_as_none_given(tmp_path, git, blank):
    resolver = Resolver(Found("abc123"))
    ingest(tmp_path, resolver, git_ref=blank)
    assert git.refs == ["abc123"] and len(resolver.asked) == 1


def test_an_explicit_ref_is_trimmed(tmp_path, git):
    assert ingest(tmp_path, None, git_ref="  main~3 \n").checkout.ref == "main~3"


# ---- 2. resolve_base_ref -----------------------------------------------------------------------------------


def test_the_resolver_is_given_the_fetched_issue(tmp_path, git):
    resolver = Resolver(Found("abc123"))
    ingest(tmp_path, resolver)
    (asked,) = resolver.asked
    assert (asked.owner, asked.repo, asked.number, asked.title) == ("acme", "calc", 7, "add is wrong")


def test_a_plain_string_from_the_resolver_is_the_ref(tmp_path, git):
    ingested = ingest(tmp_path, Resolver("9fdbf06"))
    assert git.refs == ["9fdbf06"]
    assert ingested.checkout.ref == "9fdbf06" and ingested.checkout.already_fixed is False
    assert "resolve_base_ref" in ingested.checkout.reason


def test_a_result_object_supplies_ref_reason_and_already_fixed(tmp_path, git):
    found = Found("9fdbf06", reason="parent of the merge commit of PR #12", already_fixed=True)
    ingested = ingest(tmp_path, Resolver(found))
    assert git.refs == ["9fdbf06"]
    assert ingested.checkout == Checkout("9fdbf06", "parent of the merge commit of PR #12", already_fixed=True)


def test_a_mapping_result_works_like_an_object(tmp_path, git):
    ingested = ingest(tmp_path, Resolver({"ref": "v1.0", "reason": "tag before the fix", "already_fixed": True}))
    assert ingested.checkout == Checkout("v1.0", "tag before the fix", already_fixed=True)


@pytest.mark.parametrize("name", ["ref", "base_ref", "base_sha", "sha"])
def test_the_ref_may_be_called_any_of_the_usual_names(tmp_path, git, name):
    ingested = ingest(tmp_path, Resolver({name: "abc123"}))
    assert ingested.checkout.ref == "abc123" and git.refs == ["abc123"]


# ---- 3. the default branch ---------------------------------------------------------------------------------


def test_a_resolver_that_finds_nothing_leaves_the_default_branch(tmp_path, git):
    ingested = ingest(tmp_path, Resolver(None))
    assert git.calls[0][1] == {}, "the default branch is a plain clone: no git_ref argument at all"
    assert ingested.checkout.ref is None and ingested.checkout.already_fixed is False
    assert "default branch" in ingested.checkout.reason and "no base revision" in ingested.checkout.reason


@pytest.mark.parametrize("empty", ["", "  ", {"ref": ""}, Found(ref=None)])
def test_an_empty_ref_from_the_resolver_is_no_ref(tmp_path, git, empty):
    assert ingest(tmp_path, Resolver(empty)).checkout.ref is None and git.refs == [None]


def test_already_fixed_without_a_ref_is_kept_so_the_report_can_warn(tmp_path, git):
    found = Found(ref=None, reason="fixed by PR #12 but its base commit is unknown", already_fixed=True)
    ingested = ingest(tmp_path, Resolver(found))
    assert git.refs == [None]
    assert ingested.checkout == Checkout(None, "fixed by PR #12 but its base commit is unknown", already_fixed=True)


def test_a_resolver_that_is_not_there_falls_back_to_the_default_branch(tmp_path, git, monkeypatch):
    monkeypatch.delattr("anvil.repo.ingest.resolve_base_ref", raising=False)
    ingested = ingest(tmp_path, None)
    assert git.refs == [None]
    assert "not available" in ingested.checkout.reason


def test_the_module_level_resolver_is_used_when_none_is_injected(tmp_path, git, monkeypatch):
    seen = Resolver(Found("abc123"))
    monkeypatch.setattr("anvil.repo.ingest.resolve_base_ref", seen, raising=False)
    assert ingest(tmp_path, None).checkout.ref == "abc123"
    assert len(seen.asked) == 1


def test_a_resolver_that_raises_is_reported_and_the_default_branch_is_used(tmp_path, git):
    notes: list = []
    ingested = ingest(tmp_path, Resolver(error=ConnectionError("api down")), notes)
    assert git.refs == [None] and ingested.checkout.ref is None
    assert "resolve_base_ref failed: ConnectionError" in ingested.checkout.reason
    assert [level for level, _ in notes] == ["warning"] and "api down" in notes[0][1]


def test_supplied_issue_text_has_no_issue_to_look_up_so_the_resolver_is_not_asked(tmp_path, git):
    resolver = Resolver(Found("abc123"))
    ingested = ingest(tmp_path, resolver, issue_url=REPO_URL, issue_text="add(2, 3) is -1")
    assert resolver.asked == [] and git.refs == [None]
    assert "no issue to look up" in ingested.checkout.reason


def test_an_explicit_ref_still_applies_to_supplied_issue_text(tmp_path, git):
    ingested = ingest(tmp_path, None, issue_url=REPO_URL, issue_text="add(2, 3) is -1", git_ref="v1")
    assert git.refs == ["v1"] and ingested.checkout.ref == "v1"


# ---- 4. a ref that cannot be checked out ---------------------------------------------------------------


def test_a_ref_that_cannot_be_cloned_is_replaced_by_the_default_branch(tmp_path, git):
    git.refuse = {"deadbeef"}
    notes: list = []
    found = Found("deadbeef", reason="parent of the fix", already_fixed=True)
    ingested = ingest(tmp_path, Resolver(found), notes)

    assert git.refs == ["deadbeef", None]
    assert git.calls[0][0] == git.calls[1][0], "the retry reuses the directory: the half-made clone was removed first"
    assert ingested.checkout.ref is None and ingested.checkout.already_fixed is True, "the default branch has the fix"
    assert "checking out 'deadbeef' failed" in ingested.checkout.reason and "not found" in ingested.checkout.reason
    assert [level for level, _ in notes] == ["warning"] and "deadbeef" in notes[0][1]


def test_an_explicit_ref_that_cannot_be_cloned_falls_back_too(tmp_path, git):
    git.refuse = {"nope"}
    ingested = ingest(tmp_path, None, git_ref="nope")
    assert git.refs == ["nope", None] and ingested.checkout.ref is None


def test_when_the_default_branch_cannot_be_cloned_either_the_error_reaches_the_orchestrator(tmp_path, git):
    git.refuse = {"deadbeef", None}
    with pytest.raises(RuntimeError, match="git clone failed"):
        ingest(tmp_path, Resolver(Found("deadbeef")))
    assert git.refs == ["deadbeef", None]


def test_the_root_returned_is_the_one_of_the_clone_that_worked(tmp_path, git):
    git.refuse = {"deadbeef"}
    ingested = ingest(tmp_path, Resolver(Found("deadbeef")))
    assert ingested.repo_root == git.calls[1][0].resolve() / "repo"
