from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile


def test_issue_ref_defaults():
    ref = IssueRef(owner="o", repo="r", number=1, url="https://github.com/o/r/issues/1")
    other = IssueRef(owner="o", repo="r", number=2, url="https://github.com/o/r/issues/2")
    ref.comments.append("hi")
    assert ref.title == "" and ref.body == "" and other.comments == []


def test_repo_profile_notes_default():
    profile = RepoProfile(
        languages=["python"], primary_language="python",
        install_cmd=None, test_cmd="pytest", test_framework="pytest",
    )
    assert profile.notes == ""
