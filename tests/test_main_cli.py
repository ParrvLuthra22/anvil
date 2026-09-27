"""Tests for CLI hooks that keep fixture runs on the real repository pipeline."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from anvil.__main__ import _local_repo_git_redirect


def test_local_fixture_redirect_uses_git_url_rewrite(tmp_path: Path, monkeypatch) -> None:
    """A local path is cloned through Git without replacing pipeline functions."""
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    source = tmp_path / "local fixture"
    source.mkdir()
    (source / "hello.py").write_text("MESSAGE = 'from fixture'\n", encoding="utf-8")
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "ANVIL test",
        "GIT_AUTHOR_EMAIL": "anvil-test@localhost",
        "GIT_COMMITTER_NAME": "ANVIL test",
        "GIT_COMMITTER_EMAIL": "anvil-test@localhost",
    }
    subprocess.run(["git", "init", "-q", str(source)], check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=source, check=True, env=env)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True, env=env)

    synthetic_url, git_env = _local_repo_git_redirect(source.as_uri())
    cloned = tmp_path / "clone"
    subprocess.run(["git", "clone", synthetic_url, str(cloned)], check=True, capture_output=True, env={**env, **git_env})

    assert (cloned / "hello.py").read_text(encoding="utf-8") == "MESSAGE = 'from fixture'\n"
