"""Autouse fixtures for tests that build a default ``RepoPipeline`` and must stay offline.

Import the fixture into a test module (``# noqa: F401``); pytest then applies it to every test there.
"""

import pytest


@pytest.fixture(autouse=True)
def no_base_ref_lookup(monkeypatch):
    """A default ``RepoPipeline`` asks ``anvil.repo.ingest.resolve_base_ref`` (GitHub) which revision to check out."""
    monkeypatch.setattr("anvil.agent.pipeline._default_base_ref_resolver", lambda: None)
