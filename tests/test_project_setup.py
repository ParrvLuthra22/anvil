"""What a fresh ``make setup && make test`` depends on: the dev dependencies and the pytest configuration."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_the_async_test_plugin_is_a_dev_dependency():
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    assert any(dep.lower().startswith("pytest-asyncio") for dep in dev), "the TUI tests are async def"
    assert any(dep.lower().startswith("pytest>=") for dep in dev)


def test_asyncio_mode_is_configured_where_pytest_reads_it():
    config = _pyproject()["tool"]["pytest"]["ini_options"]
    assert config["asyncio_mode"] == "auto"


def test_pytest_actually_runs_with_the_configured_mode(pytestconfig):
    assert pytestconfig.getini("asyncio_mode") == "auto"


async def test_async_tests_run_without_a_marker():
    """Fails with 'async def functions are not natively supported' when the plugin or the mode is missing."""
    import asyncio

    await asyncio.sleep(0)


def test_packaging_is_a_runtime_dependency_not_only_a_dev_one():
    """anvil.repo.deps parses requires-python with it, in the installed harness, not just under pytest."""
    runtime = _pyproject()["project"]["dependencies"]
    assert any(dep.lower().startswith("packaging") for dep in runtime)
    from packaging.specifiers import SpecifierSet

    assert "3.11" in SpecifierSet(">=3.8,<3.12")
