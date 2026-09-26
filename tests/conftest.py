"""Pytest configuration for the ANVIL test suite.

Sets asyncio_mode = "auto" so that all ``async def test_*`` functions
are automatically treated as asyncio tests without needing an explicit
``@pytest.mark.asyncio`` decorator on each one.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and set asyncio mode."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as asyncio (used by pytest-asyncio)",
    )
    # Enable auto-mode for pytest-asyncio so async test functions run
    # under asyncio without needing @pytest.mark.asyncio on each one.
    config.option.__dict__.setdefault("asyncio_mode", "auto")


# Ensure pytest-asyncio uses auto mode via ini option (preferred way)
# This is read before any collection starts.
collect_ignore_glob: list[str] = []
