"""Registry that maps tool names to implementations and exposes their schemas."""

from __future__ import annotations

from anvil.tools.base import Tool


class ToolRegistry:
    """Holds the tools available to the agent."""

    def register(self, tool: Tool) -> None:
        """Add ``tool`` under its ``name``."""
        raise NotImplementedError

    def get(self, name: str) -> Tool:
        """Return the tool registered as ``name``."""
        raise NotImplementedError

    def schemas(self) -> list[dict]:
        """Return the tool definitions in OpenAI function-calling format."""
        raise NotImplementedError
