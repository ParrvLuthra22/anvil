"""Registry that maps tool names to implementations and exposes their schemas."""

from __future__ import annotations

from anvil.tools.base import Tool, ToolResult


class ToolRegistry:
    """Holds the tools available to the agent.

    Tools are stored by name.  The registry exposes them in OpenAI
    function-calling format via :meth:`schemas`.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add *tool* under its ``name``.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool registered as *name*.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool named {name!r}. Available: {list(self._tools)}")
        return self._tools[name]

    def schemas(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format.

        Each entry follows the schema::

            {
                "type": "function",
                "function": {
                    "name": "<tool_name>",
                    "description": "<tool_description>",
                    "parameters": { ...JSON Schema... }
                }
            }
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def make_default_registry(profile=None) -> ToolRegistry:
    """Build and return a :class:`ToolRegistry` pre-loaded with all standard tools.

    Args:
        profile: Optional :class:`~anvil.repo.profile.RepoProfile` passed to
                 :class:`~anvil.tools.run_tests.RunTestsTool`.
    """
    from anvil.tools.list_dir import ListDirTool
    from anvil.tools.grep import GrepTool
    from anvil.tools.read_file import ReadFileTool
    from anvil.tools.edit_file import EditFileTool
    from anvil.tools.run_cmd import RunCmdTool
    from anvil.tools.run_tests import RunTestsTool
    from anvil.tools.git_diff import GitDiffTool
    from anvil.tools.outline import OutlineTool
    from anvil.tools.find import FindSymbolTool, FindReferencesTool
    from anvil.tools.related import RelatedTestsTool

    registry = ToolRegistry()
    registry.register(ListDirTool())
    registry.register(GrepTool())
    registry.register(ReadFileTool())
    registry.register(EditFileTool())
    registry.register(RunCmdTool())
    registry.register(RunTestsTool(profile=profile))
    registry.register(GitDiffTool())
    registry.register(OutlineTool())
    registry.register(FindSymbolTool())
    registry.register(FindReferencesTool())
    registry.register(RelatedTestsTool())
    return registry
