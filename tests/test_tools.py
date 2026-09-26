from anvil.tools.base import ToolResult
from anvil.tools.registry import ToolRegistry


def test_tool_result_meta_defaults_to_independent_dicts():
    a, b = ToolResult(ok=True, output="a"), ToolResult(ok=False, output="b")
    a.meta["k"] = 1
    assert b.meta == {}


def test_registry_exposes_contract_methods():
    for name in ("register", "get", "schemas"):
        assert callable(getattr(ToolRegistry, name))
