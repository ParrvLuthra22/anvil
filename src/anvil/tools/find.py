"""find tools: find_symbol and find_references using ripgrep."""

from __future__ import annotations

import shutil
import subprocess

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_MAX_MATCHES = 50
_SEARCH_TIMEOUT = 30  # seconds

def _run_rg(sandbox: Sandbox, pattern: str, is_regex: bool = True) -> ToolResult:
    use_rg = shutil.which("rg") is not None

    if use_rg:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never", f"--max-count={_MAX_MATCHES}"]
        if not is_regex:
            cmd.append("--fixed-strings")
        cmd.append("--")
        cmd.append(pattern)
        cmd.append(str(sandbox.root))
    else:
        # grep fallback
        cmd = ["grep", "-rn"]
        if not is_regex:
            cmd.append("-F")
        else:
            cmd.append("-E")  # extended regex
        cmd.append("--")
        cmd.append(pattern)
        cmd.append(str(sandbox.root))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SEARCH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output="Search timed out.")
    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, output=f"Search error: {exc}")

    if result.returncode not in (0, 1):
        return ToolResult(ok=False, output=f"Search failed: {result.stderr.strip()}")

    raw = result.stdout.strip()
    if not raw:
        return ToolResult(ok=True, output="No matches found.", meta={"match_count": 0})

    root_str = str(sandbox.root) + "/"
    lines = [line.replace(root_str, "") for line in raw.splitlines()]

    truncated = len(lines) > _MAX_MATCHES
    lines = lines[:_MAX_MATCHES]
    output = "\n".join(lines)
    if truncated:
        output += f"\n... [results capped at {_MAX_MATCHES} matches]"

    return ToolResult(
        ok=True,
        output=output,
        meta={"match_count": len(lines)},
    )


class FindSymbolTool:
    """Search for the definition of a symbol."""

    name = "find_symbol"
    description = (
        "Find the definition of a symbol (class, function, variable) across the repository. "
        "Returns at most 50 matches in 'file:line:content' format."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the symbol to find.",
            },
        },
        "required": ["name"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        name: str = args.get("name", "")
        if not name:
            return ToolResult(ok=False, output="'name' argument is required.")
        
        # Regex for common definition patterns:
        # class name, def name, func name, function name, name =, name:
        pattern = fr"^\s*(?:class|def|func|function|type|interface|var|let|const)\s+{name}\b|^\s*{name}\s*[:=]"
        return _run_rg(sandbox, pattern, is_regex=True)


class FindReferencesTool:
    """Search for usages of a symbol."""

    name = "find_references"
    description = (
        "Find all references/usages of a symbol across the repository. "
        "Returns at most 50 matches in 'file:line:content' format."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the symbol to find references for.",
            },
        },
        "required": ["name"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        name: str = args.get("name", "")
        if not name:
            return ToolResult(ok=False, output="'name' argument is required.")
        
        pattern = fr"\b{name}\b"
        return _run_rg(sandbox, pattern, is_regex=True)
