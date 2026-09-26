"""outline tool: lists classes and functions in a file with line numbers."""

from __future__ import annotations

import ast
import re

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

class OutlineTool:
    """List classes and functions in a file with line numbers."""

    name = "outline"
    description = (
        "Extract a high-level outline of classes and functions from a source file, "
        "including their line numbers. Uses AST for Python and regex for other languages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the source file relative to repo root.",
            },
        },
        "required": ["path"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        path = args.get("path", "")
        if not path:
            return ToolResult(ok=False, output="'path' argument is required.")

        try:
            content = sandbox.read_file(path)
        except Exception as e:
            return ToolResult(ok=False, output=f"Failed to read {path!r}: {e}")

        if path.endswith(".py"):
            try:
                tree = ast.parse(content, filename=path)
                lines = []
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        lines.append(f"{node.lineno}: class {node.name}")
                        for child in node.body:
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                lines.append(f"  {child.lineno}: def {child.name}(...)")
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        lines.append(f"{node.lineno}: def {node.name}(...)")
                if not lines:
                    return ToolResult(ok=True, output="No classes or functions found.")
                return ToolResult(ok=True, output="\n".join(lines))
            except SyntaxError:
                # Fallback to regex if syntax error
                pass

        # Regex fallback for non-python or syntactically invalid python
        lines = []
        pattern = re.compile(r"^\s*(class|def|func|function)\s+([a-zA-Z0-9_]+)", re.MULTILINE)
        for i, line in enumerate(content.splitlines(), 1):
            if pattern.search(line):
                lines.append(f"{i}: {line.strip()}")
        
        if not lines:
            return ToolResult(ok=True, output="No classes or functions found.")
        return ToolResult(ok=True, output="\n".join(lines))
