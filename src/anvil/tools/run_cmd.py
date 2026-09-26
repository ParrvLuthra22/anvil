"""run_cmd tool: run an arbitrary shell command inside the sandbox."""

from __future__ import annotations

from anvil.sandbox.base import Sandbox
from anvil.tools.base import Tool, ToolResult

_DEFAULT_TIMEOUT = 60   # seconds


class RunCmdTool:
    """Execute a shell command in the repository working directory."""

    name = "run_cmd"
    description = (
        "Execute a shell command in the repository root. "
        "stdout and stderr are captured and returned. "
        "Commands are killed after the specified timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Maximum seconds to wait before killing the command. "
                    f"Defaults to {_DEFAULT_TIMEOUT}."
                ),
                "default": _DEFAULT_TIMEOUT,
            },
        },
        "required": ["cmd"],
    }

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Delegate to *sandbox.exec* and format the result."""
        cmd: str = args.get("cmd", "")
        timeout: int = int(args.get("timeout", _DEFAULT_TIMEOUT))

        if not cmd:
            return ToolResult(ok=False, output="'cmd' argument is required.")

        result = sandbox.exec(cmd, timeout=timeout)

        parts: list[str] = []
        if result.stdout:
            parts.append(f"[stdout]\n{result.stdout}")
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        if result.timed_out:
            parts.append(f"[TIMED OUT after {timeout}s]")

        output = "\n".join(parts) if parts else "(no output)"
        ok = result.exit_code == 0 and not result.timed_out

        return ToolResult(
            ok=ok,
            output=output,
            meta={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "duration": round(result.duration, 2),
            },
        )
