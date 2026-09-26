"""Tool contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from anvil.sandbox.base import Sandbox


@dataclass
class ToolResult:
    """What a tool returns to the model; failures are values (``ok=False``), never exceptions."""

    ok: bool
    output: str
    meta: dict = field(default_factory=dict)


class Tool(Protocol):
    """A model-callable capability with a JSON-Schema-described argument object."""

    name: str
    description: str
    parameters: dict

    def run(self, args: dict, sandbox: Sandbox) -> ToolResult:
        """Execute the tool against ``sandbox`` with the model-supplied ``args``."""
        ...
