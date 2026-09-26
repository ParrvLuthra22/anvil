"""features.token_budgets through a whole scripted run: what the model is sent, and what the report says."""

from __future__ import annotations

from anvil.tools.base import ToolResult
from tests.fakes import (
    FakePipeline,
    FakeRegistry,
    FakeSandbox,
    FakeTool,
    call,
    default_tools,
    done,
    project_exec,
    project_files,
    reply,
)
from tests.test_orchestrator import execute, finalize, good_patch, reproduce, review_ok, understand, verify

BIG = "\n".join(
    ["import os", "", "class Widget:"] + [f"    def method_{i}(self):\n        return {i}\n" for i in range(150)]
)  # ~450 lines


def numbered_read(args: dict, sb: FakeSandbox) -> ToolResult:
    """read_file as the real tool prints it: ``%6d  line``."""
    path = str(args.get("path", ""))
    if path not in sb.files:
        return ToolResult(False, f"File not found: {path!r}")
    return ToolResult(True, "".join(f"{i:6d}  {line}\n" for i, line in enumerate(sb.files[path].splitlines(), start=1)))


def pipeline() -> FakePipeline:
    tools = [t for t in default_tools() if t.name != "read_file"] + [
        FakeTool("read_file", numbered_read, parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
    ]
    sandbox = FakeSandbox({**project_files(), "big.py": BIG}, on_exec=project_exec)
    return FakePipeline(sandbox, FakeRegistry(tools))


def localize_reading_the_big_file():
    return [reply(call("read_file", path="big.py")), reply(call("read_file", path="calc.py")), done("calc.py:2 uses a - b")]


def script():
    return understand() + localize_reading_the_big_file() + reproduce() + good_patch() + verify() + review_ok() + finalize()


def tool_message(run, marker_in: str) -> str:
    """The first tool message the model was sent whose content includes ``marker_in``'s first line of the big file."""
    for messages, _ in run.llm.calls:
        for message in messages:
            if message["role"] == "tool" and "class Widget:" in message["content"]:
                return message["content"]
    raise AssertionError("the model never saw the big file")


def test_with_token_budgets_on_the_model_is_sent_the_head_and_an_outline_of_a_big_file(tmp_path):
    run = execute(script(), tmp_path, pipeline=pipeline())
    seen = tool_message(run, "big.py")
    assert seen.startswith("[read_file shortened: ")
    assert "read_file(path='big.py', start=<first line>, end=<last line>)" in seen
    assert len(seen) <= 4000, "within the (tightened) tool output cap"
    assert "more definitions not listed]" in seen and "def method_98(self):" in seen, "80 outline entries, then a count"


def test_with_token_budgets_off_the_big_file_is_sent_as_before_with_only_the_ordinary_output_cap(tmp_path):
    run = execute(script(), tmp_path, pipeline=pipeline(), features={"token_budgets": False})
    seen = tool_message(run, "big.py")
    assert "read_file shortened" not in seen
    assert "lines omitted]" in seen, "the ordinary head-and-tail truncation of 8,000 characters"


def test_a_short_read_is_sent_whole_either_way(tmp_path):
    run = execute(script(), tmp_path, pipeline=pipeline())
    calc = next(
        m["content"] for messages, _ in run.llm.calls for m in messages if m["role"] == "tool" and "return a - b" in m["content"]
    )
    assert "shortened" not in calc and "     2      return a - b" in calc
