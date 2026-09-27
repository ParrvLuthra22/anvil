"""The README's Mermaid diagrams must be kinds GitHub renders as intended: `flowchart` or `stateDiagram-v2`.

`graph`, `stateDiagram` (the v1 syntax) and the rest render differently or not at all, and a diagram that fails to render shows
up in the README as a block of raw text. The architecture document carries the same two diagrams as the README.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_STARTS = ("flowchart", "stateDiagram-v2")
BLOCK = re.compile(r"^```mermaid[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)


def mermaid_blocks(markdown: str) -> list[str]:
    """The body of every ```mermaid fenced block in ``markdown``."""
    return [match.group(1) for match in BLOCK.finditer(markdown)]


def first_line(block: str) -> str:
    return next((line.strip() for line in block.splitlines() if line.strip()), "")


def starts_with_an_allowed_type(block: str) -> bool:
    words = first_line(block).split()
    return bool(words) and words[0] in ALLOWED_STARTS


def bad_blocks(markdown: str) -> list[str]:
    """The first line of each mermaid block that does not start with an allowed diagram type."""
    return [first_line(b) for b in mermaid_blocks(markdown) if not starts_with_an_allowed_type(b)]


# ---- the check itself fails when it should ---------------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["graph TD", "graph LR", "stateDiagram", "sequenceDiagram", "classDiagram", "erDiagram", "gantt", "pie", "mindmap"])
def test_the_check_flags_every_other_kind_of_diagram(first):
    assert bad_blocks(f"text\n\n```mermaid\n{first}\n    A --> B\n```\n") == [first]


@pytest.mark.parametrize("first", ["flowchart TB", "flowchart LR", "flowchart", "stateDiagram-v2", "  stateDiagram-v2  "])
def test_the_check_accepts_the_two_kinds(first):
    assert bad_blocks(f"```mermaid\n{first}\n    A --> B\n```\n") == []


def test_the_check_reads_every_block_not_just_the_first():
    text = "```mermaid\nflowchart TB\nA-->B\n```\n\ntext\n\n```mermaid\ngraph TD\nA-->B\n```\n"
    assert bad_blocks(text) == ["graph TD"] and len(mermaid_blocks(text)) == 2


def test_the_check_skips_a_leading_blank_line_and_ignores_other_code_fences():
    text = "```mermaid\n\nflowchart TB\nA-->B\n```\n\n```python\ngraph = 1\n```\n"
    assert bad_blocks(text) == [] and len(mermaid_blocks(text)) == 1


def test_an_empty_block_is_flagged():
    assert bad_blocks("```mermaid\n```\n") == [""]


# ---- the real files --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["README.md", "docs/ARCHITECTURE.md"])
def test_every_mermaid_block_starts_with_flowchart_or_statediagram_v2(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    assert mermaid_blocks(text), f"{path} has no mermaid block"
    assert bad_blocks(text) == [], f"{path}: a mermaid block does not start with flowchart or stateDiagram-v2"


def test_the_readme_has_an_architecture_section_with_the_two_diagrams_in_order():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "\n## Architecture\n" in text
    kinds = [first_line(b) for b in mermaid_blocks(text)]
    assert kinds == ["flowchart TB", "stateDiagram-v2"]
    section = text.split("\n## Architecture\n", 1)[1].split("\n## ", 1)[0]
    assert len(mermaid_blocks(section)) == 2, "both diagrams are inside the Architecture section"


def test_the_architecture_document_carries_the_same_two_diagrams_as_the_readme():
    readme = mermaid_blocks((ROOT / "README.md").read_text(encoding="utf-8"))
    architecture = mermaid_blocks((ROOT / "docs/ARCHITECTURE.md").read_text(encoding="utf-8"))
    assert architecture == readme, "docs/ARCHITECTURE.md and README.md must show the same diagrams"


def test_each_diagram_is_followed_by_three_or_four_sentences():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("\n## Architecture\n", 1)[1].split("\n---\n", 1)[0]
    after_each = [chunk.strip().split("\n\n", 1)[0] for chunk in BLOCK.sub("@@DIAGRAM@@", section).split("@@DIAGRAM@@")[1:]]
    assert len(after_each) == 2
    for prose in after_each:
        sentences = re.findall(r"[^.!?]+[.!?](?=\s|$)", re.sub(r"`[^`]*`", "X", prose))
        assert 3 <= len(sentences) <= 4, prose
