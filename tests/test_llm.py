import pytest

from anvil.llm.client import LLMResponse
from tests.mock_llm import MockLLM


def test_mock_llm_replays_script_in_order():
    first = LLMResponse(text="one", tool_calls=[], usage={})
    second = LLMResponse(text="two", tool_calls=[], usage={})
    llm = MockLLM([first, second])
    assert llm.chat([{"role": "user", "content": "hi"}]) is first
    assert llm.chat([]) is second


def test_mock_llm_raises_when_script_exhausted():
    llm = MockLLM([])
    with pytest.raises(RuntimeError):
        llm.chat([])
