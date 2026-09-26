import anvil.context
from anvil.context import ContextManager


def test_context_package_imports():
    assert anvil.context.__doc__


def test_build_messages_leads_with_the_phase_goal_as_system_message():
    ctx = ContextManager()
    ctx.add_message("user", "the issue")
    messages = ctx.build_messages("Localize the bug.")
    assert messages[0] == {"role": "system", "content": "Localize the bug."}
    assert messages[1] == {"role": "user", "content": "the issue"}


def test_history_is_kept_in_order_and_grows():
    ctx = ContextManager()
    ctx.add_message("user", "a")
    ctx.add_message("assistant", "b", pinned=True)
    ctx.add_message("user", "c")
    assert [m["content"] for m in ctx.build_messages("goal")[1:]] == ["a", "b", "c"]
    assert len(ctx) == 3


def test_a_new_phase_goal_replaces_the_system_message_but_keeps_history():
    ctx = ContextManager()
    ctx.add_message("user", "the issue")
    ctx.build_messages("first goal")
    messages = ctx.build_messages("second goal")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "second goal"


def test_tool_calling_fields_are_carried_through():
    calls = [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}]
    ctx = ContextManager()
    ctx.add_message("assistant", "", tool_calls=calls)
    ctx.add_message("tool", "3 matches", tool_call_id="c1")
    assistant, tool = ctx.build_messages("goal")[1:]
    assert assistant["tool_calls"] == calls
    assert tool == {"role": "tool", "content": "3 matches", "tool_call_id": "c1"}


def test_pinned_flag_is_not_sent_to_the_model():
    ctx = ContextManager()
    ctx.add_message("user", "keep me", pinned=True)
    assert "pinned" not in ctx.build_messages("goal")[1]


def test_mutating_a_built_prompt_does_not_corrupt_history():
    ctx = ContextManager()
    ctx.add_message("user", "original")
    ctx.build_messages("goal")[1]["content"] = "tampered"
    assert ctx.build_messages("goal")[1]["content"] == "original"
