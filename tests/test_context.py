import json
import re

import pytest

import anvil.context
from anvil.context import ContextManager
from anvil.context.tokens import estimate_messages_tokens


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


# ---- helpers for the compaction tests -------------------------------------------------------


def payload(step: int, lines: int = 40, width: int = 60) -> str:
    return "\n".join(f"step {step} line {j:03d} " + "x" * width for j in range(lines))


def exchange(ctx: ContextManager, i: int, *, text: str | None = None, tool: str = "read_file", ok: bool = True) -> None:
    """One model step: an assistant tool call and the tool's result."""
    call_id = f"call_{i}"
    args = {"path": f"src/file_{i}.py", "start": 1, "end": 40}
    ctx.add_message(
        "assistant",
        f"reading file {i}",
        tool_calls=[{"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}],
    )
    ctx.add_message("tool", text if text is not None else payload(i), tool_call_id=call_id, ok=ok)


def assert_valid(messages: list[dict]) -> None:
    """What a strict OpenAI-compatible provider demands of the history."""
    assert messages[0]["role"] == "system"
    pending: set[str] = set()
    for message in messages[1:]:
        if pending:
            assert message["role"] == "tool", f"unanswered tool calls {pending}"
        if message["role"] == "assistant":
            pending |= {c["id"] for c in message.get("tool_calls") or []}
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending, "tool result without its call"
            pending.discard(message["tool_call_id"])
        assert isinstance(message["content"], str)
    assert not pending, f"unanswered tool calls {pending}"


def tool_contents(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


# ---- tool output truncation -----------------------------------------------------------------


def test_tool_output_is_truncated_to_the_cap_with_a_line_marker():
    ctx = ContextManager(tool_output_char_cap=1000)
    exchange(ctx, 1, text=payload(1, lines=200))
    (output,) = tool_contents(ctx.build_messages("goal"))
    assert len(output) <= 1000
    assert output.startswith("step 1 line 000") and output.rstrip().endswith("x" * 60)
    match = re.search(r"\[(\d+) lines omitted\]", output)
    assert match and int(match.group(1)) + output.count("step 1 line") == 200


def test_only_tool_output_is_truncated_not_prompts_or_assistant_text():
    ctx = ContextManager(tool_output_char_cap=500)
    ctx.add_message("user", "u" * 5000)
    ctx.add_message("assistant", "a" * 5000)
    user, assistant = ctx.build_messages("goal")[1:]
    assert len(user["content"]) == 5000 and len(assistant["content"]) == 5000


def test_output_within_the_cap_is_left_alone():
    ctx = ContextManager(tool_output_char_cap=1000)
    exchange(ctx, 1, text="short result")
    assert tool_contents(ctx.build_messages("goal")) == ["short result"]


# ---- pruning stale observations -------------------------------------------------------------


def test_observations_older_than_the_window_become_one_line_summaries_and_actions_stay():
    ctx = ContextManager(keep_steps=3)
    for i in range(1, 7):
        exchange(ctx, i, ok=(i != 2))
    messages = ctx.build_messages("goal")
    results = tool_contents(messages)
    assert results[3:] == [payload(4), payload(5), payload(6)]
    first, second, third = results[:3]
    prefix = "[output pruned] read_file(path='src/file_1.py', start=1, end=40) -> ok: step 1 line 000 xxx"
    assert first.startswith(prefix)
    assert second.startswith("[output pruned] read_file(path='src/file_2.py', start=1, end=40) -> failed: step 2")
    assert third.startswith("[output pruned] read_file(path='src/file_3.py', start=1, end=40) -> ok: step 3")
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert [m["content"] for m in assistants] == [f"reading file {i}" for i in range(1, 7)]
    assert all(m["tool_calls"] for m in assistants)
    assert_valid(messages)


def test_a_pruned_result_keeps_its_call_id_and_is_a_single_line():
    ctx = ContextManager(keep_steps=1)
    exchange(ctx, 1)
    exchange(ctx, 2)
    pruned = [m for m in ctx.build_messages("goal") if m["role"] == "tool"][0]
    assert pruned["tool_call_id"] == "call_1" and "\n" not in pruned["content"]
    assert len(pruned["content"]) < 300


def test_keep_steps_counts_model_steps_not_messages():
    ctx = ContextManager(keep_steps=2)
    exchange(ctx, 1)
    exchange(ctx, 2)
    assert tool_contents(ctx.build_messages("goal")) == [payload(1), payload(2)]
    exchange(ctx, 3)
    results = tool_contents(ctx.build_messages("goal"))
    assert results[0].startswith("[output pruned]") and results[1:] == [payload(2), payload(3)]


def test_pruning_is_permanent_and_repeatable():
    ctx = ContextManager(keep_steps=2)
    for i in range(1, 6):
        exchange(ctx, i)
    assert ctx.build_messages("goal") == ctx.build_messages("goal")
    assert ctx.token_estimate() == estimate_messages_tokens(ctx.build_messages("goal"))


def test_a_pinned_observation_is_never_pruned():
    ctx = ContextManager(keep_steps=1)
    ctx.add_message("tool", "the important result", pinned=True, tool_call_id="c0")
    for i in range(1, 5):
        exchange(ctx, i)
    assert tool_contents(ctx.build_messages("goal"))[0] == "the important result"


def test_a_result_without_a_known_call_is_still_summarised():
    ctx = ContextManager(keep_steps=1)
    ctx.add_message("tool", "orphan output\nmore", tool_call_id="ghost", name="mystery")
    exchange(ctx, 1)
    exchange(ctx, 2)
    assert tool_contents(ctx.build_messages("goal"))[0] == "[output pruned] mystery(): orphan output"


# ---- pinned items, phase goal, diff, repo map -----------------------------------------------

ISSUE = "# Issue: add() is wrong\n<issue>add(1, 2) returns -1</issue>"
MAP = "\n".join(f"src/module_{i}.py" for i in range(120))


def synthetic_run(ctx: ContextManager, steps: int = 100, *, budget: int, goal: str = "Phase goal: PATCH") -> list[int]:
    """Drive ``steps`` model steps with big outputs; returns the estimated prompt size before every step."""
    sizes = []
    for i in range(1, steps + 1):
        messages = ctx.build_messages(goal)
        assert_valid(messages)
        sizes.append(estimate_messages_tokens(messages))
        assert sizes[-1] <= budget, f"step {i}: prompt ~{sizes[-1]} tokens exceeds the {budget} budget"
        exchange(ctx, i, text=payload(i, lines=80, width=80))
    return sizes


@pytest.mark.parametrize("summarizer", ["none", "works", "fails", "empty"])
@pytest.mark.parametrize("budget,cap,keep", [(6000, 4000, 8), (12000, 6000, 6), (32000, 8000, 8)])
def test_the_prompt_stays_inside_the_budget_over_a_100_step_run(summarizer, budget, cap, keep):
    calls = []

    def works(transcript: str) -> str:
        calls.append(transcript)
        return "The model read several files and found nothing yet."

    def fails(transcript: str) -> str:
        calls.append(transcript)
        raise TimeoutError("provider down")

    impl = {"none": None, "works": works, "fails": fails, "empty": lambda t: "  "}[summarizer]
    ctx = ContextManager(max_context_tokens=budget, tool_output_char_cap=cap, keep_steps=keep, summarizer=impl)
    ctx.add_message("user", ISSUE, pinned=True)
    ctx.set_repo_map(MAP)
    synthetic_run(ctx, 100, budget=budget)
    if summarizer == "works":
        assert calls and len(calls) < 100


def test_pinned_items_survive_a_100_step_run_unchanged():
    budget = 6000
    ctx = ContextManager(max_context_tokens=budget, tool_output_char_cap=4000, keep_steps=8, summarizer=None)
    ctx.add_message("user", ISSUE, pinned=True)
    ctx.set_repo_map(MAP)
    ctx.set_diff("--- a/calc.py\n+++ b/calc.py\n@@ -1 +1 @@\n-return a - b\n+return a + b\n")
    ctx.add_message("user", "[LOCALIZE summary]\ncalc.py:2 uses a - b", pinned=True)
    for i in range(1, 101):
        messages = ctx.build_messages("Phase goal: PATCH")
        assert messages[0] == {"role": "system", "content": "Phase goal: PATCH"}
        contents = [m["content"] for m in messages]
        assert ISSUE in contents
        assert "[LOCALIZE summary]\ncalc.py:2 uses a - b" in contents
        assert any(c.startswith("[Latest diff of the working tree]") and "+return a + b" in c for c in contents)
        assert any(c.startswith("## Repository overview") and "src/module_0.py" in c for c in contents)
        exchange(ctx, i, text=payload(i, lines=80, width=80))


def test_the_estimate_matches_the_prompt_that_was_built():
    ctx = ContextManager()
    ctx.add_message("user", ISSUE, pinned=True)
    exchange(ctx, 1)
    messages = ctx.build_messages("goal")
    assert ctx.token_estimate() == estimate_messages_tokens(messages)


def test_the_estimate_is_zero_before_anything_is_built():
    assert ContextManager().token_estimate() == 0


def test_set_diff_replaces_the_previous_diff_and_an_empty_diff_removes_it():
    ctx = ContextManager()
    ctx.set_diff("diff one")
    ctx.set_diff("diff two")
    contents = [m["content"] for m in ctx.build_messages("goal")[1:]]
    assert len(contents) == 1 and "diff two" in contents[0] and "diff one" not in contents[0]
    ctx.set_diff("\n")
    assert ctx.build_messages("goal")[1:] == []


def test_a_huge_diff_is_truncated_to_the_output_cap():
    ctx = ContextManager(tool_output_char_cap=600)
    ctx.set_diff("\n".join(f"+line {i}" for i in range(500)))
    (message,) = ctx.build_messages("goal")[1:]
    assert len(message["content"]) < 700 and "lines omitted" in message["content"]


def test_the_repo_map_is_kept_whole_when_there_is_room():
    ctx = ContextManager()
    ctx.set_repo_map(MAP)
    (message,) = ctx.build_messages("goal")[1:]
    assert message["content"] == "## Repository overview\n" + MAP


def test_the_repo_map_is_trimmed_not_dropped_when_the_prompt_is_tight():
    ctx = ContextManager(max_context_tokens=1500, tool_output_char_cap=800, keep_steps=2)
    ctx.add_message("user", ISSUE, pinned=True)
    ctx.set_repo_map(MAP * 3)
    for i in range(1, 6):
        exchange(ctx, i, text=payload(i, lines=20, width=40))
    messages = ctx.build_messages("goal")
    (repo_map,) = [m["content"] for m in messages if m["content"].startswith("## Repository overview")]
    assert "src/module_0.py" in repo_map and "more lines omitted]" in repo_map
    assert len(repo_map) < len(MAP * 3)
    assert estimate_messages_tokens(messages) <= 1500


# ---- summarising ----------------------------------------------------------------------------


def filled(summarizer, *, steps: int = 20, budget: int = 6000, keep: int = 10) -> ContextManager:
    ctx = ContextManager(max_context_tokens=budget, tool_output_char_cap=3000, keep_steps=keep, summarizer=summarizer)
    ctx.add_message("user", ISSUE, pinned=True)
    for i in range(1, steps + 1):
        exchange(ctx, i, text=payload(i, lines=60, width=80))
    return ctx


def test_the_summariser_is_not_called_while_the_prompt_is_small():
    calls = []
    ctx = ContextManager(max_context_tokens=100_000, summarizer=lambda t: calls.append(t) or "s")
    for i in range(1, 30):
        exchange(ctx, i, text=payload(i, lines=10))
        ctx.build_messages("goal")
    assert calls == []


def test_past_the_threshold_the_oldest_turns_are_folded_into_one_summary_message():
    transcripts = []

    def summarizer(transcript: str) -> str:
        transcripts.append(transcript)
        return "Files 1 to 9 were read; nothing suspicious."

    ctx = filled(summarizer)
    messages = ctx.build_messages("goal")
    assert transcripts, "the summariser was never called"
    assert "file_1.py" in transcripts[0] and "reading file 1" in transcripts[0]
    summaries = [m for m in messages if m["content"].startswith("[Summary of earlier work]")]
    assert len(summaries) == 1 and "nothing suspicious" in summaries[0]["content"]
    assert messages[1]["content"] == ISSUE
    assert messages[-1]["role"] == "tool" and messages[-1]["tool_call_id"] == "call_20"
    assert "reading file 1" not in [m["content"] for m in messages]
    assert_valid(messages)
    assert ctx.token_estimate() <= 6000


def test_a_second_fold_hands_the_first_summary_to_the_summariser_so_it_rolls_up():
    transcripts = []

    def summarizer(transcript: str) -> str:
        transcripts.append(transcript)
        return f"summary number {len(transcripts)}"

    ctx = ContextManager(max_context_tokens=6000, tool_output_char_cap=3000, keep_steps=3, summarizer=summarizer)
    for i in range(1, 60):
        exchange(ctx, i, text=payload(i, lines=60, width=80))
        ctx.build_messages("goal")
    assert len(transcripts) >= 2
    assert "summary number 1" in transcripts[1]
    summaries = [m for m in ctx.build_messages("goal") if m["content"].startswith("[Summary of earlier work]")]
    assert len(summaries) == 1


def test_an_overlong_summary_is_clipped():
    ctx = filled(lambda t: "word " * 5000)
    (summary,) = [m for m in ctx.build_messages("goal") if m["content"].startswith("[Summary of earlier work]")]
    assert len(summary["content"]) < 2600


def test_a_summariser_failure_falls_back_to_pruning_without_raising():
    calls = []

    def broken(transcript: str) -> str:
        calls.append(1)
        raise RuntimeError("rate limited")

    ctx = filled(broken)
    messages = ctx.build_messages("goal")
    assert calls
    assert not [m for m in messages if m["content"].startswith("[Summary of earlier work]")]
    (digest,) = [m for m in messages if m["content"].startswith("[Earlier work was dropped")]
    assert "read_file(path='src/file_" in digest["content"]
    assert messages[1]["content"] == ISSUE
    assert_valid(messages)
    assert ctx.token_estimate() <= 6000


def test_without_a_summariser_old_turns_are_hard_pruned_into_a_digest_of_actions():
    ctx = filled(None)
    messages = ctx.build_messages("goal")
    (digest,) = [m for m in messages if m["content"].startswith("[Earlier work was dropped")]
    assert "Actions taken, oldest first" in digest["content"]
    listed = [int(n) for n in re.findall(r"file_(\d+)\.py", digest["content"])]
    assert len(listed) > 3 and listed == sorted(listed)
    assert "reading file 1" not in [m["content"] for m in messages]
    kept_raw = [m["tool_call_id"] for m in messages if m["role"] == "tool" and not m["content"].startswith("[")]
    assert kept_raw[-1] == "call_20" and int(kept_raw[0].removeprefix("call_")) > listed[-1]
    assert ctx.token_estimate() <= 6000


def test_a_failing_summariser_is_left_alone_for_a_few_steps_before_it_is_tried_again():
    calls = []

    def broken(transcript: str) -> str:
        calls.append(1)
        raise RuntimeError("down")

    ctx = ContextManager(max_context_tokens=6000, tool_output_char_cap=3000, keep_steps=3, summarizer=broken)
    per_step = []
    for i in range(1, 40):
        exchange(ctx, i, text=payload(i, lines=60, width=80))
        ctx.build_messages("goal")
        per_step.append(len(calls))
    first = per_step.index(1)
    assert per_step[first : first + 5] == [1] * 5
    assert len(calls) < 39 - first


def test_a_summariser_that_recovers_is_used_again():
    state = {"fail": True, "calls": 0}

    def flaky(transcript: str) -> str:
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("down")
        return "recovered summary"

    ctx = ContextManager(max_context_tokens=6000, tool_output_char_cap=3000, keep_steps=3, summarizer=flaky)
    for i in range(1, 30):
        if i == 15:
            state["fail"] = False
        exchange(ctx, i, text=payload(i, lines=60, width=80))
        ctx.build_messages("goal")
    assert any("recovered summary" in m["content"] for m in ctx.build_messages("goal"))


def test_the_transcript_given_to_the_summariser_is_bounded():
    sizes = []
    ctx = ContextManager(
        max_context_tokens=60_000, tool_output_char_cap=8000, keep_steps=100, summarizer=lambda t: sizes.append(len(t)) or "s"
    )
    for i in range(1, 60):
        exchange(ctx, i, text=payload(i, lines=90, width=90))
        ctx.build_messages("goal")
    assert sizes and max(sizes) <= 24_500


# ---- phases ---------------------------------------------------------------------------------


def test_ending_a_phase_replaces_its_turns_with_the_summary():
    ctx = ContextManager()
    ctx.add_message("user", ISSUE, pinned=True)
    ctx.begin_phase("Begin phase LOCALIZE.")
    exchange(ctx, 1)
    exchange(ctx, 2)
    ctx.end_phase("[LOCALIZE summary]\ncalc.py:2", pinned=True)
    contents = [m["content"] for m in ctx.build_messages("goal")[1:]]
    assert contents == [ISSUE, "[LOCALIZE summary]\ncalc.py:2"]


def test_the_next_phase_starts_from_the_summary_not_the_old_transcript():
    ctx = ContextManager()
    ctx.add_message("user", ISSUE, pinned=True)
    ctx.begin_phase("Begin phase LOCALIZE.")
    exchange(ctx, 1)
    ctx.end_phase("[LOCALIZE summary]\ncalc.py:2", pinned=True)
    ctx.begin_phase("Begin phase PATCH.")
    exchange(ctx, 2)
    messages = ctx.build_messages("goal")
    contents = [m["content"] for m in messages[1:]]
    assert contents[:3] == [ISSUE, "[LOCALIZE summary]\ncalc.py:2", "Begin phase PATCH."]
    assert not any("file_1.py" in c for c in contents if not c.startswith("[LOCALIZE"))
    assert_valid(messages)


def test_only_the_ended_phase_is_compressed():
    ctx = ContextManager()
    ctx.begin_phase("Begin phase A.")
    exchange(ctx, 1)
    ctx.end_phase("[A summary]")
    ctx.begin_phase("Begin phase B.")
    exchange(ctx, 2)
    ctx.end_phase("[B summary]")
    assert [m["content"] for m in ctx.build_messages("goal")[1:]] == ["[A summary]", "[B summary]"]


def test_an_unpinned_phase_summary_is_ordinary_history_but_a_pinned_one_outlives_pressure():
    ctx = ContextManager(max_context_tokens=3000, tool_output_char_cap=2000, keep_steps=2)
    ctx.begin_phase("Begin phase A.")
    ctx.end_phase("[A summary] unpinned")
    ctx.begin_phase("Begin phase B.")
    ctx.end_phase("[B summary] pinned", pinned=True)
    for i in range(1, 40):
        exchange(ctx, i, text=payload(i, lines=50, width=60))
        messages = ctx.build_messages("goal")
    contents = [m["content"] for m in messages]
    assert "[B summary] pinned" in contents
    assert "[A summary] unpinned" not in contents
    assert estimate_messages_tokens(messages) <= 3000


def test_ending_a_phase_keeps_the_pinned_diff_and_map_and_folded_summaries():
    ctx = ContextManager(max_context_tokens=3000, tool_output_char_cap=2000, keep_steps=2)
    ctx.set_repo_map("a.py\nb.py")
    ctx.begin_phase("Begin phase A.")
    ctx.set_diff("+fix")
    for i in range(1, 30):
        exchange(ctx, i, text=payload(i, lines=50, width=60))
        ctx.build_messages("goal")
    ctx.end_phase("[A summary]")
    contents = [m["content"] for m in ctx.build_messages("goal")[1:]]
    assert any(c.startswith("## Repository overview") for c in contents)
    assert any(c.startswith("[Latest diff") for c in contents)
    assert contents[-1] == "[A summary]"
    assert any(c.startswith("[Earlier work was dropped") for c in contents)
    assert not any(c.startswith("reading file") for c in contents)


def test_the_phase_kickoff_survives_pruning_pressure_while_the_phase_runs():
    kickoff = "Begin phase PATCH.\n\nRepro command: `python .anvil/repro.py`. Before any fix it fails with: boom"
    ctx = ContextManager(max_context_tokens=2500, tool_output_char_cap=1500, keep_steps=2)
    ctx.add_message("user", ISSUE, pinned=True)
    ctx.begin_phase(kickoff)
    for i in range(1, 40):
        exchange(ctx, i, text=payload(i, lines=40, width=60))
        messages = ctx.build_messages("goal")
        assert kickoff in [m["content"] for m in messages]
        assert estimate_messages_tokens(messages) <= 2500


def test_a_phase_that_was_never_ended_stops_being_protected_when_the_next_one_begins():
    ctx = ContextManager(max_context_tokens=2500, tool_output_char_cap=1500, keep_steps=2)
    ctx.begin_phase("Begin phase A (unfinished).")
    ctx.begin_phase("Begin phase B.")
    for i in range(1, 40):
        exchange(ctx, i, text=payload(i, lines=40, width=60))
        messages = ctx.build_messages("goal")
    contents = [m["content"] for m in messages]
    assert "Begin phase A (unfinished)." not in contents and "Begin phase B." in contents


def test_ending_without_a_phase_only_appends_the_summary():
    ctx = ContextManager()
    ctx.add_message("user", "earlier")
    ctx.end_phase("[summary]")
    assert [m["content"] for m in ctx.build_messages("goal")[1:]] == ["earlier", "[summary]"]


def test_a_second_end_phase_does_not_touch_messages_added_after_the_first():
    ctx = ContextManager()
    ctx.begin_phase("Begin.")
    ctx.end_phase("[one]")
    ctx.add_message("user", "between phases")
    ctx.end_phase("[two]")
    assert [m["content"] for m in ctx.build_messages("goal")[1:]] == ["[one]", "between phases", "[two]"]


# ---- the history stays valid whatever gets dropped ------------------------------------------


def test_tool_calls_and_their_results_are_dropped_together():
    ctx = ContextManager(max_context_tokens=2000, tool_output_char_cap=1200, keep_steps=2)
    ctx.add_message("user", ISSUE, pinned=True)
    for i in range(1, 60):
        exchange(ctx, i, text=payload(i, lines=30, width=60))
        assert_valid(ctx.build_messages("goal"))


def test_an_assistant_reply_with_several_tool_calls_is_kept_or_dropped_as_a_unit():
    ctx = ContextManager(max_context_tokens=2000, tool_output_char_cap=1200, keep_steps=2)
    for i in range(1, 40):
        ids = [f"c{i}a", f"c{i}b"]
        ctx.add_message(
            "assistant",
            "",
            tool_calls=[{"id": c, "type": "function", "function": {"name": "grep", "arguments": "{}"}} for c in ids],
        )
        for c in ids:
            ctx.add_message("tool", payload(i, lines=30), tool_call_id=c, ok=True)
        assert_valid(ctx.build_messages("goal"))


def test_a_prompt_that_cannot_fit_is_still_returned_intact_rather_than_broken():
    ctx = ContextManager(max_context_tokens=50)
    ctx.add_message("user", "x" * 4000, pinned=True)
    exchange(ctx, 1)
    messages = ctx.build_messages("goal")
    assert messages[1]["content"] == "x" * 4000
    assert_valid(messages)
