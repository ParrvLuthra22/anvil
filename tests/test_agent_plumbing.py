"""Settings, budget and emitter: the small pieces the orchestrator is built from."""

import pytest

from anvil.agent.budget import Budget, BudgetExceeded
from anvil.agent.emitter import MESSAGE_EVENT_CHARS, PREVIEW_CHARS, Emitter
from anvil.agent.orchestrator import load_config
from anvil.agent.settings import AgentSettings
from anvil.events import AgentEvent, EventBus, Phase


# ---- settings -------------------------------------------------------------------------------


def test_defaults_apply_to_an_empty_mapping():
    assert AgentSettings.from_mapping({}) == AgentSettings()


def test_repo_config_yaml_parses_into_settings(monkeypatch):
    monkeypatch.delenv("AI_MODEL", raising=False)
    monkeypatch.delenv("AI_BASE_URL", raising=False)
    settings = AgentSettings.from_mapping(load_config())
    assert settings.max_steps_per_phase == 25
    assert settings.max_patch_attempts >= 1
    assert settings.output_dir == "output"


def test_unknown_keys_are_ignored():
    assert AgentSettings.from_mapping({"model": "m", "sandbox": "auto"}) == AgentSettings()


@pytest.mark.parametrize(
    "bad",
    [
        {"max_total_steps": "many"},
        {"max_total_steps": 0},
        {"max_total_steps": 2.5},
        {"max_patch_attempts": True},
        {"max_rollbacks": -1},
        {"wall_clock_seconds": -5},
        {"output_dir": ""},
        {"output_dir": 3},
    ],
)
def test_invalid_values_raise_naming_the_key(bad):
    key = next(iter(bad))
    with pytest.raises(ValueError, match=key):
        AgentSettings.from_mapping(bad)


def test_cost_estimate_uses_configured_prices():
    settings = AgentSettings.from_mapping(
        {"cost_per_million_prompt_tokens": 2, "cost_per_million_completion_tokens": 10}
    )
    assert settings.cost_estimate(1_000_000, 500_000) == pytest.approx(7.0)
    assert AgentSettings().cost_estimate(1_000_000, 1_000_000) == 0.0


# ---- budget ---------------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _budget(clock=None, **overrides) -> Budget:
    return Budget(AgentSettings(**overrides), clock=clock or FakeClock())


def test_charge_step_counts_calls():
    budget = _budget()
    budget.charge_step()
    budget.charge_step()
    assert budget.steps == 2


def test_step_budget_stops_the_next_call():
    budget = _budget(max_total_steps=2)
    budget.charge_step()
    budget.charge_step()
    with pytest.raises(BudgetExceeded) as info:
        budget.charge_step()
    assert info.value.kind == "steps"
    assert budget.steps == 2


def test_token_budget_stops_the_next_call():
    budget = _budget(max_tokens_total=1000)
    budget.charge_step()
    budget.add_tokens(1000)
    with pytest.raises(BudgetExceeded) as info:
        budget.charge_step()
    assert info.value.kind == "tokens"


def test_wall_clock_budget_uses_the_injected_clock():
    clock = FakeClock()
    budget = _budget(clock=clock, wall_clock_seconds=60)
    budget.charge_step()
    clock.now += 59.9
    budget.charge_step()
    clock.now += 0.2
    with pytest.raises(BudgetExceeded) as info:
        budget.charge_step()
    assert info.value.kind == "wall_clock"
    assert budget.elapsed == pytest.approx(60.1)


def test_negative_token_counts_are_ignored():
    budget = _budget()
    budget.add_tokens(-50)
    assert budget.tokens == 0


# ---- emitter --------------------------------------------------------------------------------


def _collect(emitter_action) -> list[AgentEvent]:
    bus = EventBus()
    queue = bus.subscribe()
    emitter_action(Emitter(bus, clock=lambda: 42.0))
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_events_carry_timestamp_and_current_phase():
    def act(em: Emitter) -> None:
        em.set_phase(Phase.LOCALIZE)
        em.tool_call("grep", {"pattern": "x"})

    phase_event, call_event = _collect(act)
    assert phase_event == AgentEvent(ts=42.0, type="phase", phase=Phase.LOCALIZE, data={"name": "localize"})
    assert call_event.phase is Phase.LOCALIZE
    assert call_event.data == {"tool": "grep", "args": {"pattern": "x"}}


def test_events_before_any_phase_have_no_phase():
    (event,) = _collect(lambda em: em.error("io", "disk full"))
    assert event.phase is None
    assert event.data == {"kind": "io", "message": "disk full"}


def test_usage_and_done_payloads_match_the_contract():
    def act(em: Emitter) -> None:
        em.usage(10, 5, 15, 0.25)
        em.done(resolved_confidence=0.9, patch_path="p", report_path="r", steps=3, tokens=15, seconds=1.5)

    usage, done = _collect(act)
    assert usage.data == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_estimate": 0.25}
    assert set(done.data) == {"resolved_confidence", "patch_path", "report_path", "steps", "tokens", "seconds"}


def test_long_text_is_truncated_in_events_only():
    long_text = "x" * (MESSAGE_EVENT_CHARS + 100)
    (msg,) = _collect(lambda em: em.message("assistant", long_text))
    assert len(msg.data["text"]) < len(long_text)
    assert msg.data["text"].endswith("[100 more chars]")

    (result,) = _collect(lambda em: em.tool_result("read_file", True, "y" * 2000))
    assert result.data["ok"] is True
    assert len(result.data["output_preview"]) < PREVIEW_CHARS + 40


def test_a_failing_subscriber_never_breaks_the_run():
    class ExplodingBus(EventBus):
        def emit(self, event: AgentEvent) -> None:
            raise RuntimeError("subscriber bug")

    Emitter(ExplodingBus()).set_phase(Phase.INGEST)
