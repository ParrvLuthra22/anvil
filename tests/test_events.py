from anvil.events import AgentEvent, EventBus, Phase


def test_phase_cycle_order():
    assert [p.value for p in Phase] == [
        "ingest", "profile", "understand", "localize", "reproduce",
        "patch", "verify", "review", "finalize",
    ]


def test_agent_event_holds_fields():
    event = AgentEvent(ts=1.0, type="phase", phase=Phase.INGEST, data={"name": "ingest"})
    assert event.phase is Phase.INGEST
    assert event.data == {"name": "ingest"}


def test_event_bus_exposes_contract_methods():
    assert callable(EventBus.emit) and callable(EventBus.subscribe)
