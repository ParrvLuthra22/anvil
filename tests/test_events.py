import asyncio
import threading

from anvil.events import AgentEvent, EventBus, Phase


def _event(n: int) -> AgentEvent:
    return AgentEvent(ts=float(n), type="message", phase=Phase.INGEST, data={"n": n})


def _drain(queue: asyncio.Queue) -> list[AgentEvent]:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


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


def test_every_subscriber_receives_every_event_in_order():
    bus = EventBus()
    first, second = bus.subscribe(), bus.subscribe()
    for n in range(3):
        bus.emit(_event(n))
    assert [e.data["n"] for e in _drain(first)] == [0, 1, 2]
    assert [e.data["n"] for e in _drain(second)] == [0, 1, 2]


def test_subscriber_only_sees_events_emitted_after_it_subscribed():
    bus = EventBus()
    bus.emit(_event(0))
    late = bus.subscribe()
    bus.emit(_event(1))
    assert [e.data["n"] for e in _drain(late)] == [1]


def test_emit_without_subscribers_is_a_noop():
    EventBus().emit(_event(0))


def test_emit_from_worker_thread_reaches_a_subscriber_on_the_running_loop():
    bus = EventBus()

    async def main() -> list[int]:
        queue = bus.subscribe()
        worker = threading.Thread(target=lambda: [bus.emit(_event(n)) for n in range(5)])
        worker.start()
        received = [(await asyncio.wait_for(queue.get(), timeout=2)).data["n"] for _ in range(5)]
        worker.join()
        return received

    assert asyncio.run(main()) == [0, 1, 2, 3, 4]


def test_emit_to_a_subscriber_whose_loop_has_closed_does_not_raise():
    bus = EventBus()

    async def subscribe_then_exit() -> None:
        bus.subscribe()

    asyncio.run(subscribe_then_exit())
    bus.emit(_event(0))


def test_subclass_can_override_both_methods_without_calling_super_init():
    class Recording(EventBus):
        def __init__(self) -> None:
            self.seen: list[AgentEvent] = []

        def emit(self, event: AgentEvent) -> None:
            self.seen.append(event)

        def subscribe(self) -> asyncio.Queue:
            return asyncio.Queue()

    bus = Recording()
    bus.emit(_event(0))
    assert len(bus.seen) == 1
