from anvil.trace.recorder import TraceRecorder


def test_recorder_exposes_contract_methods():
    assert callable(TraceRecorder.record)
    assert callable(TraceRecorder.load)
