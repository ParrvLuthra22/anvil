from anvil.sandbox.base import ExecResult


def test_exec_result_fields():
    result = ExecResult(exit_code=0, stdout="ok", stderr="", timed_out=False, duration=0.1)
    assert result.exit_code == 0 and not result.timed_out
