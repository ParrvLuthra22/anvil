"""run_tests now takes ``targets`` (a list); the report's check labels must read that and the older ``target`` too."""

from __future__ import annotations

import pytest

from anvil.agent.orchestrator import _test_targets
from anvil.agent.prompts import environment_note, PHASE_SPECS, WEAK_PHASE_SPECS
from anvil.events import Phase
from anvil.repo.profile import RepoProfile
from tests.fakes import call, done, reply
from tests.test_orchestrator import execute, finalize, good_patch, localize, reproduce, review_ok, understand


@pytest.mark.parametrize(
    "args, expected",
    [
        ({"targets": ["tests/test_calc.py"]}, "tests/test_calc.py"),
        ({"targets": ["tests/a.py::test_x", "tests/b.py"]}, "tests/a.py::test_x tests/b.py"),
        ({"target": "tests/test_calc.py"}, "tests/test_calc.py"),
        ({"targets": [], "target": "tests/old.py"}, "tests/old.py"),
        ({"targets": ["new.py"], "target": "old.py"}, "new.py"),
        ({}, ""),
        ({"targets": None}, ""),
        ({"target": "  padded.py  "}, "padded.py"),
        ({"targets": "tests/as_a_string.py"}, "tests/as_a_string.py"),
    ],
)
def test_the_targets_of_a_run_tests_call(args, expected):
    assert _test_targets(args) == expected


@pytest.mark.parametrize("arguments, label", [({"targets": ["tests/test_calc.py", "tests/test_more.py"]}, "run_tests tests/test_calc.py tests/test_more.py"), ({"target": "tests/test_calc.py"}, "run_tests tests/test_calc.py"), ({}, "run_tests")])
def test_the_report_names_what_the_tests_run_was_narrowed_to(tmp_path, arguments, label):
    script = (
        understand() + localize() + reproduce() + good_patch()
        + [reply(call("run_tests", **arguments)), done("repro and tests pass")] + review_ok() + finalize()
    )
    run = execute(script, tmp_path)
    assert f"- pass: {label}\n" in run.report


def test_the_prompts_no_longer_tell_the_model_to_pass_a_target_the_tool_calls_them_targets():
    verify = {PHASE_SPECS[Phase.VERIFY].system_prompt, WEAK_PHASE_SPECS[Phase.VERIFY].system_prompt}
    assert all("run_tests with a target" not in text and 'run_tests(target=' not in text for text in verify)
    assert 'run_tests(targets=["tests/test_calc.py"])' in WEAK_PHASE_SPECS[Phase.VERIFY].system_prompt
    note = environment_note(RepoProfile(["python"], "python", None, "pytest", "pytest"))
    assert "pass targets to narrow it" in note and "pass a target " not in note
