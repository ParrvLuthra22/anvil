"""Which shell commands are runs of the repository's own tests (so that a `run_cmd pytest` in VERIFY counts as one)."""

from __future__ import annotations

import pytest

from anvil.agent.testcmd import is_repository_test_run, is_test_command


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest",
        "pytest tests/test_calc.py -q",
        "pytest tests/test_calc.py::test_add -x",
        "python -m pytest tests/",
        "python3 -m pytest -q",
        "python3.11 -m pytest",
        "/usr/bin/python3 -m pytest tests",
        "./.venv/bin/python -m pytest tests",
        ".venv/bin/pytest -q",
        "py.test tests",
        "python -m unittest discover",
        "python -m coverage run -m pytest",
        "coverage run -m pytest tests",
        "coverage run -m unittest",
        "tox -e py311",
        "nox -s tests",
        "uv run pytest tests",
        "poetry run pytest",
        "PYTHONPATH=src pytest tests",
        "env PYTHONPATH=src pytest tests",
        "time pytest",
        "cd sub && pytest",
        "npm test",
        "npm run test",
        "npm run test:unit",
        "yarn test",
        "pnpm test -- --watch=false",
        "npx jest",
        "npx vitest run",
        "mocha test/",
        "go test ./...",
        "go test -run TestFoo ./pkg/...",
        "cargo test",
        "cargo test --lib foo",
        "cargo nextest run",
        "mvn test",
        "mvn -q -pl core test -Dtest=FooTest",
        "./mvnw verify",
        "./gradlew test",
        "gradle check",
        "make test",
        "make -j4 check",
        "dotnet test",
        "bazel test //...",
        "ctest --output-on-failure",
        "bundle exec rspec",
        "rspec spec/",
        "phpunit",
        "false; pytest",
        "pytest tests | tail -20",
    ],
)
def test_commands_that_run_the_repositorys_tests(cmd):
    assert is_test_command(cmd), cmd


@pytest.mark.parametrize(
    "cmd",
    [
        "",
        "   ",
        "python .anvil/repro.py",
        "python3 .anvil/repro.py",
        "python -m pytest .anvil/repro.py",
        "pytest .anvil/repro.py -q",
        "cd .anvil && pytest",
        "node .anvil/repro.js",
        "go run .anvil/repro.go",
        "python -c 'from calc import add; print(add(2, 3))'",
        "grep -rn pytest .",
        "cat tests/test_calc.py",
        "echo pytest",
        "git diff",
        "ls tests",
        "pip install pytest",
        "pip install -e .",
        "npm install",
        "npm run build",
        "npm run lint",
        "cargo build",
        "cargo run",
        "go build ./...",
        "make build",
        "make install",
        "mvn package",
        "python setup.py build",
        "python -m pip install -U pytest",
        "sed -n 1,20p tests/test_calc.py",
        "rm -rf test",
        "coverage report",
        "coverage run script.py",
    ],
)
def test_commands_that_are_not(cmd):
    assert not is_test_command(cmd), cmd


def test_a_command_that_is_not_a_string_is_not_a_test_run():
    assert not is_test_command(None) and not is_test_command(42)  # type: ignore[arg-type]


def test_the_run_tests_tool_is_always_a_test_run_and_run_cmd_only_when_its_command_is():
    assert is_repository_test_run("run_tests", {})
    assert is_repository_test_run("run_tests", {"targets": ["tests/test_calc.py"]})
    assert is_repository_test_run("run_cmd", {"cmd": "pytest tests"})
    assert not is_repository_test_run("run_cmd", {"cmd": "python .anvil/repro.py"})
    assert not is_repository_test_run("run_cmd", {})
    assert not is_repository_test_run("read_file", {"path": "tests/test_calc.py"})
    assert not is_repository_test_run("grep", {"pattern": "pytest"})
