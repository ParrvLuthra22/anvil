"""Which tool calls are runs of the repository's own tests.

VERIFY's evidence is the ``run_tests`` tool, but a model often runs the suite with ``run_cmd`` instead (``pytest tests/``,
``go test ./...``). Both are repository tests. A script the model wrote itself under ``.anvil/`` is not, even when a test
runner executes it: it is the repro, and a fix verified on it alone has not been checked against the repository's tests.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

_SEPARATORS = re.compile(r"&&|\|\||;|\||\n")
_ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPERS = {"env", "time", "sudo", "nice", "exec", "command", "npx", "xvfb-run", "poetry", "pipenv", "pdm", "hatch", "uv", "run", "bundle"}
_RUNNERS = {"pytest", "py.test", "tox", "nox", "jest", "vitest", "mocha", "ctest", "rspec", "phpunit"}
_PYTHON = re.compile(r"python[0-9.]*$")
_MODULE_RUNNERS = {"pytest", "unittest", "tox", "nose", "nose2"}
_PACKAGE_MANAGERS = {"npm", "yarn", "pnpm", "bun"}
_BUILD_TOOLS = {"mvn", "mvnw", "gradle", "gradlew"}
_SCRATCH_DIR = ".anvil"


def is_test_command(cmd: object) -> bool:
    """True if ``cmd`` (a shell command line) runs the repository's tests; a run of the model's own ``.anvil/`` scripts is not one."""
    if not isinstance(cmd, str) or _SCRATCH_DIR in cmd:
        return False
    return any(_segment_runs_tests(segment) for segment in _SEPARATORS.split(cmd))


def is_repository_test_run(tool: str, args: dict[str, Any]) -> bool:
    """True for a ``run_tests`` call and for a ``run_cmd`` call whose command runs the repository's tests."""
    if tool == "run_tests":
        return True
    return tool == "run_cmd" and is_test_command(args.get("cmd"))


def _segment_runs_tests(segment: str) -> bool:
    try:
        words = shlex.split(segment)
    except ValueError:
        words = segment.split()
    while words and (_ENV_ASSIGNMENT.match(words[0]) or words[0] in _WRAPPERS):
        words = words[1:]
    if not words:
        return False
    program = words[0].rsplit("/", 1)[-1]
    rest = words[1:]
    if program in _RUNNERS:
        return True
    if _PYTHON.match(program):
        return any(flag == "-m" and nxt in _MODULE_RUNNERS for flag, nxt in zip(rest, rest[1:]))
    if program == "coverage":
        return any(flag == "-m" and nxt in _MODULE_RUNNERS for flag, nxt in zip(rest, rest[1:]))
    if program in _PACKAGE_MANAGERS:
        return bool(rest) and (rest[0].split(":")[0] == "test" or (rest[0] == "run" and len(rest) > 1 and rest[1].split(":")[0] == "test"))
    if program in {"go", "dotnet", "bazel"}:
        return bool(rest) and rest[0] == "test"
    if program == "cargo":
        return bool(rest) and rest[0] in {"test", "nextest"}
    if program == "make":
        return any(word in {"test", "tests", "check"} for word in rest)
    if program in _BUILD_TOOLS:
        return any(word in {"test", "verify", "check"} for word in rest)
    return False
