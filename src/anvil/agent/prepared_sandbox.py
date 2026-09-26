"""A sandbox whose commands run in a prepared environment.

``PreparedSandbox`` wraps any ``Sandbox`` and changes nothing about files, diffs, checkpoints or
rollbacks; it only decides how commands start. Every command, whether the model asked for it or
the harness runs it (the repro re-run), gets:

* ``PYTHONDONTWRITEBYTECODE=1``. Python trusts a cached ``.pyc`` when the source's mtime (in whole
  seconds) and size are unchanged, and the classic bug fix (``-`` to ``+``, ``<`` to ``>``) keeps the
  size. An edit within the second of a previous run would execute the *old* code, and the repro
  would keep failing for a patch that is right.
* stdin from ``/dev/null``. Otherwise a command that reads input (a REPL, ``cat``, a prompt) blocks on
  the terminal the TUI owns until its timeout.
* optionally, the dependency venv first on ``PATH`` (``venv_dir``, relative to the repository root), so
  that ``pytest`` and ``python3`` are the ones the dependencies were installed for.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from anvil.sandbox.base import ExecResult, Sandbox


class PreparedSandbox:
    """Delegates everything to ``inner``, and starts each command in the prepared environment."""

    def __init__(self, inner: Sandbox, *, venv_dir: str | None = None) -> None:
        self._inner = inner
        self._venv_dir = venv_dir

    @property
    def root(self) -> Path:
        """The repository working directory of the wrapped sandbox."""
        return self._inner.root

    def exec(self, cmd: str, timeout: int = 120) -> ExecResult:
        """Run ``cmd`` (a shell command line) in the prepared environment."""
        return self._inner.exec(self.prepare(cmd), timeout)

    def prepare(self, cmd: str) -> str:
        """``cmd`` preceded by the statements that set up the environment (one per line, so comments are safe)."""
        setup = ["export PYTHONDONTWRITEBYTECODE=1"]
        if self._venv_dir:
            venv = shlex.quote(self._venv_dir)
            setup += [f'export VIRTUAL_ENV="$PWD"/{venv}', f'export PATH="$PWD"/{venv}/bin:"$PATH"']
        setup.append("exec </dev/null")
        return "\n".join([*setup, cmd])

    def read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        return self._inner.read_file(path, start, end)

    def write_file(self, path: str, content: str) -> None:
        self._inner.write_file(path, content)

    def diff(self) -> str:
        return self._inner.diff()

    def checkpoint(self, label: str) -> str:
        return self._inner.checkpoint(label)

    def rollback(self, ref: str) -> None:
        self._inner.rollback(ref)

    def close(self) -> None:
        self._inner.close()
