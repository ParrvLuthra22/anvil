# Minimal Makefile; the TUI/docs owner will replace it. Targets: setup run test clean.

PYTHON ?= $(shell for p in python3.11 python3.12 python3.13 python3; do command -v $$p && break; done)
VENV   := .venv
VPY    := $(VENV)/bin/python
ARGS   ?=

.PHONY: setup run test clean

setup:
	@test -n "$(PYTHON)" || { echo "error: no python3 found on PATH"; exit 1; }
	@$(PYTHON) -c 'import sys; sys.exit("error: Python 3.11+ required, found " + sys.version.split()[0] if sys.version_info < (3, 11) else 0)'
	$(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --quiet --upgrade pip
	$(VPY) -m pip install --quiet -e ".[dev]"

run:
	$(VPY) -m anvil $(ARGS)

test:
	$(VPY) -m pytest

clean:
	rm -rf output build dist .pytest_cache src/*.egg-info
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -prune -exec rm -rf {} +
