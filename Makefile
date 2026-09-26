# ANVIL — Makefile
# Targets: setup | run | test | clean | bench
# Never write AI_API_KEY to disk.

# ---------------------------------------------------------------------------
# Toolchain detection
# ---------------------------------------------------------------------------
PYTHON ?= $(shell for p in python3.11 python3.12 python3.13 python3; do command -v $$p 2>/dev/null && break; done)
VENV   := .venv
VPY    := $(VENV)/bin/python
VPIP   := $(VENV)/bin/pip

# Optional: pass a GitHub issue URL via  make run ISSUE=https://...
ISSUE  ?=

.PHONY: setup run test clean bench

# ---------------------------------------------------------------------------
# setup — build the virtual environment and install all dependencies
# ---------------------------------------------------------------------------
setup:
	@# ── Python version guard ────────────────────────────────────────────
	@test -n "$(PYTHON)" || { \
	  echo "ERROR: no python3 found on PATH. Install Python 3.11+."; exit 1; }
	@$(PYTHON) -c \
	  'import sys; v=sys.version_info; \
	   sys.exit("ERROR: Python 3.11+ required, found "+sys.version.split()[0]) \
	   if v < (3,11) else None'
	@# ── Virtual environment ─────────────────────────────────────────────
	@echo "==> Creating virtual environment in $(VENV)/"
	$(PYTHON) -m venv $(VENV)
	@echo "==> Upgrading pip"
	$(VPY) -m pip install --quiet --upgrade pip
	@echo "==> Installing anvil[dev]"
	$(VPY) -m pip install --quiet -e ".[dev]"
	@# ── Optional tool checks (warnings, not failures) ────────────────────
	@if command -v git >/dev/null 2>&1; then \
	  echo "  ✓ git found: $$(git --version)"; \
	else \
	  echo "WARNING: git not found. Clone and diff operations will fail."; \
	fi
	@if command -v rg >/dev/null 2>&1; then \
	  echo "  ✓ ripgrep found: $$(rg --version | head -1)"; \
	else \
	  echo "WARNING: ripgrep (rg) not found. Grep tool will fall back to grep (slower)."; \
	fi
	@echo "==> Setup complete. Run:  export AI_API_KEY=<your-key> && make run"

# ---------------------------------------------------------------------------
# run — launch the TUI harness (requires AI_API_KEY in the environment)
# ---------------------------------------------------------------------------
run:
	@test -n "$(AI_API_KEY)" || { \
	  echo ""; \
	  echo "ERROR: AI_API_KEY is not set."; \
	  echo "  export AI_API_KEY=<your-api-key>  then re-run make run"; \
	  echo ""; \
	  exit 1; }
	@test -f $(VPY) || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	AI_API_KEY=$(AI_API_KEY) $(VPY) -m anvil \
	  $(if $(ISSUE),--issue $(ISSUE))

# ---------------------------------------------------------------------------
# test — run the full pytest suite
# ---------------------------------------------------------------------------
test:
	@test -f $(VPY) || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	$(VPY) -m pytest -q

# ---------------------------------------------------------------------------
# bench — run the benchmark suite against issues.yaml
# ---------------------------------------------------------------------------
bench:
	@test -f $(VPY) || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	@test -n "$(AI_API_KEY)" || { \
	  echo "ERROR: AI_API_KEY is not set."; exit 1; }
	AI_API_KEY=$(AI_API_KEY) $(VPY) bench/run_bench.py

# ---------------------------------------------------------------------------
# clean — remove caches, build artefacts, and run outputs
# ---------------------------------------------------------------------------
clean:
	@echo "==> Cleaning build artefacts and caches"
	rm -rf output build dist .pytest_cache
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" \
	  -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -not -path "./$(VENV)/*" \
	  -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "==> Done."
