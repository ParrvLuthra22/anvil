# ANVIL — Makefile
# Targets: setup | run | test | clean | bench | demo
# Never write AI_API_KEY to disk.

# ---------------------------------------------------------------------------
# Toolchain detection
# Prefer newer Pythons first; any >= 3.11 is acceptable.
# ---------------------------------------------------------------------------
PYTHON ?= $(shell \
  for p in python3.13 python3.12 python3.11 python3 python; do \
    cmd=$$(command -v $$p 2>/dev/null) && \
    $$cmd -c 'import sys; sys.exit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null && \
    echo $$cmd && break; \
  done)

VENV   := .venv
VPY    := $(VENV)/bin/python
VPIP   := $(VENV)/bin/pip

# Optional: pass a GitHub issue URL via  make run ISSUE=https://...
ISSUE  ?=

.PHONY: setup run test clean bench demo

# ---------------------------------------------------------------------------
# setup — idempotent: creates venv only if missing, always syncs packages
# ---------------------------------------------------------------------------
setup:
	@# ── Python version guard ────────────────────────────────────────────
	@test -n "$(PYTHON)" || { \
	  echo ""; \
	  echo "ERROR: Python 3.11+ not found on PATH."; \
	  echo "  Install Python 3.11, 3.12, or 3.13 and ensure it is on PATH."; \
	  echo "  macOS: brew install python@3.12"; \
	  echo "  Linux: sudo apt install python3.12"; \
	  echo ""; \
	  exit 1; }
	@echo "==> Using Python: $(PYTHON) ($$($(PYTHON) --version))"
	@# ── Virtual environment (idempotent) ────────────────────────────────
	@if [ ! -f "$(VPY)" ]; then \
	  echo "==> Creating virtual environment in $(VENV)/"; \
	  $(PYTHON) -m venv $(VENV); \
	else \
	  echo "==> Virtual environment already exists — skipping creation"; \
	fi
	@echo "==> Upgrading pip"
	@$(VPY) -m pip install --quiet --upgrade pip
	@echo "==> Installing anvil[dev]"
	@$(VPY) -m pip install --quiet -e ".[dev]"
	@# ── Optional tool checks (warnings, not failures) ────────────────────
	@if command -v git >/dev/null 2>&1; then \
	  echo "  ✓ git: $$(git --version)"; \
	else \
	  echo "  WARNING: git not found. Clone and diff operations will fail."; \
	fi
	@if command -v rg >/dev/null 2>&1; then \
	  echo "  ✓ ripgrep: $$(rg --version | head -1)"; \
	else \
	  echo "  WARNING: ripgrep (rg) not found — grep tool will fall back to grep (slower)."; \
	fi
	@echo ""
	@echo "  Setup complete!"
	@echo "  Next: export AI_API_KEY=<your-key> && make run"
	@echo ""

# ---------------------------------------------------------------------------
# run — launch the TUI harness (requires AI_API_KEY in the environment)
# ---------------------------------------------------------------------------
run:
	@if [ -z "$(AI_API_KEY)" ]; then \
	  echo ""; \
	  echo "  ┌──────────────────────────────────────────────────────────┐"; \
	  echo "  │  ERROR: AI_API_KEY is not set.                           │"; \
	  echo "  │                                                          │"; \
	  echo "  │  Fix:  export AI_API_KEY=<your-openai-compatible-key>   │"; \
	  echo "  │        make run                                          │"; \
	  echo "  │                                                          │"; \
	  echo "  │  To run in demo mode (no key needed):  make demo        │"; \
	  echo "  └──────────────────────────────────────────────────────────┘"; \
	  echo ""; \
	  exit 1; \
	fi
	@test -f "$(VPY)" || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	AI_API_KEY=$(AI_API_KEY) $(VPY) -m anvil \
	  $(if $(ISSUE),--issue $(ISSUE))

# ---------------------------------------------------------------------------
# demo — run the TUI with the fake event stream (no API key needed)
# ---------------------------------------------------------------------------
demo:
	@test -f "$(VPY)" || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	$(VPY) -m anvil --demo

# ---------------------------------------------------------------------------
# test — run the full pytest suite (offline, no API key required)
# ---------------------------------------------------------------------------
test:
	@test -f "$(VPY)" || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	$(VPY) -m pytest -q

# ---------------------------------------------------------------------------
# bench — run the benchmark suite against issues.yaml
# ---------------------------------------------------------------------------
bench:
	@test -f "$(VPY)" || { \
	  echo "ERROR: virtual environment not found. Run: make setup"; exit 1; }
	@if [ -z "$(AI_API_KEY)" ]; then \
	  echo "ERROR: AI_API_KEY is not set."; exit 1; \
	fi
	AI_API_KEY=$(AI_API_KEY) $(VPY) bench/run_bench.py

# ---------------------------------------------------------------------------
# clean — remove caches, build artefacts, and run outputs
#         NEVER deletes files outside the repo directory.
# ---------------------------------------------------------------------------
clean:
	@echo "==> Cleaning build artefacts and caches"
	@# Remove only known safe paths — no wildcards that could escape the repo
	rm -rf output build dist .pytest_cache .mypy_cache .ruff_cache
	find . -maxdepth 6 -type d -name __pycache__ \
	  -not -path "./.git/*" -not -path "./$(VENV)/*" \
	  -prune -exec rm -rf {} + 2>/dev/null || true
	find . -maxdepth 6 -type d -name "*.egg-info" \
	  -not -path "./.git/*" -not -path "./$(VENV)/*" \
	  -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "==> Done."
