# ⚒ ANVIL — Autonomous Coding-Agent Harness

> Given a GitHub issue URL, ANVIL clones the repo, understands the issue,
> navigates the code, writes a failing reproduction, patches it, verifies the
> fix, self-reviews the diff, and writes a final report — all autonomously,
> using only a text LLM and a curated set of engineering tools.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/ParrvLuthra22/anvil.git && cd anvil

# 2. Set your API key (never written to disk)
export AI_API_KEY=<your-openai-compatible-key>

# 3. Install everything
make setup

# 4. Launch the TUI and paste a GitHub issue URL
make run

# Or pass the issue directly:
make run ISSUE=https://github.com/owner/repo/issues/42
```

> **First-time note:** `make setup` will warn (but not fail) if `git` or
> `ripgrep` (`rg`) are missing.  `git` is required for cloning; `rg` is
> optional — the grep tool falls back to system `grep`.

---

## Configuration

All settings live in [`config.yaml`](config.yaml). **No code edit is ever
needed** to change the model or provider.

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `gemini-2.0-flash` | Model name (any OpenAI-compatible) |
| `base_url` | Gemini endpoint | Provider base URL |
| `temperature` | `0` | Deterministic output |
| `max_steps_per_phase` | `25` | Steps before advancing phase |
| `max_total_steps` | `120` | Hard cap across all phases |
| `max_tokens_total` | `1 500 000` | Token budget for the whole run |
| `wall_clock_seconds` | `1800` | 30-minute wall-clock limit |
| `tool_output_char_cap` | `8000` | Max chars fed back per tool call |
| `sandbox` | `auto` | `auto` \| `worktree` \| `docker` |

### Environment overrides

| Variable | Effect |
|----------|--------|
| `AI_API_KEY` | **(required)** API key — never hard-coded anywhere |
| `AI_BASE_URL` | Override `base_url` without editing `config.yaml` |
| `AI_MODEL` | Override `model` without editing `config.yaml` |

---

## Entry points

```bash
# Full TUI (default)
python -m anvil
python -m anvil --issue <url>

# Headless (CI / scripting)
python -m anvil --issue <url> --headless

# Fallback when GitHub rate-limits you
python -m anvil --repo <repo-url> --issue-text "bug description" --headless

# Replay a saved trace without any API calls
python -m anvil replay output/trace.jsonl
python -m anvil replay output/trace.jsonl --speed 4   # 4× fast-forward
```

Every run writes three files to `output/`:

| File | Contents |
|------|----------|
| `output/patch.diff` | Unified diff of all changes |
| `output/report.md` | Human-readable summary |
| `output/trace.jsonl` | Full event log (JSONL) for replay / audit |

---

## TUI keyboard shortcuts

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Restart (clear log, reset phases) |
| `d` | Toggle diff preview pane |

---

## Running tests

```bash
make test           # runs pytest -q
```

All tests are offline — no network, no real LLM, no API key required.

---

## Benchmarks

```bash
AI_API_KEY=<key> make bench
```

Runs each issue in `bench/issues.yaml` headlessly and writes results to
`bench/results.md`.  Add your own issues by extending the YAML file.

---

## Project layout

```
anvil/
├── Makefile                   # setup · run · test · clean · bench
├── config.yaml                # model + budget settings (no secrets)
├── pyproject.toml             # package metadata + dependencies
├── .env.example               # AI_API_KEY=  (only empty placeholder)
│
├── src/anvil/
│   ├── __main__.py            # CLI entry point (Sneha)
│   ├── events.py              # AgentEvent · Phase · EventBus (Parrv)
│   ├── agent/
│   │   └── orchestrator.py   # Phase cycle driver (Parrv)
│   ├── context/               # Context window manager (Parrv)
│   ├── llm/
│   │   └── client.py         # OpenAI-compatible LLM client (Parrv)
│   ├── repo/
│   │   ├── ingest.py         # GitHub issue fetch + clone (Akshat)
│   │   └── profile.py        # Language/framework detection (Akshat)
│   ├── sandbox/
│   │   └── base.py           # Sandbox protocol + worktree impl (Akshat)
│   ├── tools/
│   │   ├── base.py           # Tool protocol + ToolResult (Akshat)
│   │   └── registry.py       # ToolRegistry (Akshat)
│   ├── trace/
│   │   └── recorder.py       # JSONL trace recorder/loader (Sneha)
│   └── tui/
│       └── app.py            # Textual TUI application (Sneha)
│
├── tests/
│   ├── mock_llm.py           # MockLLM + fake_event_stream (Sneha)
│   ├── test_trace.py         # Trace recorder tests (Sneha)
│   ├── test_tui.py           # TUI import/contract tests (Sneha)
│   └── test_*.py             # Per-module tests (Parrv / Akshat)
│
├── bench/
│   ├── issues.yaml           # Benchmark issue list (Sneha)
│   ├── run_bench.py          # Headless benchmark runner (Sneha)
│   └── results.md            # Auto-generated results table
│
└── docs/
    ├── ARCHITECTURE.md        # Component diagram, phase FSM, decision records
    └── EVALUATION.md          # How to reproduce a run and read a trace
```

---

## Decisions

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full decision log.
Key choices at a glance:

- **Phase state machine** — explicit phases give the LLM a focused prompt and a bounded budget per sub-task; they also make the TUI legible.
- **Reproduce-first** — writing a failing test before patching prevents placebo patches and gives a binary pass/fail signal for verification.
- **Worktree sandbox** — `git worktree` isolates every run without Docker overhead; Docker is auto-detected and used if available.
- **Event bus** — decouples the agent from the TUI and the trace recorder; replay is free.
- **JSONL trace** — append-only, crash-safe, streamable; one line per event.

---

## License

MIT — see `LICENSE`.
