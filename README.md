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

# 3. Install everything (idempotent — safe to re-run)
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

## Demo mode (no API key needed)

```bash
make demo
```

Plays a scripted fake event stream through the TUI so you can explore the
interface without any API key or network access.

---

## Configuration

All settings live in [`config.yaml`](config.yaml). **No code edit is ever
needed** to change the model or provider.  The table below documents every
key in `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `gemini-2.0-flash` | Model name passed to the OpenAI-compatible chat-completions endpoint |
| `base_url` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Provider base URL |
| `temperature` | `0` | Sampling temperature — 0 = deterministic output |
| `max_steps_per_phase` | `25` | LLM steps allowed before the orchestrator advances to the next phase |
| `max_total_steps` | `120` | Hard cap on total LLM steps across all phases |
| `max_tokens_total` | `1 500 000` | Token budget for the whole run; hitting it forces graceful FINALIZE |
| `wall_clock_seconds` | `1800` | 30-minute wall-clock limit; hitting it forces graceful FINALIZE |
| `tool_output_char_cap` | `8000` | Maximum characters of any single tool / shell output fed back to the model |
| `sandbox` | `auto` | Sandbox backend: `auto` (Docker if available, else worktree) \| `worktree` \| `docker` |

### Environment overrides (never put these in config.yaml)

| Variable | Effect |
|----------|--------|
| `AI_API_KEY` | **(required)** API key — read only from the environment, never hard-coded |
| `AI_BASE_URL` | Overrides `base_url` without editing `config.yaml` |
| `AI_MODEL` | Overrides `model` without editing `config.yaml` |

---

## Entry points

```bash
# Full TUI (default) — requires AI_API_KEY
python -m anvil
python -m anvil --issue <url>

# Headless — runs the REAL agent pipeline, no TUI, prints result to stdout
# Requires AI_API_KEY and the real orchestrator to be wired up.
python -m anvil --issue <url> --headless

# Fallback when GitHub rate-limits you (passes repo URL + description to pipeline)
python -m anvil --repo <repo-url> --issue-text "bug description" --headless

# Demo — fake event stream, no key needed
python -m anvil --demo

# Replay a saved trace without any API calls
python -m anvil replay output/trace.jsonl
python -m anvil replay output/trace.jsonl --speed 4   # 4× fast-forward
python -m anvil replay output/trace.jsonl --speed 0   # instant
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
| `q` | Quit (output files are preserved) |
| `r` | Restart (clear log, reset phases) |
| `d` | Toggle diff preview pane |
| `space` | Pause / resume replay |
| `→` / `l` | Step one event forward (replay) |
| `←` / `h` | Step one event back (replay) |
| `1` / `4` / `0` | Set replay speed to 1× / 4× / instant |

---

## Running tests

```bash
make test           # runs pytest -q (offline, no API key required)
```

All tests are offline — no network, no real LLM, no API key required.

---

## Benchmarks

```bash
AI_API_KEY=<key> make bench
```

Runs each issue in `bench/issues.yaml` headlessly and writes results to
`bench/results.md`.  Issues are verified by Akshat from `candidates.yaml`
before being added — no fabricated entries.

> **No results yet** — `bench/issues.yaml` will be populated once the real
> pipeline is operational and Akshat's verified candidates are confirmed.

---

## Project layout

```
anvil/
├── Makefile                   # setup · run · test · clean · bench · demo
├── config.yaml                # model + budget settings (no secrets)
├── pyproject.toml             # package metadata + dependencies
├── .env.example               # AI_API_KEY=  (only empty placeholder)
│
├── src/anvil/
│   ├── __main__.py            # CLI entry point (Sneha)
│   ├── events.py              # AgentEvent · Phase · EventBus (Parrv)
│   ├── agent/
│   │   ├── orchestrator.py   # Phase cycle driver + load_config (Parrv)
│   │   ├── loop.py           # Per-phase LLM + tool loop (Parrv)
│   │   ├── pipeline.py       # Phase sequencer (Parrv)
│   │   ├── budget.py         # Step / token / wall-clock budgets (Parrv)
│   │   ├── prompts.py        # Phase-specific system prompts (Parrv)
│   │   ├── outputs.py        # patch.diff / report.md writer (Parrv)
│   │   └── state.py          # Mutable run state (Parrv)
│   ├── context/
│   │   └── manager.py        # Context window manager + pruning (Parrv)
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
│   ├── test_makefile.py      # Makefile security tests (Sneha)
│   ├── test_trace.py         # Trace recorder tests (Sneha)
│   ├── test_tui.py           # TUI tests (Sneha)
│   └── test_*.py             # Per-module tests (Parrv / Akshat)
│
├── bench/
│   ├── issues.yaml           # Benchmark issue list (verified by Akshat)
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
