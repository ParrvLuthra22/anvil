# ANVIL — Evaluation Guide

This document explains how a judge or evaluator can reproduce a run from
scratch and interpret the output artefacts.

---

## Prerequisites

```bash
# 1. Clone the repo
git clone https://github.com/ParrvLuthra22/anvil.git
cd anvil

# 2. Set the API key (the ONLY required secret)
export AI_API_KEY=<your-openai-compatible-key>

# 3. Install
make setup
```

`make setup` prints a ✓ for each tool found (`git`, `rg`) and a WARNING
(non-fatal) if `rg` is missing.

---

## Running a fresh evaluation

### Interactive TUI

```bash
make run
# Type or paste a GitHub issue URL into the input field, press Enter or Start.
```

### Unattended headless mode (evaluator flow)

```bash
make run ISSUE=https://github.com/owner/repo/issues/42
# Equivalent to:
AI_API_KEY=$AI_API_KEY .venv/bin/python -m anvil --issue <url>
```

Or directly:

```bash
.venv/bin/python -m anvil --issue https://github.com/owner/repo/issues/42 --headless
```

---

## Output artefacts

After every run (successful or not), three files are written to `output/`:

| File | Description |
|------|-------------|
| `output/patch.diff` | Unified diff of all source changes. Empty if no patch was found. |
| `output/report.md` | Human-readable summary: issue, root cause, patch rationale, confidence score. |
| `output/trace.jsonl` | Full JSONL event log — one JSON object per line, in chronological order. |

---

## Replaying a trace (no API key needed)

```bash
# 1× real-time
.venv/bin/python -m anvil replay output/trace.jsonl

# 4× fast-forward
.venv/bin/python -m anvil replay output/trace.jsonl --speed 4

# Instant (no delays)
.venv/bin/python -m anvil replay output/trace.jsonl --speed 0
```

The TUI renders the replay identically to a live run.  No API calls are made.

---

## Reading a trace file

Each line of `output/trace.jsonl` is a JSON object:

```json
{"ts": 1727366400.5, "type": "phase",    "phase": "ingest",   "data": {"name": "ingest"}}
{"ts": 1727366401.1, "type": "message",  "phase": "ingest",   "data": {"role": "assistant", "text": "…"}}
{"ts": 1727366402.3, "type": "tool_call","phase": "localize",  "data": {"tool": "grep", "args": {"pattern": "KeyError"}}}
{"ts": 1727366402.9, "type": "tool_result","phase":"localize", "data": {"tool": "grep", "ok": true, "output_preview": "…"}}
{"ts": 1727366403.0, "type": "llm_usage","phase": "patch",    "data": {"prompt_tokens": 1200, "completion_tokens": 340, "total_tokens": 1540, "cost_estimate": 0.000231}}
{"ts": 1727366404.5, "type": "done",     "phase": "finalize", "data": {"resolved_confidence": 0.92, "patch_path": "output/patch.diff", "report_path": "output/report.md", "steps": 47, "tokens": 38500, "seconds": 142.3}}
```

### Event types

| `type` | `data` fields | Description |
|--------|--------------|-------------|
| `phase` | `name` | Agent entered a new phase |
| `message` | `role`, `text` | LLM message (assistant or user) |
| `tool_call` | `tool`, `args` | Agent invoked a tool |
| `tool_result` | `tool`, `ok`, `output_preview` | Tool returned |
| `llm_usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_estimate` | Token usage for one LLM call |
| `error` | `kind`, `message` | A recoverable error occurred |
| `done` | `resolved_confidence`, `patch_path`, `report_path`, `steps`, `tokens`, `seconds` | Run complete |

### Useful one-liners

```bash
# Count phases reached
grep '"type":"phase"' output/trace.jsonl | wc -l

# Show all tool calls
grep '"type":"tool_call"' output/trace.jsonl | python3 -c \
  "import sys,json; [print(json.loads(l)['data']['tool']) for l in sys.stdin]"

# Total token usage
python3 -c "
import json, pathlib
evs = [json.loads(l) for l in pathlib.Path('output/trace.jsonl').read_text().splitlines() if l.strip()]
done = next((e for e in evs if e['type']=='done'), {})
print('tokens:', done.get('data',{}).get('tokens','n/a'))
"

# Check confidence
python3 -c "
import json, pathlib
for line in pathlib.Path('output/trace.jsonl').read_text().splitlines():
    ev = json.loads(line)
    if ev['type'] == 'done':
        print('confidence:', ev['data']['resolved_confidence'])
        break
"
```

---

## Verifying the patch

```bash
# View the diff
cat output/patch.diff

# Apply the patch to a fresh clone (for independent verification)
git clone --depth 1 https://github.com/owner/repo /tmp/verify
cd /tmp/verify
git apply /path/to/anvil/output/patch.diff
# Run the repo's own test suite
pytest   # or: npm test / go test ./... / cargo test
```

---

## Running the test suite (offline, no API key needed)

```bash
make test
# All tests pass with no network and no AI_API_KEY.
```

---

## Benchmark

```bash
AI_API_KEY=<key> make bench
# Results written to bench/results.md
```

See `bench/issues.yaml` to add or modify benchmark issues.
