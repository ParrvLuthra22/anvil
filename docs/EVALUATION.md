# ANVIL Evaluation

This guide describes a reproducible live run, its artifacts, and benchmark
reporting. Live benchmark runs use the configured provider and consume API
quota. The scorer runs on disposable repository checkouts after the agent
finishes; it applies the agent patch first and only then applies the hidden
test patch and reads the test oracle fields.

## Setup and one issue

```sh
git clone https://github.com/ParrvLuthra22/anvil.git
cd anvil
export AI_API_KEY=<your-openai-compatible-key>
make setup
```

The key must be set in the environment. It is not stored in `config.yaml` or
passed on a command line. `AI_BASE_URL` and `AI_MODEL` optionally override the
configured endpoint and model.

Interactive run:

```sh
make run
# Paste a GitHub issue URL into the TUI.
```

Headless issue run:

```sh
.venv/bin/python -m anvil --issue https://github.com/owner/repo/issues/42 --headless
```

Fallback when issue retrieval is unavailable; `--ref` pins the base revision:

```sh
.venv/bin/python -m anvil --headless --repo https://github.com/owner/repo \
  --ref <base-commit> --issue-text "Describe the bug and expected behavior"
```

Each run attempts to write `output/patch.diff`, `output/report.md`, and
`output/trace.jsonl`. Errors and exhausted budgets still lead to finalization.

## Replay

Replay needs no API key and makes no model calls:

```sh
python -m anvil replay docs/sample_trace.jsonl
python -m anvil replay docs/sample_trace.jsonl --speed 0
```

The first command plays the trace in the TUI at recorded timing. `--speed 0`
plays without delays. Keyboard controls pause/resume, step forward/back, and
change speed.

## Benchmark protocol

`bench/instances.json` supplies the instance id, repository, base commit,
problem statement, recorded `test_cmd`, `test_patch`, `FAIL_TO_PASS`, and
`PASS_TO_PASS`. Only id, repository, base revision, and problem statement are
passed to the agent. The scorer uses the hidden test fields after the model run.

The runner is resumable: ids already present in `bench/results.jsonl` are
skipped. It is sequential by default. It supports per-instance timeouts,
`--jobs N`, `--limit N`, `--only <id>`, and `--label <name>`. A failed run whose
trace has a retryable LLM error waits 60 seconds and retries. HTTP 402 responses
that ask for in-flight requests to settle use up to 5 attempts with jittered
2-to-30-second waits; other 402 responses fail as out of credit. One run record
contains the id, resolved/category fields, step/token/second counts, label, and
artifact paths. `bench/score.py` adds the result category and resolved verdict;
`bench/analyze.py` summarizes failures, tokens per resolved instance, phases,
and trace error/recovery events.

Run and label a benchmark:

```sh
.venv/bin/python bench/run_bench.py --label before-docs --limit 1
.venv/bin/python bench/score.py
.venv/bin/python bench/analyze.py
```

Or run all configured instances and print the summary:

```sh
make bench
```

To use a label with the full pipeline, pass it to `run_bench.py` before running
the scorer and analyzer. `--only <id>` restricts the run to one known instance.
Use a new results file when intentionally rerunning completed IDs, because
resumption is keyed by completed instance id.

## Early Results

These real runs used **Qwen3-Coder-30B via OpenRouter with default settings**.
They are early results, **not a pass rate**.

| Run | Result |
|---|---|---|
| Toy repository | 2 of 2 patches correct |
| `pallets__flask-4045` | Unresolved: the patch used `assert`, but the hidden test expects `ValueError` |
| `pytest-dev__pytest-11143` | Empty patch: `edit_file` could not apply edits; fix in progress |

## Known Limitations

- These are a small number of early runs, not a pass rate or a controlled benchmark.
- The two toy-repository results are not issue-based repository runs.
- The Flask patch did not satisfy the hidden exception-type assertion.
- The pytest run produced no patch because `edit_file` could not apply its edits; the fix is in progress.
- Parrv will send newer numbers; update this table when they arrive.

Failure categories emitted by the scorer are `no_patch`, `empty_patch`,
`patch_does_not_apply`, `f2p_fail`, `p2p_regression`, `timeout`, and
`harness_error`. The resolved rate is resolved instances divided by scored
instances. Mean tokens per run includes all scored runs; tokens per resolved
instance is also reported separately by the analyzer.

## Trace format

Each JSONL row is one `AgentEvent`: timestamp, event type, optional phase, and a
data object. Event types include phase transitions, messages, tool calls and
results, usage, errors, and the final `done` event. The trace is append-only and
can be inspected or replayed without a provider connection.

```sh
python3 -c 'import json; from pathlib import Path; rows=[json.loads(x) for x in Path("docs/sample_trace.jsonl").read_text().splitlines()]; print("events:", len(rows)); print("phases:", sorted({r["data"].get("name") for r in rows if r["type"]=="phase"}))'
```

## Offline tests

```sh
make test
```

Tests use local Git fixtures and mocked model responses; they need no provider
key. Language support is implemented for Python, JavaScript/TypeScript, Go,
Rust, and Java, but has been verified on Python only. Docker is opt-in and
experimental.
