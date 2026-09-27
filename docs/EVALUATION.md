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

Failure categories emitted by the scorer are `no_patch`, `empty_patch`,
`patch_does_not_apply`, `f2p_fail`, `p2p_regression`, `timeout`, and
`harness_error`. The resolved rate is resolved instances divided by scored
instances. Mean tokens per run includes all scored runs; tokens per resolved
instance is also reported separately by the analyzer.

## Early results (Qwen3-Coder-30B; not a pass rate)

These are the only real runs recorded so far. They used **Qwen3-Coder-30B on
OpenRouter**, with the settings that were current at the time. They are early
observations, not a pass rate: four runs are too few for one, and two of them
are on a toy repository written for the harness, not on scored issue instances.
No per-run timing or token totals are reported because the run records for these
four results are not in this checkout.

| Run | Target | Result |
|---|---|---|
| Toy run 1 | Toy repository | Correct patch |
| Toy run 2 | Toy repository | Correct patch |
| Flask | `pallets__flask-4045` | Unresolved (`f2p_fail`): the patch used `assert` where the issue asked for `ValueError` |
| pytest | `pytest-dev__pytest-11143` | Empty patch (`empty_patch`): `edit_file` could not apply the model's edits |

So: the toy repository, 2 of 2 correct patches; the two real issues, 0 of 2
resolved. Do not add these up into a resolved rate. The locally saved audit
under `bench/results-integration-real-audit.jsonl` is a separate run set and is
not one of these four results.

## Known limitations

- **There is no pass rate.** Four runs, one model, one provider. Nothing here
  supports a claim about how often ANVIL resolves an issue.
- **Neither real issue was resolved.** The Flask patch raised the wrong kind of error
  (`assert`, not `ValueError`), and the pytest run produced no patch because the edit
  tool could not apply the model's edits. Since then `edit_file` has been made
  tolerant of indentation differences, and the retry that replaces a bare `assert`
  has been changed so that a repro written for the assert cannot undo it. No rerun
  is recorded in this document, so neither change is shown to fix these two issues.
- **The toy successes prove little.** The toy repository was written to exercise
  the harness. A correct patch there says the pipeline runs end to end, not that
  it can fix a real project.
- **Only Python repositories have been run.** JavaScript, TypeScript, Go, Rust and
  Java are implemented but not verified end to end.
- **A weak model often fails to close phases.** The harness closes them and says so
  in `report.md`; results with a small model are usually medium confidence at
  best, and confidence is not a measured probability of being right.
- **Scoring needs a period-correct environment.** Old repositories need an old
  Python and date-pinned dependencies, and a scorer failure on that is a harness
  result, not evidence about the patch.

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
