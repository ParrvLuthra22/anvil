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

## Measured real runs

No real baseline from before the benchmark implementation is available, so
these measured runs make no before/after improvement claim. Values below come
from the saved runner records, traces, and scorer output.

| Label | Instance | Resolved | Category | Steps | Tokens | Wall time (s) |
|---|---|---:|---|---:|---:|---:|
| `live-audit-openrouter-nemotron-20260927` | `psf__requests-1963` | 0/1 (0%) | `harness_error` | 54 | 302,266 | 202.6 |
| `integration-deps-fix-20260927` | `pallets__flask-4045` | 0/1 (0%) | `harness_error` | 67 | 493,518 | 300.2 |
| `integration-deps-fix-20260927` | `pallets__flask-4992` | 0/1 (0%) | `empty_patch` | 1 | 0 | 13.8 |

For `integration-deps-fix-20260927`, resolved rate was 0/2 (0%), failure
categories were `harness_error`: 1 and `empty_patch`: 1, and mean tokens per
run were 246,759.

### Gold-patch scorer check

| Check | Instance | Gold patch | Scorer result | Category |
|---|---|---|---|---|
| `upstream-gold-check-20260927` | `pallets__flask-4045` | Merged upstream fix, commit `08e459e` | Fail | `f2p_fail` |

The gold implementation patch applied successfully and passed patch sanity.
The FAIL_TO_PASS command did not reach assertions: the scorer's isolated
environment selected Python 3.14, where the historically pinned pytest stack
fails during collection with `AttributeError: __spec__` from `py.path`. The
machine has Python 3.13 and 3.14, but no Python 3.11 or 3.12. This is a failed
gold-check result, not evidence that the gold implementation is wrong. Akshat's
latest dependency changes were present in the fresh `main` checkout at
`f70e8f0`. No new real-model run from Parrv was supplied for this audit.

### Live run record

- Benchmark instance: `psf__requests-1963`, real issue
  [psf/requests#1963](https://github.com/psf/requests/issues/1963).
- Agent input: repository `https://github.com/psf/requests`, base commit
  `110048f9837f8441ea536804115e80b69f400277`, and the instance's public
  `problem_statement`. `test_patch`, `FAIL_TO_PASS`, and `PASS_TO_PASS` were
  not passed to the agent.
- The issue-fetch route first returned GitHub HTTP 403. The benchmark's
  repo/ref/issue-text fallback then ran the model. The model reached PATCH but
  the OpenRouter free-model daily quota was exhausted at 50 requests; the
  harness finalized after 202.6 seconds with 54 steps and 302,266 tokens,
  confidence 0.30, and no completed verification/review.
- Scoring applied the agent patch and hidden test patch in a fresh checkout,
  then categorized the run as `harness_error`: installing this historical
  Requests revision failed under Python 3.13. PASS_TO_PASS and FAIL_TO_PASS
  collection were also attempted without that install and both stopped at
  collection on the vendored urllib3 `_implementation` import error, before
  assertions. The result is not counted as a verified fix.
- The agent patch applied, but included an unrelated `requests/utils.py`
  compatibility change and remained unverified. This is a concrete failed
  run, not a success claim.
- Full run files: `bench/runs/psf__requests-1963/patch.diff`,
  `report.md`, and `trace.jsonl`. Score record: `bench/score-live-audit.json`;
  runner record: `bench/results-live-audit.jsonl`. The replay sample is
  `docs/sample_trace.jsonl`; its event sequence is complete through FINALIZE.

### Integration dependency-fix run

- Instances: `pallets__flask-4045` and `pallets__flask-4992`, both from
  `bench/instances.json`. The agent received each repository, pinned base
  commit, and public problem statement only; hidden test patches and oracle
  lists were applied only by the scorer.
- `pallets__flask-4045` reached FINALIZE after 67 steps, 493,518 tokens, and
  300.2 seconds. The run ended at the configured wall-clock budget. Scoring
  classified it as `harness_error`: the issue's Python requirement and
  historical cutoff require a Python below 3.12, but the machine has Python
  3.13 and 3.14 only. Dependency setup therefore could not provide an
  interpreter and the scorer did not run its oracle tests.
- `pallets__flask-4992` stopped after 1 step, 0 tokens, and 13.8 seconds when
  OpenRouter returned HTTP 429 with its free daily request quota at zero. No
  patch was produced; scoring classified it as `empty_patch`.
- Runner data: `bench/results-integration-real-audit.jsonl`. Scorer output:
  `bench/score-integration-audit.json`. Run traces, reports, and patches are
  under `bench/runs/pallets__flask-4045/` and
  `bench/runs/pallets__flask-4992/`.

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
