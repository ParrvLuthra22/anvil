# ANVIL Architecture

ANVIL is a text-model coding harness. `python -m anvil` enters the Textual UI;
`--headless` runs the same pipeline without the UI. `run_harness()` owns a run,
while `RepoPipeline`, `PhaseRunner`, the tool registry, and the sandbox keep
repository setup, model turns, and filesystem operations separate.

```mermaid
flowchart TB
    subgraph UI["Interface"]
        TUI["Textual TUI<br/>make run"]
        HL["Headless CLI<br/>--headless --issue"]
        RP["Replay<br/>anvil replay trace.jsonl"]
    end
    subgraph CORE["Agent core"]
        ORCH["Orchestrator<br/>phase state machine"]
        REC["Recovery<br/>rollback, forced close,<br/>loop detector, budgets"]
        CTX["Context manager<br/>truncate, prune, pin, summarise"]
        PR["Per-phase prompts<br/>and tool allowlists"]
    end
    subgraph LLM["LLM layer"]
        CL["OpenAI-compatible client<br/>retry, 402 backoff, think-strip"]
        PAR["Tool-call parser<br/>native, fenced JSON, Qwen XML,<br/>tool(key=value), bracket forms"]
        EP["Endpoint resolver<br/>AI_BASE_URL, AI_MODEL,<br/>key only to resolved host"]
    end
    subgraph TOOLS["Tools"]
        T1["list_dir, grep, read_file"]
        T2["edit_file<br/>exact + whitespace-tolerant"]
        T3["run_cmd (safety layer), run_tests, git_diff"]
        T4["outline, find_symbol<br/>(flag: nav_tools)"]
    end
    subgraph ENV["Repo and sandbox"]
        ING["Ingest<br/>issue fetch, clone, base-ref resolve"]
        PRO["Profile and deps<br/>interpreter probe, date-pinned venv"]
        SB["WorktreeSandbox<br/>checkpoint, rollback, diff<br/>(Docker opt-in)"]
    end
    subgraph OBS["Observability"]
        BUS["EventBus"]
        TR["TraceRecorder<br/>trace.jsonl"]
    end
    OUT["output/<br/>patch.diff, report.md, trace.jsonl"]
    MODEL[("Foundation model<br/>DeepSeek or Qwen")]
    TUI --> ORCH
    HL --> ORCH
    RP --> TR
    ORCH --> PR
    ORCH --> CTX
    ORCH --> REC
    ORCH --> CL
    CL --> EP
    CL --> PAR
    CL <--> MODEL
    PAR --> TOOLS
    TOOLS --> SB
    ING --> PRO
    PRO --> SB
    ORCH --> ING
    ORCH -.events.-> BUS
    BUS --> TUI
    BUS --> TR
    ORCH --> OUT
    TR --> OUT
```

You start a run from the TUI, the headless command or a replay, and the orchestrator drives everything from there. It gives each phase its own prompt and tool list, keeps the conversation small, and steps in when the model loops, fails an edit or runs out of budget. The LLM layer talks to whatever OpenAI-compatible endpoint `AI_BASE_URL` and `AI_MODEL` name (read in `llm/config.py`), sends the key only to that host, and turns the model's reply into a tool call even when the model writes it as plain text. Tools change files only inside a git-worktree sandbox, and every event goes out on one bus to the TUI and to `trace.jsonl`, so a finished run can be replayed with no model at all.

## Where each box of the first diagram lives

Every box was checked against the source tree. Two labels are roles rather than
modules: "Endpoint resolver" has no module of its own, and "Recovery" spans three.

| Box | Where in `src/anvil/` |
|---|---|
| Textual TUI | `tui/app.py`; started by `__main__.py` (`make run`) |
| Headless CLI, Replay | `__main__.py` (`--headless`, `replay`); replay loads the trace with `trace/recorder.py` |
| Orchestrator | `agent/orchestrator.py` (phase order, the facts it checks itself), with `agent/pipeline.py` for INGEST and PROFILE |
| Recovery | `agent/recovery.py` (loop detector, repeated-call refusal, checkpoints and rollback), `agent/closure.py` (the harness's own phase close), `agent/budget.py` (step, token and wall-clock budgets) |
| Context manager | `context/manager.py`, `context/truncate.py`, `context/tokens.py`; summaries by `agent/summarizer.py` |
| Per-phase prompts and tool allowlists | `agent/prompts.py` |
| OpenAI-compatible client | `llm/client.py` (retry, the 402 rules), `llm/reasoning.py` (think-strip), `llm/profiles.py` |
| Tool-call parser | `llm/toolcalls.py` |
| Endpoint resolver | `llm/config.py` applies `AI_BASE_URL` and `AI_MODEL` over `config.yaml`; `llm/client.py` reads `AI_API_KEY` and sends it only in the request to that base URL |
| list_dir, grep, read_file, edit_file, run_cmd, run_tests, git_diff | `tools/*.py`; the safety layer is in `tools/run_cmd.py` |
| outline, find_symbol | `tools/outline.py`, `tools/find.py`; offered only with `features.nav_tools`. `find_references` and `related_tests` are registered but never offered to the model |
| Ingest | `repo/ingest.py` (`fetch_issue`, `clone_repo`, `resolve_base_ref`) |
| Profile and deps | `repo/profile.py`, `repo/deps.py` (interpreter choice by release date, date-pinned installs) |
| WorktreeSandbox | `sandbox/worktree.py`; `sandbox/docker.py` is the opt-in alternative |
| EventBus, TraceRecorder | `events.py`, `trace/recorder.py` |
| output/ | written by `agent/outputs.py` |

## Run lifecycle and phases

`Orchestrator.run()` catches budget, model, pipeline, and internal failures and
always enters finalization. INGEST and PROFILE are deterministic pipeline work;
the remaining phases are LLM tool loops. Each transition is emitted as a trace
event.

```mermaid
stateDiagram-v2
    [*] --> INGEST
    INGEST --> PROFILE: repo cloned at pre-fix commit
    PROFILE --> UNDERSTAND: venv built, tests detected
    UNDERSTAND --> LOCALIZE
    LOCALIZE --> REPRODUCE
    REPRODUCE --> PATCH: failing repro (model-closed or adopted)
    PATCH --> VERIFY
    VERIFY --> REVIEW: tests pass
    VERIFY --> PATCH: fail, rollback to checkpoint (max 3 attempts total)
    REVIEW --> FINALIZE: approved
    REVIEW --> PATCH: changes requested
    FINALIZE --> [*]: patch.diff, report.md, trace.jsonl
    LOCALIZE --> FINALIZE: give_up
    PATCH --> FINALIZE: budget exhausted
```

A run moves through nine phases, and the first two, INGEST and PROFILE, are plain code with no model. A failed check sends the run back to PATCH, at most three attempts in total, and with the default settings a retry keeps the edits already made instead of rolling back. REVIEW either approves the patch or asks for one rework round, and a run can also end early when LOCALIZE gives up or a budget runs out. FINALIZE always runs and writes `patch.diff`, `report.md` and `trace.jsonl`.

| Phase | Work |
|---|---|
| INGEST | Resolve supplied issue text or fetch a GitHub issue; clone at the selected base revision. |
| PROFILE | Detect languages, framework, dependency and test commands; prepare the sandbox and tools. |
| UNDERSTAND | Form a root cause hypothesis and identify relevant symbols/files. |
| LOCALIZE | Search and read likely code locations. |
| REPRODUCE | Write a minimal failing repro under `.anvil/` and run it before source edits. |
| PATCH | Edit implementation code and run the repro; retry within configured budgets. |
| VERIFY | Run relevant tests and check the repro. Failed verification can return to PATCH. |
| REVIEW | Inspect the diff; one requested rework can return to PATCH. |
| FINALIZE | Sanity-check and write the patch/report, emit `done`, then close resources. |

The phase runner filters tools by phase, validates call arguments, checks loop
patterns, and enforces per-phase and total call/step limits. Global limits also
include tokens and elapsed wall time. Hitting a budget ends model work and
finalizes with the patch available so far.

## Write policy and patch delivery

The model can write only through registered tools operating on the sandbox.
Sandbox paths are resolved under the repository root; absolute paths and paths
that escape the root are rejected. The `write_repro` tool is the exception in
purpose, not location: it creates repro material only beneath `.anvil/`.
`edit_file` performs an exact string replacement and returns a closest-match
hint when the old text is not found. Shell commands run with timeouts, closed
stdin, output caps, and the prepared dependency environment.

Before delivery, `features.patch_sanity` checks that the patch is non-empty and
applies to a scratch checkout at the original base. Test-file changes are
removed from the delivered diff unless the issue is specifically about tests.
An empty or invalid patch gets one forced-fix retry, and so does a patch that
adds a bare `assert` to non-test code: that retry asks for a specific exception,
tells the model to update its repro to expect it, and is judged by the
repository's tests if the old repro fails on the new exception. Harness scratch files,
virtual environments, bytecode, and package metadata are excluded from the
delivered diff.

## Base revision resolution

For headless input, `--ref` is passed as `git_ref`. Revision selection is:

1. An explicit `--ref` / `git_ref`.
2. `anvil.repo.ingest.resolve_base_ref(issue_ref)` for a fetched issue. The
   resolver can report a commit/ref, reason, and whether the issue appears
   already fixed.
3. The repository default branch when no revision can be resolved.

If checkout of a selected ref fails, the pipeline reports a warning and falls
back to the default branch. Supplied `--issue-text` has no GitHub issue number,
so it skips issue-specific base-ref lookup. `--repo` can be used with
`--issue-text` when fetching the issue itself is unavailable.

## Tool-call adapter

The client supports three `tool_mode` settings. `native` sends OpenAI-compatible
tool schemas. `text` adds tool instructions to the prompt and parses returned
text. `auto` starts with native calls and falls back when the provider does not
support the format or swallows the response, and then stays in text mode for the
rest of the run. Text parsing accepts fenced JSON, tool tags, XML-style function
parameters, OpenAI-shaped JSON, literal Python-call syntax, and calls wrapped in
brackets or parentheses (`[phase_done(summary)="..."]`); arguments are parsed as
data, never evaluated. In every
mode the canonical internal call is normalized before `PhaseRunner` checks it
against the phase allowlist and registry.

## Recovery

| Failure | Response |
|---|---|
| Invalid/unknown tool call | Do not execute it; return the schema or closest tool name. |
| Failed exact edit | Return closest matching lines and ask the model to reread before retrying. |
| Test failure or command timeout | Return focused failure output and advice for a narrower or corrected attempt. |
| Repeated or alternating calls | Reject the repeating call, emit loop guidance, and end the phase after the strike limit. |
| Failed patch/verification attempts | Keep a sandbox checkpoint; rollback after the configured repeated failures and try a different hypothesis. |
| LLM request failure | Client retries transient HTTP/network errors with backoff; an exhausted/unusable client ends the run gracefully. |
| Token, step, or wall-clock budget | Stop model work and finalize with reduced confidence. |
| Setup, dependency, or output error | Emit an error/warning event, preserve what was produced, and attempt finalization. |

An HTTP 402 response that asks the provider to retry after in-flight requests
settle is retried up to 5 times, with jittered waits from 2 to 30 seconds. A
different 402 response is classified as out of credit and fails immediately.
Ordinary transient 429, 5xx, and network errors use the configured retry count
and capped exponential backoff.

The worktree sandbox uses Git worktrees where possible and a plain copy as a
fallback. Checkpoints use Git snapshots and rollback restores the recorded tree
state. Docker implements the same sandbox protocol but is opt-in and
experimental; it is not the default and container networking is unavailable.

## Context manager

`ContextManager` estimates message tokens from character counts, caps tool
outputs, prunes stale tool observations after `context_keep_steps`, and folds
older conversation units when the configured threshold is exceeded. The
summarizer call is optional recovery: if summarization fails, old history is
pruned instead. The issue brief, system prompt, repo map, current phase kickoff,
and latest diff are pinned or retained according to the manager's compaction
rules. With `features.token_budgets` enabled, configured `token_saving` values
can tighten the output cap, retained steps, read-file output, repo map size, and
calls allowed per phase.

## Events, outputs, and replay

The in-process event bus gives each subscriber its own queue. The CLI subscribes
the trace recorder before starting a run; the recorder appends one JSON event
per line and flushes it. The TUI consumes the same events. Replay loads the
recorded events and feeds them to the TUI without constructing an LLM client or
making API calls. Every run attempts to write `patch.diff`, `report.md`, and
`trace.jsonl` under the configured output directory.

`report.md` states a confidence that follows the evidence: 0.90 needs repository
tests that passed and an approved review, a review that never finished costs
nothing but the approval bonus (0.75), and a patch verified only on the model's
own repro is capped at 0.50 and says no repository tests were run.

## Design decisions measured

No implementation decision is labelled measured yet. The only real runs so far
are the early Qwen3-Coder-30B results in `docs/EVALUATION.md`: two toy
repositories and two SWE-bench-style issues, one unresolved and one with an empty
patch. They are not a pass rate and cannot support a design or token-efficiency
comparison. Add a measured decision here only after comparable before/after runs
complete successfully.

## Support status and future work

Multi-language profiling and tool execution paths are implemented for Python,
JavaScript/TypeScript, Go, Rust, and Java; the current offline and live
verification has been performed on Python only. Docker is opt-in and
experimental.

Future work includes broader language-specific integration testing, stronger
Docker lifecycle/security validation, and publishing repeatable benchmark
measurements. These are not claims of completed implementation.
