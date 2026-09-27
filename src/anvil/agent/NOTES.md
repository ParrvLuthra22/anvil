# How the agent layer behaves (notes for the docs)

Everything here describes the code as it is. Where something is *not* done, it says so. Sources: `src/anvil/context/`,
`src/anvil/agent/recovery.py`, `loop.py`, `orchestrator.py`, `pipeline.py`, `prepared_sandbox.py`.

## 1. Context manager (`src/anvil/context/`)

One `ContextManager` holds the whole conversation. Every model call gets its prompt from `build_messages(phase_goal)`,
which first compacts the history so the prompt fits a token budget.

**Token estimate.** `ceil(chars / 4)` per message, plus 4 tokens per message, plus the JSON of any tool-call arguments.
The system message (phase prompt + environment note) is counted. The tool *schemas* are not counted, and text-mode tool
descriptions are added by the LLM client afterwards, so leave headroom below the model's real window. There is no
per-model tokenizer.

**Config** (`config.yaml`): `max_context_tokens` 32000 (minimum 1000), `context_keep_steps` 3,
`context_summarize_threshold` 0.75 (0.1 to 1.0), `tool_output_char_cap` 4000 (minimum 200). Those two are the `token_saving`
values that `features.token_budgets` (on by default) applies; they only ever tighten the top-level keys, and with the feature
switched off the settings are the top-level ones, 6 and 8000 when absent (see section 6).

**Compaction, in this order, on every `build_messages` call:**

1. **Truncate on entry.** A tool result longer than `tool_output_char_cap` keeps its head and tail lines with
   `[N lines omitted]` between them (`[N chars omitted]` when the text is a few huge lines). Only `tool` messages are
   truncated; user and assistant text is never cut.
2. **Prune stale observations.** A tool result from a step older than `context_keep_steps` becomes one line:
   `[output pruned] read_file(path='a.py', start=1, end=40) -> ok: <first line>` (`failed` when the tool failed). A step is
   one model reply. The assistant's message and its tool call stay; only the payload goes. Pruning is permanent.
3. **Fold the oldest turns** once the estimate passes `threshold x budget`. An assistant message and its tool results are
   one unit, never split, so the history stays valid chat format. The newest 2 units, pinned messages and the current
   phase's kickoff are never folded. Folding aims for about 50 % of the budget and is skipped if it would free less than
   10 % of it.
   - *With a summariser* (the orchestrator always supplies one): one LLM call turns the old turns into a message that starts
     `[Summary of earlier work]`, clipped to 2400 characters. The next fold includes the previous summary, so it rolls up.
     The call counts as a step, its tokens are charged to the run budget, and the TUI sees a system message and an
     `llm_usage` event.
   - *Without one, or if it fails or returns nothing:* a digest replaces the turns: `[Earlier work was dropped to fit the
     context budget. Actions taken, oldest first:]` followed by one line per tool call (newest kept, at most 2400
     characters). After a failed summariser call it is not tried again for 5 steps. An `LLMError` from it emits an `error`
     event of kind `context`; other failures are only logged.
4. **Squeeze**, only if it is still over budget: trim the repo map (never below 800 characters), prune remaining raw
   observations regardless of age, then drop the oldest units into a digest (the newest unit is kept). If it *still*
   does not fit, the prompt is sent anyway (a warning is logged): pinned items alone can exceed a tiny budget.

**Never dropped.** The issue brief (issue text, comments, repo profile) is pinned. The repo map is pinned (it can be
trimmed, not removed). The latest diff is pinned: the orchestrator refreshes it at the start of every phase, the heading
says "as of the start of this phase", it is truncated to the output cap, and it replaces the previous one (an empty diff
removes it). The phase kickoff stays until its phase ends. The system prompt is on every call.

**Phase compression.** When a phase ends with `phase_done` or `give_up`, its whole transcript (kickoff, calls, results) is
replaced by one message `[PHASE summary]` holding the text the model wrote. UNDERSTAND, LOCALIZE and REPRODUCE summaries are
pinned; PATCH, VERIFY and REVIEW summaries are ordinary history and can be folded later. A phase that ended by hitting its
step limit, stalling or looping has no closing text, so its transcript is kept and simply ages out through steps 2 to 4.

**Not implemented:** retrieval or embeddings, a tokenizer per model, compaction of the system prompt.

## 2. Recovery (`src/anvil/agent/recovery.py`, used by `loop.py` and `orchestrator.py`)

Each intervention is announced as an `error` event; `kind` says which. The model sees the response as the tool result.

| `kind` | Trigger | Response |
|---|---|---|
| `loop` | the same tool with the same arguments 3 times in a row, or A, B, A, B | the repeated call is **not run**; the model gets "you are repeating yourself... try a different approach: <hint for this phase>" and "Strike n of 3". Each further repeat is another strike; the third ends the phase (status `looped`). Counted per phase run; `phase_done` and `give_up` are exempt |
| `edit` | `edit_file` failed | the closest lines of the file with line numbers, plus "re-read the file before editing". A multi-line `old` is matched as a block (whitespace ignored, up to 8 lines, needs 2 agreeing lines); otherwise the 3 lines most like its first line. Notes when only whitespace differs. Not repeated if the tool already listed close lines |
| `invalid_call` | unknown tool, a tool not allowed in this phase, arguments that are not valid JSON, a missing required argument, a wrong type | the call is not run; the model gets the tool list (with a "did you mean" for typos) or the tool's JSON schema. Extra arguments are tolerated. `phase_done`/`give_up` are not schema-checked (the phase gates handle them) |
| `no_tool_call` | a reply without a tool call in a phase that has tools | nudged once; a second such reply in a row ends the phase (status `stalled`). UNDERSTAND and FINALIZE have no tools: a plain reply is their answer |
| `test_failure` | `run_tests` failed (not a timeout) | the output is prefixed with "[test run failed]", up to 10 key failure lines (pytest, unittest, jest/mocha, go, cargo, maven, TAP) and a reminder not to weaken tests. The raw output follows. `run_cmd` exiting non-zero is *not* an error (a repro is meant to fail) |
| `timeout` | any tool result reporting a timeout | advice to run something narrower |
| `tool` | a tool other than `edit_file`, `run_tests` and `run_cmd` returned `ok=False`, or any tool raised | advice (and "find the path first" for not-found); a crash becomes a failed result, the run continues |
| `llm` | the LLM client gave up after its own retries | the run stops calling the model, writes the outputs and finishes. The event and the report say why, with advice by cause (credentials, 404, prompt too large, rate limit, provider down) |
| `budget` | steps, tokens or wall-clock limit reached | same graceful finish. Limits are checked before each LLM call, so one long tool call can overshoot the wall clock. A run cut short is never reported above `medium` confidence |

Other kinds the orchestrator emits: `rollback`, `sandbox` (checkpoint/rollback/diff/close problems), `deps` (dependency
install warnings), `context` (summariser failed), `config`, `io`, `finalize`, and the phase name (for example `ingest`) or
`internal` for an unexpected exception. All of them still end with `patch.diff`, `report.md` and a `done` event.

**Per-call order in `PhaseRunner`:** loop check, bad-JSON check, `phase_done`/`give_up`, tool allowed in this phase and
registered, argument check, run the tool (a crash is caught), review the result. History is always valid: every tool call
gets a reply, including the ones that were not run.

**Confidence** (`RunState.confidence`): `none` if there is no patch; `low` unless the bug was reproduced before the patch
*and* verification passed; `high` only with that, an approving review and no failed check; else `medium`. A halted run is
capped at `medium`. The done event carries 0.0, 0.3, 0.6 or 0.9.

**Patch, verify, rollback (orchestrator + `Checkpointer`):**
- A checkpoint is taken when PATCH starts (`patch-start`). After each PATCH the harness re-runs the repro itself, then
  the model runs the tests. On failure the model retries (up to `max_patch_attempts`, 3); after that the tree is rolled back
  (up to `max_rollbacks`, 2) and the model is told the approaches already tried plus the last failing output.
  When both are used up the last patch is kept, unverified, at low confidence.
- The repro script (written with `write_repro` under `.anvil/`) is saved and restored around every checkpoint and rollback.
- **REPRODUCE must not fix the bug.** A checkpoint (`reproduce-start`) is taken when the phase begins, and the phase has no
  `edit_file` (only `read_file`, `grep`, `run_cmd`, `write_repro`). A shell command can still edit, so when the model calls
  `phase_done`, and again when the phase ends any other way, everything changed outside `.anvil/` is rolled back to that
  checkpoint. At `phase_done` the call is then refused with a message saying so, so the model re-confirms on the original
  code and does not hand a summary of a fix to the later phases; the report lists the reverted files under known limitations.
  Without a usable checkpoint nothing can be reverted, and the report says that instead. This exists because a real model
  fixed the bug during REPRODUCE, after which the harness's own repro run exited 0 and the model spent the rest of the phase
  trying to reproduce a bug it had fixed.
- **A repro the model showed but never confirmed is adopted.** If REPRODUCE ends without a confirmed repro (closed at its cap,
  the step limit, a stall or a loop; not when the model gave up) the harness looks at the model's own `run_cmd` calls, newest
  first, and takes the first that (a) ran a script under `.anvil/`, (b) failed, not by timeout or "command not found", (c) has
  output that reports the issue's bug (`agent/repro.py: reports_the_issue`: not a typo, missing import or wrong path unless the
  issue names that very error; sharing at least one distinctive word with the issue, file paths and common words not counting)
  and (d) fails again, with matching output, when the harness runs it itself on the clean tree. At most two commands are run
  again. The result is `Reproduced before patching: yes`, the command becomes `repro_cmd` for PATCH and VERIFY, and both the
  report and a `system` message say it was adopted. Found in real runs: a valid failing repro by call 3 to 5, then 25 calls (or a
  forced close ignored) and "not reproduced, confidence low" for a patch that was correct.
- A checkpoint result that is empty, `None` or starts with `error` is a failed checkpoint (a limitation is recorded and
  no rollback will be attempted). If a sandbox's checkpoint empties the working tree (a `git stash` does), the edited files
  are written back through the Sandbox interface; if that is impossible the checkpoint is undone and reported as failed.
- **VERIFY is decided by evidence when the model does not decide it.** If VERIFY ends without `phase_done` or `give_up` (stalled,
  looped, step limit, or closed at its call cap) the harness looks at what it holds: its own repro re-run (which passed, or VERIFY
  would not have started) and the phase's `run_tests` calls. The last test run failed: verification failed, and its output goes to
  the retry. The last passed: verified, with a limitation note. No test run but a repro that passes: verified on the repro alone,
  flagged (`verify_without_tests`) so the confidence cannot be high. Neither: failed. A `give_up` is the model's own verdict and
  stands. A real run went back through PATCH and VERIFY for a fix that worked because its VERIFY stalled after the tests passed.
- **How many times VERIFY can send the run back to PATCH.** A failed verification is a loop-back: the first attempt plus
  `max_patch_attempts - 1` (2 by default) retries per approach. After that the tree is rolled back and the model must take a
  different approach (at most `max_rollbacks`, 2 by default), each approach again getting up to 3 attempts, so PATCH runs at most
  `max_patch_attempts * (1 + max_rollbacks)` times (9 by default). A VERIFY that merely stops without a verdict never loops
  back by itself when the evidence passes, so it cannot re-open a fix that works, and a VERIFY after a loop-back is decided the
  same way as the first.
- REVIEW reads and may run tests but cannot edit: its tools are `git_diff`, `read_file` and `run_tests` (not `run_cmd`, not
  `edit_file`). `run_tests` is there because a real model spent its REVIEW calls on it and was refused; it changes nothing in the
  repository.
- REVIEW may ask for changes with `give_up`. That triggers exactly one rework round (checkpoint `before-rework`), never a
  rethink, and it is not re-reviewed. If the rework fails verification the reviewed patch is restored.

## 3. Related behaviour the docs should get right

- `run_harness(issue_url, config, bus, *, llm=None, pipeline=None, repo_url=None, issue_text=None, git_ref=None)`. With `issue_text` the
  GitHub API is not asked; the repository comes from `repo_url` (or `issue_url`). A failed fetch (rate limit, 404,
  network) ends the run with an `ingest` error and no LLM call; it does not pass GitHub's error note to the model. Mapping the
  CLI flags (`--repo`, `--issue-text`) onto these parameters is the entry point's job.
- **Which revision is checked out** (`RepoPipeline.ingest`, in this order): the `git_ref` passed to `run_harness`; else the
  ref that `anvil.repo.ingest.resolve_base_ref(issue_ref)` finds for the issue (imported defensively: if the function is not
  there, or raises, or the run has no issue number because the issue text was supplied, it is skipped); else the repository's
  default branch. The resolver may return `None`, a ref string, or an object or mapping with `ref`, `reason` and
  `already_fixed`. If the chosen ref cannot be cloned, a warning is emitted and the default branch is cloned instead.
  `git clone --branch` takes branches and tags only, so a commit SHA always takes this fallback until `clone_repo` learns to fetch one.
  A `message` event from the ingest phase says what happened (`Checked out <ref or the default branch>: <reason>.`), with the
  fields `ref` (`None` for the default branch), `reason` and `already_fixed` beside `role` and `text`. When `already_fixed` is
  true and the checkout ended up on the default branch, the code probably already contains the fix: `report.md` then starts
  with a `## Warnings` section saying so, and the closing summary is written with the warning in its facts.
- All paths handed to the sandbox are absolute, so a relative `output_dir` (the default `output`) works.
- After profiling, `anvil.repo.deps.ensure_deps` installs the repository's dependencies (Python: a venv in `.anvil_venv`,
  put first on PATH for every command) unless `install_dependencies: false`. If it is missing or fails, a `deps` warning
  is emitted and the run continues; the model is told the state in its system prompt.
- Every command runs with `PYTHONDONTWRITEBYTECODE=1` and stdin from `/dev/null` (`PreparedSandbox`).
- `.anvil/` and `.anvil_venv/` never appear in `patch.diff` and are excluded from the clone's git.
- Each phase's system prompt ends with an "Environment" note: primary language, interpreter (`python3`), test command, and
  the dependency state.
- Default sandbox is `worktree`; Docker is opt-in and has been tested only with mocks (its containers have no network).

## 4. What the LLM client absorbs (`src/anvil/llm/`), and why

Observed with Qwen3-Coder through OpenRouter (upstream provider Novita); no other provider or model family has been run yet.

- **Swallowed replies.** The provider bills tokens and returns `content: null` and no `tool_calls`, with `tools` attached and
  also without them. In auto mode a swallowed native reply is repeated once in text mode; two of them in a run keep the client
  in text mode. In text mode an empty but billed reply is asked again up to twice, with a reminder added for that call only and
  a temperature of at least 0.5 (a greedy repeat is swallowed the same way). Cut-off replies, replies with no tokens billed
  and reasoning-only replies are not retried. The tokens of every attempt are added to the response's usage and to the
  budget, so the agent loop sees at most one reply and never a provider glitch. Explicit `tool_mode: native` stays strict.
- **Call dialects read from text:** a fenced JSON block, Hermes/Qwen `<tool_call>` tags, Qwen3-Coder `<function=...>` XML
  (also when the opening `<tool_call>` is missing, if the block is closed), bare JSON, and `tool_name(key="value")` as the last
  thing in a reply for a known tool (read with `ast.literal_eval`; nothing is executed). Only the first call of a reply runs;
  the model is told how many more there were.
- Reasoning (`<think>` blocks, `reasoning_content`) is removed before parsing and never enters the history; its tokens are counted.
- **HTTP 402 comes in two kinds, told apart by the body.** OpenRouter returned both to a free-tier key. One says to retry after
  in-flight requests settle (`in_flight_budget_exhausted`, with a `Retry-After` of about 2 minutes): a temporary budget, retried
  on a budget of its own, separate from the attempts 429 and 5xx share: 5 requests in all, waits of 2 to 4, 4 to 8, 8 to 16 and
  15 to 30 seconds (equal jitter), a `Retry-After` hint as a floor and never more than 30 s per wait; if it never settles the
  error says so in one line (retryable, with the seconds waited) and what to do. Any other 402 is "out of credit": not retried,
  one line naming the provider host, the model and `AI_API_KEY` (never the key, never credentials in `base_url`), with the
  provider's own words kept in `LLMError.detail` for the trace. The same rule applies to a 402 reported inside a 200 body.
  A provider reserves `max_tokens` against the account before answering, which is why the Qwen profile's reply cap is 4096
  (the largest reply seen was 541 tokens; 8192 was refused, 4096 accepted); `max_output_tokens` in `config.yaml` overrides it.

## 5. Not implemented (do not document as features)

- Installing dependencies for non-Python repositories beyond running the profile's `install_cmd` once; no venv-like isolation.
- Scrubbing secrets from the environment of model-run commands beyond the sandbox's fixed list of API-key variables.
- A wall-clock check in the middle of a tool call.
- Any model other than through an OpenAI-compatible chat-completions endpoint.
- Checking out a commit SHA (see the revision paragraph above), and the command-line flag for `git_ref` (the entry point's job).

## 6. Feature switches (`features:` in `config.yaml`)

Each switch is on unless set to `false` (except `nav_tools`, which is off until measured), and off restores the behaviour from before it existed. The numbers behind
`token_budgets` are under `token_saving:`.

- **`token_budgets`** (spend fewer tokens per model call)
  - A `read_file` result of more than `token_saving.read_file_max_lines` (150) lines becomes its first 60 lines, an outline
    of the definitions in the rest (`line: signature`, at most 80) and a hint to call `read_file` with a line range. Only
    numbered file text is treated so; short reads and other tools' output are untouched. Flask's `blueprints.py` went from
    about 6,400 tokens to about 900.
  - Each phase has a call cap (UNDERSTAND 1, LOCALIZE 8, REPRODUCE 10, PATCH 15, VERIFY 8, REVIEW 6; per PATCH attempt).
    A cap of N gives the model N calls of its own, and call N+1 is the harness's forced close, so a phase makes at most N+1 calls.
    The forced close: only `phase_done` and `give_up` are offered, with a phase-specific instruction to summarise the best
    findings so far, any other tool is refused, and `phase_done` still goes through the phase's gate.
  - **The harness closes the phase itself; it never depends on the model.** The forced call is the model's one chance to write
    its own summary. If it does not end the phase with it (it ignores the request and calls another tool, answers in words
    only, calls a tool that does not exist, is refused by the gate, or sends a `phase_done` with an empty summary) the phase is
    over anyway: `PhaseStatus.CLOSED`, `PhaseOutcome.closed`, with a summary composed by `agent/closure.py` from the calls made
    (files read, searches, commands with their exit status, tests, edits) and the model's last note. `outcome.done` still means
    "the model called `phase_done`", so a harness closure can never be mistaken for a model's verdict. Like any summary it
    replaces the phase's transcript in the history, so the next phase does not see the argument about the close. UNDERSTAND and
    LOCALIZE hand the summary on as their findings; the report says which phases the harness closed. (Found in real runs: REPRODUCE
    ignored the forced close 2 times out of 2 and was left "not completed".) A cap at or above `max_steps_per_phase` never comes up.
    Why these numbers (9 real Qwen3-Coder runs on the toy repo): when the model closed a phase itself it needed at most 1 call
    (UNDERSTAND), 8 (LOCALIZE), 6 (REPRODUCE), 10 (PATCH), 6 (VERIFY) and 5 (REVIEW). A REPRODUCE that got past 6 calls never closed
    (11 calls at the cap, 25 without one), so 10 is enough and more only buys wandering. REVIEW took 4 to 5 calls, so the old cap of 3
    forced the close in 4 of 4 capped runs; 6 lets a natural close happen.
  - `context_keep_steps` and `tool_output_char_cap` are only ever tightened (to 3 and 4000), and the repository map is
    `token_saving.repo_map_chars` (3000) instead of 6000 characters.
  - `report.md` gets a "Tokens by phase" table (calls, prompt and completion tokens, prompt tokens per call, and a total).
- **`weak_model_prompts`** (**off by default**, pending a measurement on real instances): with it on, every phase gets a shorter system prompt (a third fewer characters overall): one shared list of
  rules (exactly ONE tool call per turn, read a file before editing it, smallest change that fixes the issue, never edit
  tests, end with `phase_done`) and one example call per phase, which a test parses against the real tool schemas. PATCH and
  REVIEW add how errors are raised: the most specific built-in exception with a clear message, never `assert` for input
  validation, and follow how the module already reports similar errors. Same tools and the same `Phase: NAME` marker as the
  original prompts, which come back unchanged with the switch off.
- **Validation prompts (both prompt tables, so they apply with `weak_model_prompts` off).** PATCH: if the issue asks for
  validation, raise a specific exception (ValueError, TypeError, or the type the issue names) with a clear message, and never use
  `assert` to validate input. REVIEW asks "Does the exception type and message match what the issue asks for?". Found on
  `pallets__flask-4045`, where the model used `assert` and the issue asked for a `ValueError`. `patch_sanity` checks the patch too
  (below).
- **`patch_sanity`**: before FINALIZE the patch is built from the diff without `.anvil/`, `.anvil_venv/`, `*.egg-info`, bytecode
  and binary sections; changes to test files are taken out of it unless the issue is about tests (the title mentions tests,
  testing, coverage or flakiness, or the text asks to add or write tests); it must not be empty; and `git apply --check` must
  accept it in a scratch worktree at the base commit (created outside the repository and removed again; skipped, and said
  so, outside a git checkout or when git cannot make the worktree). An empty or non-applying patch gets ONE forced-fix retry:
  a PATCH attempt whose prompt carries the reason, then the usual verification (test edits are undone first when they were
  the problem). The outcome (`passed`, `passed after one forced-fix retry`, `FAILED ...`, or a skipped check) is a line in
  the Outcome section of `report.md` and a fact in the closing summary. A patch that still fails is delivered anyway,
  flagged, with confidence capped at low. A run the budget stopped is checked without a retry.
- **`nav_tools`** (**off by default**): with it on, the `outline(path)` and `find_symbol(name)` tools are offered in LOCALIZE and
  PATCH only, with a line in the prompt; `find_references` is never offered. A tool the registry does not have is skipped. Its
  effect on tokens has not been measured, so it stays off until a real run shows tokens per call dropping.
