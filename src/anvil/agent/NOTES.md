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

**Config** (`config.yaml`): `max_context_tokens` 32000 (minimum 1000), `context_keep_steps` 6,
`context_summarize_threshold` 0.75 (0.1 to 1.0), `tool_output_char_cap` 8000 (minimum 200).

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
- A checkpoint result that is empty, `None` or starts with `error` is a failed checkpoint (a limitation is recorded and
  no rollback will be attempted). If a sandbox's checkpoint empties the working tree (a `git stash` does), the edited files
  are written back through the Sandbox interface; if that is impossible the checkpoint is undone and reported as failed.
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

## 5. Not implemented (do not document as features)

- Installing dependencies for non-Python repositories beyond running the profile's `install_cmd` once; no venv-like isolation.
- Scrubbing secrets from the environment of model-run commands beyond the sandbox's fixed list of API-key variables.
- A wall-clock check in the middle of a tool call.
- Any model other than through an OpenAI-compatible chat-completions endpoint.
- Checking out a commit SHA (see the revision paragraph above), and the command-line flag for `git_ref` (the entry point's job).
