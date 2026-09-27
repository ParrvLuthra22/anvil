"""Entry point that drives one full run from issue URL to patch and report.

The cycle is INGEST, PROFILE, UNDERSTAND, LOCALIZE, REPRODUCE, then PATCH and VERIFY
in a loop (with retries and rollbacks), REVIEW (with at most one rework round) and
FINALIZE. INGEST and PROFILE are plain code behind ``Pipeline``; every other phase is
an LLM tool loop (``PhaseRunner``) whose facts the harness checks for itself: it
runs the repro, reads the diff and counts failures rather than trusting the model.

Failures are met by ``anvil.agent.recovery`` (loops, failed edits, invalid calls, silent replies,
timeouts, rollbacks); a budget or LLM failure ends the run with a lowered confidence, never a crash.

Every model call sees a history kept within ``max_context_tokens`` by the ``ContextManager``:
a finished phase is replaced in it by its closing summary, old tool output shrinks to one line,
and the issue, repo map and latest diff stay pinned.

FINALIZE always happens: whatever goes wrong, ``patch.diff`` and ``report.md`` are
written and a ``done`` event is emitted.
"""

from __future__ import annotations

import os
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from anvil.agent.budget import Budget, BudgetExceeded
from anvil.agent.emitter import Emitter
from anvil.agent.loop import Gate, PhaseOutcome, PhaseRunner, PhaseStatus, RunAborted, ToolRecord
from anvil.agent.outputs import SCRATCH_DIR, changed_files, filter_diff, render_report, write_outputs
from anvil.agent.pipeline import Checkout, Ingested, Pipeline, RepoPipeline, Workspace
from anvil.agent.prompts import (
    environment_note,
    finalize_kickoff,
    issue_brief,
    patch_kickoff,
    phase_kickoff,
    phase_spec,
    phase_summary,
    verify_kickoff,
)
from anvil.agent.recovery import Checkpointer, ErrorClass
from anvil.agent.repro import WriteReproTool, reports_the_issue
from anvil.agent.sanity import PatchCheck, inspect_patch, issue_is_about_tests
from anvil.agent.settings import AgentSettings
from anvil.agent.state import REVIEW_APPROVED, REVIEW_CHANGES_REQUESTED, CheckRun, RunState
from anvil.agent.summarizer import HistorySummarizer
from anvil.agent.testcmd import is_repository_test_run
from anvil.agent.text import clip_head, clip_middle
from anvil.context import ContextManager
from anvil.events import EventBus, Phase
from anvil.llm.client import LLMClient, make_client
from anvil.llm.errors import LLMError
from anvil.sandbox.base import ExecResult
from anvil.tools.base import ToolResult

MAX_ADOPTION_RERUNS = 2  # how many failing commands the harness will run again when it looks for a repro to adopt

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"

_ENV_OVERRIDES = {"AI_BASE_URL": "base_url", "AI_MODEL": "model"}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (or ``path``) and apply the AI_BASE_URL / AI_MODEL env overrides.

    The API key is deliberately not part of the returned dict; the LLM client
    reads it straight from ``AI_API_KEY``.
    """
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}
    for env_var, key in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            config[key] = value
    return config


def run_harness(
    issue_url: str,
    config: dict,
    bus: EventBus,
    *,
    llm: LLMClient | None = None,
    pipeline: Pipeline | None = None,
    repo_url: str | None = None,
    issue_text: str | None = None,
    git_ref: str | None = None,
) -> None:
    """Run the full phase cycle for ``issue_url``, emitting events on ``bus``.

    Blocks until the run ends (call it from a worker thread under the TUI).
    ``llm`` and ``pipeline`` default to the real OpenAI-compatible client and the
    real ``RepoPipeline``; tests pass fakes. ``issue_text`` (with ``repo_url``, or an ``issue_url``
    that names the repository) is the fallback for when GitHub cannot be asked for the issue: it is
    used as the issue body and nothing is fetched. ``git_ref`` is the branch, tag or commit to check out;
    without it the revision the issue was reported against is looked up, and failing that the default
    branch is used (a ``message`` event says which, and why). Whatever happens, ``output/patch.diff`` and
    ``output/report.md`` are written and a ``done`` event is emitted. Nothing is raised
    to the caller, except that Ctrl-C still propagates once those outputs exist.
    """
    Orchestrator(
        issue_url, config, bus, llm=llm, pipeline=pipeline, repo_url=repo_url, issue_text=issue_text, git_ref=git_ref
    ).run()


# Phases that end with a summary (the model's own, or the harness's after a call cap); only those can be replaced by it.
_COMPRESSIBLE = (PhaseStatus.DONE, PhaseStatus.GAVE_UP, PhaseStatus.CLOSED)


@dataclass(frozen=True)
class _Verdict:
    passed: bool
    feedback: str = ""
    repro_failed: bool = False  # the harness's own repro re-run failed, so VERIFY never reached the model


class Orchestrator:
    """One run of the harness. Single use: construct it, call ``run`` once."""

    def __init__(
        self,
        issue_url: str,
        config: dict,
        bus: EventBus,
        *,
        llm: LLMClient | None = None,
        pipeline: Pipeline | None = None,
        repo_url: str | None = None,
        issue_text: str | None = None,
        git_ref: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._repo_url = repo_url
        self._issue_text = issue_text
        self._git_ref = git_ref
        self._settings_error = ""
        try:
            self._settings = AgentSettings.from_mapping(config)
        except ValueError as exc:
            self._settings, self._settings_error = AgentSettings.fallback(config), str(exc)
        self._emitter = Emitter(bus)
        self._budget = Budget(self._settings, clock)
        self._ctx = ContextManager(
            max_context_tokens=self._settings.max_context_tokens,
            tool_output_char_cap=self._settings.tool_output_char_cap,
            keep_steps=self._settings.context_keep_steps,
            summarize_threshold=self._settings.context_summarize_threshold,
            summarizer=HistorySummarizer(lambda: self._llm, self._budget, self._emitter, self._settings),
            read_file_max_lines=self._settings.read_file_max_lines if self._settings.token_budgets else None,
            read_file_head_lines=self._settings.read_file_head_lines,
        )
        self._state = RunState(issue_url)
        self._llm = llm
        self._owns_llm = False
        self._pipeline = pipeline or RepoPipeline(
            config, Path(self._settings.output_dir), notify=self._pipeline_note
        )
        self._workspace: Workspace | None = None
        self._runner: PhaseRunner | None = None
        self._repro_tool = WriteReproTool()
        self._checkpointer: Checkpointer | None = None
        self._reproduce_ref: str | None = None  # the clean tree REPRODUCE started from, to undo edits made in it
        self._sanity_done = False
        self._sanity_retried = False  # the one forced-fix retry of features.patch_sanity has been spent
        self._sanity_first_problem = ""
        self._sanity_first_was_warning = False  # the first patch was flagged (a warning), not refused (a problem)
        self._sanity_patch: str | None = None  # the patch as delivered, after the sanity check took test files out

    def run(self) -> None:
        """Execute the run; see ``run_harness`` for the guarantees."""
        try:
            self._execute()
        except BudgetExceeded as exc:
            self._emitter.error(ErrorClass.BUDGET.value, f"{exc}; finalising with what exists and lowering the confidence")
            self._stop(
                "budget",
                f"Stopped early: {exc}. The results reflect the work completed until then, and confidence is lowered.",
            )
        except RunAborted as exc:
            self._stop("aborted", f"Stopped early: {exc}")
        except Exception as exc:  # noqa: BLE001 - the run must always reach FINALIZE
            phase = self._emitter.phase
            kind = "llm" if isinstance(exc, LLMError) else phase.value if phase else "internal"
            message = f"{type(exc).__name__}: {exc}"
            self._emitter.error(kind, message)
            self._stop("error", f"Stopped early by an error ({kind}): {message}")
        except BaseException:
            self._state.halted = "interrupted"
            raise
        finally:
            self._finalize()

    def _pipeline_note(self, level: str, message: str) -> None:
        """Progress and problems from the pipeline's setup work: warnings are ``error`` events of kind ``deps``."""
        if level == "warning":
            self._emitter.error("deps", message)
        else:
            self._emitter.message("system", message)

    def _stop(self, reason: str, note: str) -> None:
        self._state.halted = reason
        self._state.limit(note)

    def _execute(self) -> None:
        if self._settings_error:
            self._emitter.error("config", f"invalid agent settings, using defaults: {self._settings_error}")
            self._state.limit(f"Invalid settings in config.yaml ({self._settings_error}); defaults were used.")
        if self._llm is None:
            self._llm = make_client(self._config)
            self._owns_llm = True
        ingested = self._ingest()
        self._profile(ingested)
        self._understand()
        self._localize()
        self._reproduce()
        self._solve()
        self._sanity(allow_retry=True)

    # ---- INGEST / PROFILE ---------------------------------------------------------------------

    def _ingest(self) -> Ingested:
        self._enter(Phase.INGEST)
        ingested = self._pipeline.ingest(
            self._state.issue_url, repo_url=self._repo_url, issue_text=self._issue_text, git_ref=self._git_ref
        )
        issue = self._state.issue = ingested.issue
        repo = f"{issue.owner}/{issue.repo}"
        if (self._issue_text or "").strip():
            self._emitter.message("system", f"Using the issue text you supplied and cloned {repo}.")
        else:
            self._emitter.message("system", f"Fetched issue #{issue.number} ({issue.title}) and cloned {repo}.")
        self._announce_checkout(ingested.checkout)
        return ingested

    def _announce_checkout(self, checkout: Checkout | None) -> None:
        """Say which revision was checked out; warn in the report when it probably already has the fix."""
        if checkout is None:
            return
        where = checkout.ref or "the default branch"
        self._emitter.message(
            "system",
            f"Checked out {where}: {checkout.reason}.",
            ref=checkout.ref,
            reason=checkout.reason,
            already_fixed=checkout.already_fixed,
        )
        if checkout.already_fixed and checkout.ref is None:
            self._state.warn(
                "The fix for this issue is already merged upstream, and no earlier revision of the repository could "
                f"be found ({checkout.reason}). The code checked out is the default branch, which probably no longer "
                "contains the bug: it may not reproduce, and a patch may be unnecessary or meaningless."
            )
    def _profile(self, ingested: Ingested) -> None:
        self._enter(Phase.PROFILE)
        workspace = self._pipeline.profile(ingested)
        self._workspace = workspace
        self._checkpointer = Checkpointer(workspace.sandbox, self._emitter, lambda: self._repro_tool.written)
        self._state.profile = workspace.profile
        try:
            workspace.tools.register(self._repro_tool)
        except ValueError:
            pass  # a tool with this name is already registered
        self._runner = PhaseRunner(
            llm=self._llm,
            ctx=self._ctx,
            registry=workspace.tools,
            sandbox=workspace.sandbox,
            emitter=self._emitter,
            budget=self._budget,
            settings=self._settings,
            environment=environment_note(workspace.profile, workspace.deps),
        )
        profile = workspace.profile
        self._emitter.message(
            "system",
            f"Profile: {profile.primary_language or 'unknown language'}, tests: {profile.test_cmd or 'unknown'}.",
        )
        brief = issue_brief(workspace.issue, profile)
        self._ctx.add_message("user", brief, pinned=True)
        self._emitter.message("user", brief)
        if workspace.repo_map.strip():
            self._ctx.set_repo_map(workspace.repo_map)
            self._emitter.message("user", f"Repository overview\n{workspace.repo_map.strip()}")

    # ---- UNDERSTAND / LOCALIZE / REPRODUCE ----------------------------------------------------

    def _understand(self) -> None:
        outcome = self._run_phase(Phase.UNDERSTAND, phase_kickoff(Phase.UNDERSTAND), pin=True)
        self._state.understanding = outcome.summary if outcome.done or outcome.closed else ""

    def _localize(self) -> None:
        outcome = self._run_phase(Phase.LOCALIZE, phase_kickoff(Phase.LOCALIZE), pin=True)
        if outcome.done or outcome.closed:
            self._state.localization = outcome.summary
        else:
            self._state.limit(f"The fault was not localized ({outcome.status.value}: {outcome.summary}).")

    def _reproduce(self) -> None:
        self._enter(Phase.REPRODUCE)  # so the checkpoint's events belong to this phase, not LOCALIZE
        self._reproduce_ref = self._checkpoint("reproduce-start")
        outcome = self._run_phase(Phase.REPRODUCE, phase_kickoff(Phase.REPRODUCE), gate=self._repro_gate, pin=True)
        self._undo_source_edits()
        if not self._state.repro_confirmed and outcome.status is not PhaseStatus.GAVE_UP:
            self._adopt_failing_command(outcome)
        if not self._state.repro_confirmed:
            self._state.limit(
                f"The bug was not reproduced ({outcome.status.value}: {clip_head(outcome.summary, 200)}); confidence is capped at low."
            )

    def _adopt_failing_command(self, outcome: PhaseOutcome) -> None:
        """REPRODUCE ended without a confirmed repro: adopt the model's own failing command as the repro if it qualifies.

        A model often shows the bug and then never says it is done (real runs: 25 calls, or the forced close ignored). The
        command it ran is the evidence, so the harness does not throw it away. The most recent ``run_cmd`` qualifies when it
        ran a script under ``.anvil/``, failed (not a timeout, not "command not found"), and its output reports the bug the
        issue describes (``reports_the_issue``: not a typo or a missing import, sharing words with the issue). The harness
        then runs it itself on the clean tree, as it does for ``phase_done``, and adopts it only if it fails again the same
        way. A model that gave up is taken at its word and nothing is adopted.
        """
        issue = self._state.issue
        issue_text = f"{issue.title}\n{issue.body}" if issue else ""
        tried = 0
        for record in reversed(outcome.records):
            if record.tool != "run_cmd" or record.ok or tried == MAX_ADOPTION_RERUNS:
                continue
            cmd = str(record.args.get("cmd") or "").strip()
            if not cmd or SCRATCH_DIR not in cmd or record.meta.get("timed_out") or record.meta.get("exit_code") in (126, 127):
                continue
            if not reports_the_issue(record.output, issue_text):
                continue
            tried += 1
            result, text = self._run_repro(cmd)
            if result.timed_out or result.exit_code in (0, 126, 127) or not reports_the_issue(text, issue_text):
                continue
            self._state.repro_cmd, self._state.repro_output, self._state.repro_confirmed = cmd, self._trim(text), True
            note = (
                f"REPRODUCE ended without the model's phase_done ({outcome.status.value}); the harness adopted its failing "
                f"command `{cmd}` as the repro: it fails with output that matches the issue, and fails again when the harness runs it."
            )
            self._state.limit(note)
            self._emitter.message("system", note)
            return

    def _undo_source_edits(self) -> list[str]:
        """Revert what the model changed outside ``.anvil/`` since REPRODUCE began; returns the files it had changed.

        REPRODUCE proves the bug and must not fix it: a fixed tree makes the harness's own repro run exit 0, and a
        real model that has fixed the bug then spends the phase trying to reproduce it against its own fix. The phase
        offers no edit tool, but a shell command can still edit, so the tree is checked and reset here.
        """
        try:
            changed = [change.path for change in changed_files(filter_diff(self._ws.sandbox.diff()))]
        except Exception:  # noqa: BLE001 - without a readable diff there is nothing to undo
            return []
        if not changed:
            return []
        if not self._rollback(self._reproduce_ref):
            self._state.limit(
                f"The model changed source files during REPRODUCE ({', '.join(changed)}) and they could not be reverted."
            )
            return []
        self._state.limit(f"The model changed source files during REPRODUCE ({', '.join(changed)}); the harness reverted them.")
        return changed

    def _repro_gate(self, args: dict) -> str | None:
        """Accept REPRODUCE's ``phase_done`` only if the repro really fails when the harness runs it."""
        reverted = self._undo_source_edits()
        if reverted:
            return (
                f"You changed source files in this phase ({', '.join(reverted)}) and the harness has reverted them: "
                "this phase only proves the bug exists, it must not fix it. Your repro script is still in place. Run "
                "it, check that it now fails, and call phase_done again with a summary of the bug you reproduced, "
                "not of a fix."
            )
        cmd = str(args.get("repro_cmd") or "").strip()
        if not cmd:
            return "phase_done needs repro_cmd: the exact shell command that runs your repro script."
        if not self._repro_tool.written:
            return f"No repro script exists yet. Create one with write_repro (under {SCRATCH_DIR}/) and run it first."
        if SCRATCH_DIR not in cmd:
            return f"repro_cmd must run the script you wrote under {SCRATCH_DIR}/, not an existing test command."
        result, text = self._run_repro(cmd)
        if result.timed_out:
            return (
                f"The repro timed out after {self._settings.command_timeout_seconds}s. Make it finish quickly and "
                f"exit non-zero when the bug is present.\n{text}"
            )
        if result.exit_code in (126, 127):
            return f"The shell could not run repro_cmd (exit {result.exit_code}). Fix the command.\n{text}"
        if result.exit_code == 0:
            return (
                "The repro exited 0, so it does NOT reproduce the bug. Change it so that it fails (non-zero exit) "
                f"because of the behaviour in the issue, or call give_up(reason) if that is impossible.\n{text}"
            )
        self._state.repro_cmd, self._state.repro_output, self._state.repro_confirmed = cmd, self._trim(text), True
        return None

    # ---- PATCH / VERIFY / REVIEW --------------------------------------------------------------

    def _solve(self) -> None:
        """PATCH and VERIFY until verified, then REVIEW, allowing the reviewer one rework round."""
        self._patch_until_verified("first")
        if not self._state.verified:
            return
        changes = self._review()
        if not changes:
            return
        if self._patch_attempts_left() <= 0:
            self._state.limit(
                f"The reviewer requested changes, but the limit of {self._settings.max_total_patch_attempts} PATCH attempts "
                "per run was already used, so no rework was made; the reviewed patch is delivered as it is."
            )
            return
        self._state.limit("The reviewer requested changes; one rework round was made and was not re-reviewed.")
        self._patch_until_verified("rework", changes)

    def _patch_attempts_left(self) -> int:
        """PATCH attempts the run may still make: ``max_total_patch_attempts`` counts every one (attempts, rework, sanity retry)."""
        return self._settings.max_total_patch_attempts - self._state.patch_attempts

    def _patch_until_verified(self, kind: str, feedback: str = "") -> None:
        """Alternate PATCH and VERIFY; retry on failure, roll back and rethink after too many failures.

        ``kind`` is ``"first"`` or ``"rework"`` (a reviewer-requested revision, which never
        rethinks: if it cannot be verified, the reviewed patch is restored).
        """
        state, settings = self._state, self._settings
        rework = kind == "rework"
        ref = self._checkpoint("before-rework" if rework else "patch-start")
        max_rollbacks = 0 if rework else settings.max_rollbacks
        failed = rollbacks = 0
        approaches: list[str] = []
        reviewed_checks = list(state.checks)  # what verified the patch a failed rework will put back
        state.verified = False
        while True:
            state.patch_attempts += 1
            kickoff = patch_kickoff(
                attempt=state.patch_attempts,
                repro_cmd=state.repro_cmd,
                repro_output=state.repro_output,
                feedback=feedback,
                kind=kind,
                earlier_approaches=approaches,
            )
            outcome = self._run_phase(Phase.PATCH, kickoff, gate=self._patch_gate)
            approaches.append(outcome.summary or f"(no summary; the phase ended: {outcome.status.value})")
            verdict = self._verify()
            if verdict.passed:
                state.verified = True
                return
            failed += 1
            feedback = verdict.feedback
            if self._patch_attempts_left() <= 0:
                self._abandon_patching(rework, ref, reviewed_checks, limit_reached=True, last_result=verdict.feedback)
                return
            if failed < settings.max_patch_attempts:
                kind = "retry"
            elif rollbacks < max_rollbacks and self._rollback(ref, approaches):
                rollbacks += 1
                state.rollbacks += 1
                failed, kind = 0, "rethink"
            else:
                self._abandon_patching(rework, ref, reviewed_checks)
                return

    def _abandon_patching(
        self, rework: bool, ref: str | None, reviewed_checks: list[CheckRun], *, limit_reached: bool = False, last_result: str = ""
    ) -> None:
        """Stop patching and go on to the end of the run with the best state there is.

        The best state is a verified one: for a failed rework the reviewed patch is put back. A first solve that never
        verified has none, so the last attempt's patch is delivered, unverified. ``limit_reached`` says the hard total
        of PATCH attempts is what stopped the run, and the report says so. The restored patch brings back the checks that
        verified it, so the report and the confidence describe the patch that is delivered, not the rejected rework.
        """
        state = self._state
        if not rework:
            state.limit(
                f"Verification still failed after {state.patch_attempts} patch attempts and "
                f"{state.rollbacks} rollbacks; the patch is unverified."
            )
        elif self._rollback(ref):
            state.verified = True
            state.checks = reviewed_checks
            state.limit("The reviewer's requested rework failed verification; the reviewed patch was restored.")
        else:
            state.limit("The reviewer's requested rework failed verification and could not be rolled back.")
        if limit_reached:
            delivered = "the reviewed patch it restored" if rework and state.verified else "the last attempt's patch"
            state.limit(
                f"PATCH stopped at its limit of {self._settings.max_total_patch_attempts} attempts per run "
                f"(max_total_patch_attempts), whatever rollbacks or approaches remained; {delivered} is delivered. "
                f"Last verification result: {clip_head(' '.join(last_result.split()), 300) or 'none'}"
            )

    # ---- the patch sanity check (features.patch_sanity) -----------------------------------------

    def _deliverable_patch(self) -> str:
        """The patch that goes into the outputs: the diff, after the sanity check has taken test files out of it."""
        self._sanity(allow_retry=False)
        return self._sanity_patch if self._sanity_patch is not None else self._patch_text()

    def _sanity(self, *, allow_retry: bool) -> None:
        """Check the patch before it is delivered; an empty or non-applying one, or one that adds a bare assert, gets one forced-fix retry.

        Runs once. With ``allow_retry`` false (the run is already over) it only looks and records. The outcome
        goes to ``report.md``; a patch that still fails is delivered anyway, with the confidence capped at low. A warning
        (a bare assert) that survives the retry does not lower the confidence: it is delivered with the warning at the top
        of the report.
        """
        if not self._settings.patch_sanity or self._sanity_done or self._workspace is None:
            return
        try:
            check = self._inspect_patch()
            if (check.problem or check.warnings) and allow_retry and self._patch_attempts_left() > 0:
                reason = check.problem or check.warning_text
                self._sanity_retried = True
                self._sanity_first_problem = reason
                self._sanity_first_was_warning = not check.problem
                self._emitter.message("system", f"Patch sanity: {reason} One forced-fix retry follows.")
                self._forced_fix(check)
                check = self._inspect_patch()
        except (BudgetExceeded, RunAborted):
            raise  # the run ends as it would anywhere else; _finalize records what the patch looked like then
        except Exception as exc:  # noqa: BLE001 - the check is a safeguard, never a reason to lose the run
            self._sanity_done = True
            self._state.sanity = f"not run ({type(exc).__name__}: {exc})"
            return
        self._record_sanity(check)

    def _inspect_patch(self) -> PatchCheck:
        issue = self._state.issue
        about_tests = bool(issue and issue_is_about_tests(issue.title, issue.body))
        return inspect_patch(self._ws.sandbox, self._patch_text(), allow_tests=about_tests)

    def _forced_fix(self, check: PatchCheck) -> None:
        """The one retry: a PATCH attempt told why the patch cannot be delivered (or what is wrong with it), then verification.

        When the patch only drew a warning it already works, so the retry must not make things worse: the tree is
        checkpointed first, and if the retry fails verification the earlier patch is put back.
        """
        state = self._state
        warning_only = not check.problem
        self._revert_files(check.removed_tests)  # the retry starts without the test edits it was told not to make
        ref = self._checkpoint("before-sanity-retry") if warning_only else None
        was_verified, was_checks = state.verified, list(state.checks)
        state.patch_attempts += 1
        kickoff = patch_kickoff(
            attempt=state.patch_attempts,
            repro_cmd=state.repro_cmd,
            repro_output=state.repro_output,
            feedback=check.warning_text if warning_only else check.problem,
            kind="sanity_warning" if warning_only else "sanity",
        )
        self._run_phase(Phase.PATCH, kickoff, gate=self._patch_gate)
        verdict = self._verify()
        if warning_only and verdict.repro_failed:
            verdict = self._judge_by_repository_tests(verdict)
        state.verified = verdict.passed
        if warning_only and not verdict.passed:
            if self._rollback(ref):
                state.verified, state.checks = was_verified, was_checks
                state.limit("The retry to replace the assert failed verification; the earlier patch was restored.")
            else:
                state.limit("The retry to replace the assert failed verification and the earlier patch could not be restored.")

    def _judge_by_repository_tests(self, failed: _Verdict) -> _Verdict:
        """After the assert retry the repro may fail only because it was written for the assert: the repository's tests decide.

        A repro that catches ``AssertionError`` fails on the exception the issue asked for, and restoring the assert on that
        alone delivers the wrong exception type (a real run did). So the harness runs ``run_tests`` itself: if the tests pass
        the patch with the specific exception is kept (and the failed repro stays in the report's checks, so the confidence
        is not high); if they fail, or there is no ``run_tests`` tool to ask, the verdict stands and the assert is restored.
        """
        try:
            tool = self._ws.tools.get("run_tests")
        except KeyError:
            return failed
        self._emitter.tool_call("run_tests", {})
        try:
            result = tool.run({}, self._ws.sandbox)
        except Exception as exc:  # noqa: BLE001 - a crashing tool is a failed judgement, never a crashed run
            result = ToolResult(False, f"run_tests failed to run: {type(exc).__name__}: {exc}")
        self._emitter.tool_result("run_tests", result.ok, self._trim(result.output))
        self._state.checks.append(CheckRun("run_tests (run by the harness)", result.ok))
        if not result.ok:
            return failed
        self._state.limit(
            "After replacing the assert, the repro (written for the assert) no longer passed, but the repository's tests do, "
            "so the patch with the specific exception was kept."
        )
        return _Verdict(True)

    def _revert_files(self, paths: Sequence[str]) -> None:
        """Put ``paths`` back as they are in the base commit (a file that is not tracked is removed)."""
        for path in paths:
            quoted = shlex.quote(path)
            self._exec(f"git checkout HEAD -- {quoted} 2>/dev/null || rm -f -- {quoted}")

    def _record_sanity(self, check: PatchCheck) -> None:
        state = self._state
        self._sanity_done = True
        self._sanity_patch = check.patch
        retried = self._sanity_retried
        if check.ok:
            outcome = "passed"
            if retried:
                said = "flagged" if self._sanity_first_was_warning else "refused"
                outcome += f" after one forced-fix retry (the first patch was {said}: {self._sanity_first_problem})"
            if check.removed_tests:
                outcome += f". Changes to test files were taken out of the patch: {', '.join(check.removed_tests)}"
            if check.warnings:
                outcome += f". WARNING: {check.warning_text}"
                state.warn(check.warning_text)
            if not check.apply_checked:
                outcome += f". git apply --check was skipped: {check.skipped or 'not run'}"
        else:
            state.sanity_failed = True
            outcome = f"FAILED{' after one forced-fix retry' if retried else ''}: {check.problem}"
            if state.halted:
                outcome += (
                    f" (the retry was cut short: the run stopped: {state.halted})"
                    if retried
                    else f" (no retry: the run had already stopped: {state.halted})"
                )
            elif not retried and self._patch_attempts_left() <= 0:
                outcome += f" (no retry: the limit of {self._settings.max_total_patch_attempts} PATCH attempts per run was already used)"
            state.limit(f"The patch failed its sanity check: {check.problem}")
        state.sanity = outcome
        self._emitter.message("system", f"Patch sanity: {outcome}")

    def _patch_gate(self, args: dict) -> str | None:
        if self._patch_text():
            return None
        return (
            "No source files have changed yet (git_diff is empty). "
            "Make the fix with edit_file first, or call give_up(reason)."
        )

    def _verify(self) -> _Verdict:
        """Re-run the repro ourselves (cheap fail-fast), then let the model run the relevant tests."""
        state = self._state
        self._enter(Phase.VERIFY)
        state.checks = []
        if not self._patch_text():
            return _Verdict(False, "No source files were changed: the patch phase ended without editing anything.")
        if state.repro_cmd:
            result, text = self._run_repro(state.repro_cmd)
            passed = result.exit_code == 0 and not result.timed_out
            state.checks.append(CheckRun(f"repro `{state.repro_cmd}` (re-run by the harness)", passed, repository_tests=False))
            if not passed:
                return _Verdict(False, self._trim(f"The repro still fails after your patch:\n{text}"), repro_failed=True)
        outcome = self._run_phase(Phase.VERIFY, verify_kickoff(repro_cmd=state.repro_cmd))
        tests = [record for record in outcome.records if is_repository_test_run(record.tool, record.args)]
        for record in tests:
            state.checks.append(CheckRun(_test_label(record), record.ok))
        if outcome.done:
            return _Verdict(True)
        if outcome.status is PhaseStatus.GAVE_UP:
            return _Verdict(False, self._trim(outcome.summary or "verification failed"))
        return self._verdict_from_evidence(outcome, tests)

    def _verdict_from_evidence(self, outcome: PhaseOutcome, tests: list[ToolRecord]) -> _Verdict:
        """VERIFY ended without the model's verdict (stalled, looped, step limit, or closed at its cap): decide from the evidence.

        A model that has run the tests and then wanders off has still verified the patch, and sending it back to PATCH for
        that alone re-opens a fix that works (a real run did, twice). The evidence is the harness's own repro re-run, which
        passed or VERIFY would not have started, and the repository test runs of the phase (``run_tests``, or a test command
        through ``run_cmd``): if the last one failed the patch failed verification, if it passed the patch is verified. With no test run at all the repro alone verifies it, and the
        confidence stays below high; with neither there is no evidence and verification failed.
        """
        state, status = self._state, outcome.status.value
        if tests and not tests[-1].ok:
            return _Verdict(False, self._trim(f"Verification did not finish ({status}) and the last test run failed:\n{tests[-1].output}"))
        if tests:
            state.limit(
                f"VERIFY ended without the model's verdict ({status}); the harness closed it on the evidence: the repro "
                "passes and the last test run passed."
            )
            return _Verdict(True)
        if state.repro_cmd:
            state.limit(
                f"VERIFY ended without the model's verdict ({status}) and ran no tests; the harness verified the patch on "
                "the repro alone."
            )
            return _Verdict(True)
        return _Verdict(False, self._trim(f"Verification did not finish ({status}) and there is no repro and no test run to go on."))

    def _review(self) -> str:
        """The model reviews its own diff. Returns the requested changes, or ``""`` to go ahead."""
        outcome = self._run_phase(Phase.REVIEW, phase_kickoff(Phase.REVIEW, "Verification passed. Review the patch."))
        if outcome.done:
            self._state.review = REVIEW_APPROVED
            return ""
        if outcome.status is PhaseStatus.GAVE_UP:
            self._state.review = REVIEW_CHANGES_REQUESTED
            return outcome.summary or "(the reviewer gave no details)"
        self._state.review = f"inconclusive ({outcome.status.value})"
        self._state.limit("The self-review did not finish, so the patch is unreviewed.")
        return ""

    # ---- recovery -----------------------------------------------------------------------------

    def _checkpoint(self, label: str) -> str | None:
        if self._checkpointer is None:
            raise RuntimeError("the workspace is not ready: PROFILE has not completed")
        ref = self._checkpointer.checkpoint(label)
        if ref is None:
            self._state.limit("Checkpointing failed, so rolling back was not possible.")
        return ref

    def _rollback(self, ref: str | None, tried: Sequence[str] = ()) -> bool:
        return self._checkpointer is not None and self._checkpointer.rollback(ref, tried)

    # ---- helpers ------------------------------------------------------------------------------

    @property
    def _ws(self) -> Workspace:
        if self._workspace is None:
            raise RuntimeError("the workspace is not ready: PROFILE has not completed")
        return self._workspace

    def _enter(self, phase: Phase) -> None:
        if self._emitter.phase is not phase:
            self._emitter.set_phase(phase)

    def _run_phase(self, phase: Phase, kickoff: str, *, gate: Gate | None = None, pin: bool = False) -> PhaseOutcome:
        """Run one LLM phase, then swap its transcript for its closing summary in the history.

        A phase that ended with ``phase_done`` or ``give_up`` is compressed into that summary.
        One that ran out of steps or stalled has no closing text, so its transcript stays for the
        context manager to prune as it ages. ``pin`` keeps the summary through every later
        compaction (used for the phases whose findings the rest of the run builds on).
        """
        if self._runner is None:
            raise RuntimeError("the LLM phases cannot run before PROFILE")
        self._enter(phase)
        self._ctx.set_diff(self._patch_text())
        outcome = self._runner.run(phase_spec(phase, self._settings.weak_model_prompts, self._settings.nav_tools), kickoff, gate)
        if outcome.closed:
            self._state.limit(
                f"{phase.value.upper()} used its {self._settings.call_cap(phase.value)}-call cap and the model did not "
                "close it: the harness closed it with a summary it wrote itself from the calls made."
            )
        elif outcome.forced:
            self._state.limit(
                f"{phase.value.upper()} used its {self._settings.call_cap(phase.value)}-call cap and was closed by the "
                f"harness with what it had ({outcome.status.value})."
            )
        text = outcome.summary if outcome.done or outcome.closed else f"did not complete ({outcome.status.value}): {outcome.summary}"
        message = phase_summary(phase, text)
        if outcome.status in _COMPRESSIBLE:
            self._ctx.end_phase(message, pinned=pin)
        elif pin:
            self._ctx.add_message("user", message, pinned=True)
        if pin:
            self._emitter.message("user", message)
        return outcome

    def _patch_text(self) -> str:
        """The current diff without the harness's scratch files; empty if there is none or it cannot be read."""
        if self._workspace is None:
            return ""
        try:
            return filter_diff(self._workspace.sandbox.diff())
        except Exception as exc:  # noqa: BLE001
            self._emitter.error("sandbox", f"could not read the diff: {exc}")
            return ""

    def _exec(self, cmd: str) -> ExecResult:
        try:
            return self._ws.sandbox.exec(cmd, timeout=self._settings.command_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            return ExecResult(-1, "", f"exec error: {type(exc).__name__}: {exc}", False, 0.0)

    def _run_repro(self, cmd: str) -> tuple[ExecResult, str]:
        """Run the repro command on the harness's own account; returns the result and its readable form."""
        self._emitter.tool_call("repro", {"cmd": cmd})
        result = self._exec(cmd)
        text = self._trim(_describe(result, self._settings.command_timeout_seconds))
        self._emitter.tool_result("repro", result.exit_code == 0 and not result.timed_out, text)
        return result, text

    def _trim(self, text: str) -> str:
        return clip_middle(text, self._settings.tool_output_char_cap // 2)

    # ---- FINALIZE -----------------------------------------------------------------------------

    def _finalize(self) -> None:
        """Always runs, and never raises: closing summary (if the run is healthy), outputs, ``done``, cleanup."""
        patch = ""
        try:
            self._enter(Phase.FINALIZE)
            patch = self._deliverable_patch()
            if patch and not self._state.halted:
                self._write_summary(patch)
        except Exception as exc:  # noqa: BLE001
            self._emitter.error("finalize", f"{type(exc).__name__}: {exc}")
        try:
            self._publish(patch)
        except Exception as exc:  # noqa: BLE001 - the done event must still go out
            self._emitter.error("finalize", f"could not publish the results: {type(exc).__name__}: {exc}")
            self._emitter.done(
                resolved_confidence=0.0, patch_path="", report_path="",
                steps=self._budget.steps, tokens=self._budget.tokens, seconds=round(self._budget.elapsed, 2),
            )
        self._release()

    def _write_summary(self, patch: str) -> None:
        if self._runner is None:
            return
        try:
            outcome = self._runner.run(
                phase_spec(Phase.FINALIZE, self._settings.weak_model_prompts, self._settings.nav_tools),
                finalize_kickoff(self._facts(patch)),
            )
        except (BudgetExceeded, RunAborted) as exc:
            self._state.limit(f"The closing summary could not be written: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._emitter.error("finalize", f"{type(exc).__name__}: {exc}")
        else:
            if outcome.done:
                self._state.final_summary = outcome.summary

    def _facts(self, patch: str) -> str:
        state = self._state
        title = state.issue.title if state.issue else state.issue_url
        lines = [
            f"- Issue: {title}",
            f"- Suspected location: {clip_head(state.localization, 600) or 'unknown'}",
            f"- Reproduced before the patch: {'yes' if state.repro_confirmed else 'no'}",
            f"- Verified after the patch: {'yes' if state.verified else 'no'}",
            f"- Review: {state.review}",
            f"- Files changed: {', '.join(f.path for f in changed_files(patch)) or 'none'}",
        ]
        if state.sanity:
            lines.append(f"- Patch sanity: {state.sanity}")
        if state.warnings:
            lines.append("- Warnings: " + "; ".join(state.warnings))
        if state.limitations:
            lines.append("- Known limitations: " + "; ".join(state.limitations))
        return "\n".join(lines)

    def _publish(self, patch: str) -> None:
        state, budget = self._state, self._budget
        has_patch = bool(changed_files(patch))
        report = render_report(
            state, patch, steps=budget.steps, tokens=budget.tokens, seconds=budget.elapsed,
            phase_usage=budget.by_phase if self._settings.token_budgets else None,
        )
        output_dir = Path(self._settings.output_dir)
        try:
            patch_path, report_path = write_outputs(output_dir, patch, report)
        except OSError as exc:
            self._emitter.error("io", f"could not write the outputs: {exc}")
            patch_path, report_path = output_dir / "patch.diff", output_dir / "report.md"
        self._emitter.done(
            resolved_confidence=state.confidence_score(has_patch),
            patch_path=str(patch_path),
            report_path=str(report_path),
            steps=budget.steps,
            tokens=budget.tokens,
            seconds=round(budget.elapsed, 2),
        )

    def _release(self) -> None:
        if self._workspace is not None:
            try:
                self._workspace.sandbox.close()
            except Exception as exc:  # noqa: BLE001
                self._emitter.error("sandbox", f"could not close the sandbox: {exc}")
        close = getattr(self._llm, "close", None) if self._owns_llm else None
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - nothing useful left to do at shutdown
                pass


def _test_label(record: ToolRecord) -> str:
    """How a test run reads in the report's checks: ``run_tests <targets>`` or ``run_cmd <command>``."""
    if record.tool == "run_cmd":
        return clip_head(f"run_cmd {str(record.args.get('cmd', '')).strip()}", 120)
    return f"run_tests {_test_targets(record.args)}".strip()


def _test_targets(args: dict) -> str:
    """What a run_tests call was narrowed to, for the report: ``targets`` (a list) or the older single ``target``."""
    targets = args.get("targets") or args.get("target") or ""
    joined = " ".join(str(t) for t in targets) if isinstance(targets, (list, tuple)) else str(targets)
    return joined.strip()


def _describe(result: ExecResult, timeout: int) -> str:
    head = f"[TIMED OUT after {timeout}s]" if result.timed_out else f"[exit code {result.exit_code}]"
    parts = [head]
    if result.stdout.strip():
        parts.append(f"[stdout]\n{result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"[stderr]\n{result.stderr.strip()}")
    return "\n".join(parts)
