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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from anvil.agent.budget import Budget, BudgetExceeded
from anvil.agent.emitter import Emitter
from anvil.agent.loop import Gate, PhaseOutcome, PhaseRunner, PhaseStatus, RunAborted
from anvil.agent.outputs import SCRATCH_DIR, changed_files, filter_diff, render_report, write_outputs
from anvil.agent.pipeline import Ingested, Pipeline, RepoPipeline, Workspace
from anvil.agent.prompts import (
    PHASE_SPECS,
    environment_note,
    finalize_kickoff,
    issue_brief,
    patch_kickoff,
    phase_kickoff,
    phase_summary,
    verify_kickoff,
)
from anvil.agent.recovery import Checkpointer, ErrorClass
from anvil.agent.repro import WriteReproTool
from anvil.agent.settings import AgentSettings
from anvil.agent.state import REVIEW_APPROVED, REVIEW_CHANGES_REQUESTED, CheckRun, RunState
from anvil.agent.summarizer import HistorySummarizer
from anvil.agent.text import clip_head, clip_middle
from anvil.context import ContextManager
from anvil.events import EventBus, Phase
from anvil.llm.client import LLMClient, make_client
from anvil.llm.errors import LLMError
from anvil.sandbox.base import ExecResult

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
) -> None:
    """Run the full phase cycle for ``issue_url``, emitting events on ``bus``.

    Blocks until the run ends (call it from a worker thread under the TUI).
    ``llm`` and ``pipeline`` default to the real OpenAI-compatible client and the
    real ``RepoPipeline``; tests pass fakes. ``issue_text`` (with ``repo_url``, or an ``issue_url``
    that names the repository) is the fallback for when GitHub cannot be asked for the issue: it is
    used as the issue body and nothing is fetched. Whatever happens, ``output/patch.diff`` and
    ``output/report.md`` are written and a ``done`` event is emitted. Nothing is raised
    to the caller, except that Ctrl-C still propagates once those outputs exist.
    """
    Orchestrator(issue_url, config, bus, llm=llm, pipeline=pipeline, repo_url=repo_url, issue_text=issue_text).run()


# Phases that end with text the model wrote itself; only those can be replaced by their summary.
_COMPRESSIBLE = (PhaseStatus.DONE, PhaseStatus.GAVE_UP)


@dataclass(frozen=True)
class _Verdict:
    passed: bool
    feedback: str = ""


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._repo_url = repo_url
        self._issue_text = issue_text
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

    # ---- INGEST / PROFILE ---------------------------------------------------------------------

    def _ingest(self) -> Ingested:
        self._enter(Phase.INGEST)
        ingested = self._pipeline.ingest(
            self._state.issue_url, repo_url=self._repo_url, issue_text=self._issue_text
        )
        issue = self._state.issue = ingested.issue
        repo = f"{issue.owner}/{issue.repo}"
        if (self._issue_text or "").strip():
            self._emitter.message("system", f"Using the issue text you supplied and cloned {repo}.")
        else:
            self._emitter.message("system", f"Fetched issue #{issue.number} ({issue.title}) and cloned {repo}.")
        return ingested

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
        self._state.understanding = outcome.summary if outcome.done else ""

    def _localize(self) -> None:
        outcome = self._run_phase(Phase.LOCALIZE, phase_kickoff(Phase.LOCALIZE), pin=True)
        if outcome.done:
            self._state.localization = outcome.summary
        else:
            self._state.limit(f"The fault was not localized ({outcome.status.value}: {outcome.summary}).")

    def _reproduce(self) -> None:
        outcome = self._run_phase(Phase.REPRODUCE, phase_kickoff(Phase.REPRODUCE), gate=self._repro_gate, pin=True)
        if not self._state.repro_confirmed:
            self._state.limit(
                f"The bug was not reproduced ({outcome.status.value}: {outcome.summary}); confidence is capped at low."
            )

    def _repro_gate(self, args: dict) -> str | None:
        """Accept REPRODUCE's ``phase_done`` only if the repro really fails when the harness runs it."""
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
        self._state.limit("The reviewer requested changes; one rework round was made and was not re-reviewed.")
        self._patch_until_verified("rework", changes)

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
            if failed < settings.max_patch_attempts:
                kind = "retry"
            elif rollbacks < max_rollbacks and self._rollback(ref, approaches):
                rollbacks += 1
                state.rollbacks += 1
                failed, kind = 0, "rethink"
            else:
                self._abandon_patching(rework, ref)
                return

    def _abandon_patching(self, rework: bool, ref: str | None) -> None:
        state = self._state
        if not rework:
            state.limit(
                f"Verification still failed after {state.patch_attempts} patch attempts and "
                f"{state.rollbacks} rollbacks; the patch is unverified."
            )
        elif self._rollback(ref):
            state.verified = True
            state.limit("The reviewer's requested rework failed verification; the reviewed patch was restored.")
        else:
            state.limit("The reviewer's requested rework failed verification and could not be rolled back.")

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
            state.checks.append(CheckRun(f"repro `{state.repro_cmd}` (re-run by the harness)", passed))
            if not passed:
                return _Verdict(False, self._trim(f"The repro still fails after your patch:\n{text}"))
        outcome = self._run_phase(Phase.VERIFY, verify_kickoff(repro_cmd=state.repro_cmd))
        for record in outcome.records:
            if record.tool == "run_tests":
                target = str(record.args.get("target") or "").strip()
                state.checks.append(CheckRun(f"run_tests {target}".strip(), record.ok))
        if outcome.done:
            return _Verdict(True)
        return _Verdict(False, self._trim(outcome.summary or f"verification did not finish ({outcome.status.value})"))

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
        outcome = self._runner.run(PHASE_SPECS[phase], kickoff, gate)
        text = outcome.summary if outcome.done else f"did not complete ({outcome.status.value}): {outcome.summary}"
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
            patch = self._patch_text()
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
            outcome = self._runner.run(PHASE_SPECS[Phase.FINALIZE], finalize_kickoff(self._facts(patch)))
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
        if state.limitations:
            lines.append("- Known limitations: " + "; ".join(state.limitations))
        return "\n".join(lines)

    def _publish(self, patch: str) -> None:
        state, budget = self._state, self._budget
        has_patch = bool(changed_files(patch))
        report = render_report(state, patch, steps=budget.steps, tokens=budget.tokens, seconds=budget.elapsed)
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


def _describe(result: ExecResult, timeout: int) -> str:
    head = f"[TIMED OUT after {timeout}s]" if result.timed_out else f"[exit code {result.exit_code}]"
    parts = [head]
    if result.stdout.strip():
        parts.append(f"[stdout]\n{result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"[stderr]\n{result.stderr.strip()}")
    return "\n".join(parts)
