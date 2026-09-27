"""Facts gathered during a run: what the report says and how confident we are."""

from __future__ import annotations

from dataclasses import dataclass, field

from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile

# The done event's ``resolved_confidence`` is a fraction (the TUI renders it as a percentage).
SCORE_APPROVED = 0.90  # the repository's tests passed and the review approved: the only way to 0.90
SCORE_UNREVIEWED = 0.75  # the repository's tests passed and the review never reached a verdict: the approval bonus is missing, nothing more
SCORE_CAVEAT = 0.60  # a failing check, a run cut short, or a review that asked for changes
SCORE_NO_TESTS = 0.50  # verified on the model's own repro only: no repository test was run
SCORE_UNVERIFIED = 0.30
HIGH_FROM, MEDIUM_FROM = 0.85, 0.45  # the label is read off the score

REVIEW_NOT_RUN = "not run"
REVIEW_APPROVED = "approved"
REVIEW_CHANGES_REQUESTED = "changes requested"


@dataclass
class CheckRun:
    """One test or repro execution seen during VERIFY.

    ``repository_tests`` is true for a run of the repository's own tests and false for the model's own repro script.
    """

    label: str
    passed: bool
    repository_tests: bool = True


@dataclass
class RunState:
    """Everything the orchestrator learns about one run, in one place."""

    issue_url: str
    issue: IssueRef | None = None
    profile: RepoProfile | None = None
    understanding: str = ""
    localization: str = ""
    repro_cmd: str | None = None
    repro_output: str = ""
    repro_confirmed: bool = False
    verified: bool = False
    checks: list[CheckRun] = field(default_factory=list)
    review: str = REVIEW_NOT_RUN
    final_summary: str = ""
    patch_attempts: int = 0
    rollbacks: int = 0
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # things a reader of the report must not miss
    halted: str = ""  # why the run was cut short (budget, aborted, error, interrupted), if it was
    sanity: str = ""  # the outcome of the pre-delivery patch check, one line for report.md ("" when it did not run)
    sanity_failed: bool = False  # the patch is empty or does not apply even after the one forced-fix retry

    def limit(self, note: str) -> None:
        """Record a known limitation for the report (duplicates are ignored)."""
        if note not in self.limitations:
            self.limitations.append(note)

    def warn(self, note: str) -> None:
        """Record a warning that ``report.md`` shows before everything else (duplicates are ignored)."""
        if note not in self.warnings:
            self.warnings.append(note)

    @property
    def repository_tests_ran(self) -> bool:
        """True if VERIFY ran the repository's own tests (``run_tests``, or a test command through ``run_cmd``)."""
        return any(check.repository_tests for check in self.checks)

    @property
    def review_unfinished(self) -> bool:
        """The review neither approved nor asked for changes (it stalled, hit a cap, or never started)."""
        return self.review not in (REVIEW_APPROVED, REVIEW_CHANGES_REQUESTED)

    def confidence_score(self, has_patch: bool) -> float:
        """The confidence as a fraction, for the done event: what the evidence supports and no more.

        0.90 needs the bug reproduced before the patch, verification passing after it, repository tests run with none
        failing, a run that was not cut short, and the reviewer's approval. A review that never finished costs nothing
        (0.75: only the approval bonus is missing); one that asked for changes, a failing check, or a halted run gives
        0.60. A patch verified on the model's own repro with no repository test run is capped at 0.50 whatever the review
        said. Unverified or never reproduced is 0.30, and no patch at all is 0.
        """
        if not has_patch:
            return 0.0
        if self.sanity_failed:
            return SCORE_UNVERIFIED  # a patch that cannot be applied cannot be trusted whatever else is true of it
        if not (self.repro_confirmed and self.verified):
            return SCORE_UNVERIFIED
        if not self.repository_tests_ran:
            return SCORE_NO_TESTS
        if self.halted or self.review == REVIEW_CHANGES_REQUESTED or not all(c.passed for c in self.checks):
            return SCORE_CAVEAT
        return SCORE_APPROVED if self.review == REVIEW_APPROVED else SCORE_UNREVIEWED

    def confidence(self, has_patch: bool) -> str:
        """``"none"`` | ``"low"`` | ``"medium"`` | ``"high"``: the label for ``confidence_score``."""
        score = self.confidence_score(has_patch)
        if score >= HIGH_FROM:
            return "high"
        if score >= MEDIUM_FROM:
            return "medium"
        return "low" if score > 0 else "none"

    def confidence_notes(self, has_patch: bool) -> list[str]:
        """Sentences for ``report.md`` that explain a score a reader could otherwise misread."""
        if not has_patch or not (self.repro_confirmed and self.verified) or self.sanity_failed:
            return []
        if not self.repository_tests_ran:
            return [
                f"Capped at {SCORE_NO_TESTS:.2f}: no repository tests were run, so the patch was verified on the model's own repro alone."
            ]
        if self.review_unfinished and not self.halted and all(c.passed for c in self.checks):
            return [
                f"The self-review did not finish ({self.review}), but verification passed on the repository's tests and no changes "
                f"were requested, so this does not lower the confidence; an approved review would raise it to {SCORE_APPROVED:.2f}."
            ]
        return []
