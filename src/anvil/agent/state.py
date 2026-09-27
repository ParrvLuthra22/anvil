"""Facts gathered during a run: what the report says and how confident we are."""

from __future__ import annotations

from dataclasses import dataclass, field

from anvil.repo.ingest import IssueRef
from anvil.repo.profile import RepoProfile

# The done event's ``resolved_confidence`` is a fraction (the TUI renders it as a percentage).
CONFIDENCE_SCORES = {"high": 0.9, "medium": 0.6, "low": 0.3, "none": 0.0}

REVIEW_NOT_RUN = "not run"
REVIEW_APPROVED = "approved"
REVIEW_CHANGES_REQUESTED = "changes requested"


@dataclass
class CheckRun:
    """One test or repro execution seen during VERIFY."""

    label: str
    passed: bool


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
    verify_without_tests: bool = False  # VERIFY was closed by the harness on the repro alone: no test was run

    def limit(self, note: str) -> None:
        """Record a known limitation for the report (duplicates are ignored)."""
        if note not in self.limitations:
            self.limitations.append(note)

    def warn(self, note: str) -> None:
        """Record a warning that ``report.md`` shows before everything else (duplicates are ignored)."""
        if note not in self.warnings:
            self.warnings.append(note)

    def confidence(self, has_patch: bool) -> str:
        """``"none"`` | ``"low"`` | ``"medium"`` | ``"high"``.

        High needs all of: the bug reproduced before the patch, verification
        passing after it (with at least one test run: ``verify_without_tests``
        is a verification on the repro alone), no failing test run, and the
        reviewer's approval. A verified patch with a caveat (unapproved review,
        failing tests, no test run) is medium; anything unverified or never
        reproduced is low. A run that was cut short (``halted``) is never higher
        than medium, whatever it had by then.
        """
        if not has_patch:
            return "none"
        if self.sanity_failed:
            return "low"  # a patch that cannot be applied cannot be trusted whatever else is true of it
        if not (self.repro_confirmed and self.verified):
            return "low"
        clean = (
            self.review == REVIEW_APPROVED
            and all(c.passed for c in self.checks)
            and not self.halted
            and not self.verify_without_tests
        )
        return "high" if clean else "medium"

    def confidence_score(self, has_patch: bool) -> float:
        """Numeric form of ``confidence`` for the done event."""
        return CONFIDENCE_SCORES[self.confidence(has_patch)]
