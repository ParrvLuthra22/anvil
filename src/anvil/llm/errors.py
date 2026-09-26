"""Typed errors raised by the LLM layer."""

from __future__ import annotations


class LLMError(Exception):
    """A model call failed for good (after retries, where retries apply).

    Messages are scrubbed of the API key, so they are safe to log, trace and
    show in the TUI. ``retryable`` says whether the failure was of a transient
    kind (rate limit, server error, network) whose retries were exhausted.
    ``detail`` is a short, scrubbed excerpt of the provider's error body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        attempts: int = 0,
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts
        self.detail = detail


class LLMConfigError(LLMError):
    """Configuration is missing or invalid (for example ``AI_API_KEY`` is not set)."""
