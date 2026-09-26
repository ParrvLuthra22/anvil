"""Small text helpers for keeping model-facing content within its size budget."""

from __future__ import annotations


def clip_middle(text: str, limit: int) -> str:
    """Shorten ``text`` to about ``limit`` characters, keeping its head and tail.

    Failures usually announce themselves at the start (the command) and the end
    (the traceback or summary), so the middle is what gets dropped.
    """
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return f"{text[:half]}\n... [{len(text) - 2 * half} chars omitted] ...\n{text[-half:]}"


def clip_head(text: str, limit: int) -> str:
    """Keep the first ``limit`` characters of ``text``, noting how much was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [{len(text) - limit} more chars]"
