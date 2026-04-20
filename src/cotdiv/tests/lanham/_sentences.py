"""Sentence splitting helpers for Lanham-style tests.

Default splitter is a conservative regex — no NLTK dependency. Callers that
want NLTK `punkt` (as used in the original 2307.13702 code) can pass their
own splitter into any test function.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_SENTENCE_BOUNDARY = re.compile(
    r"(?<!\b[A-Z]\.)"
    r"(?<=[.!?])"
    r"(?:\s+|\n+)"
    r"(?=[A-Z0-9\"'\(\[])",
)


def default_sentence_split(text: str) -> list[str]:
    """Regex sentence splitter. Preserves original whitespace inside each sentence.

    Conservative on abbreviations — a capital letter followed by a period is
    treated as initials/abbreviation, not a sentence boundary.
    """
    text = text.strip()
    if not text:
        return []
    pieces = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in pieces if p.strip()]


Splitter = Callable[[str], list[str]]


def prefix_at_fraction(sentences: list[str], fraction: float) -> str:
    """Build a prefix containing ceil(fraction * n) sentences."""
    if not sentences:
        return ""
    if fraction <= 0:
        return ""
    if fraction >= 1:
        return " ".join(sentences)
    n = max(1, round(fraction * len(sentences)))
    return " ".join(sentences[:n])
