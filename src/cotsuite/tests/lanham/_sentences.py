"""Sentence splitting helpers for Lanham-style tests.

Two splitters ship:

    - ``default_sentence_split`` — regex-based, zero dependencies. The default
      for every Lanham test. Conservative on abbreviations.
    - ``nltk_sentence_split`` — NLTK Punkt, matching Lanham 2307.13702's
      original code. Requires the optional ``nltk`` install extra; the
      punkt_tab data is downloaded lazily on first use.

Callers pass whichever splitter they prefer into the ``sentence_splitter``
argument of any Lanham test. For published-number reproductions, prefer
``nltk_sentence_split``; for local / low-dependency runs the regex is fine.
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


def nltk_sentence_split(text: str) -> list[str]:
    """NLTK Punkt sentence splitter — matches Lanham 2307.13702's original code.

    Requires the ``[nlp]`` install extra: ``pip install 'cot-suite[nlp]'``.
    The ``punkt_tab`` data is downloaded lazily on first use (one-off ~40 KB).
    """
    try:
        import nltk
    except ImportError as exc:
        raise ImportError(
            "nltk_sentence_split requires the optional 'nlp' extra — "
            "install with `pip install 'cot-suite[nlp]'`.",
        ) from exc
    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)


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
