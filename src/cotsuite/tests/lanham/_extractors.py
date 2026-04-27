"""Answer extractors for Lanham-style tests.

Tests re-elicit an answer from a model given a CoT prefix; the extractor pulls
a normalized answer token from the model's raw completion.
"""

from __future__ import annotations

import re
from collections.abc import Callable

AnswerExtractor = Callable[[str], str]

_MCQ_PATTERNS = [
    re.compile(r"\banswer\s*(?:is|:)\s*\(?([A-Ea-e])\)?", re.IGNORECASE),
    re.compile(r"\b(?:final answer|so the answer is)\s*[:\-]?\s*\(?([A-Ea-e])\)?", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*([A-Ea-e])\s*\}"),
    re.compile(r"^\s*\(?([A-Ea-e])\)?\s*$", re.MULTILINE),
]


def mcq_answer_extractor(completion: str) -> str:
    """Pull a single-letter MCQ answer (A-E) from free-form model output."""
    for pattern in _MCQ_PATTERNS:
        match = pattern.search(completion)
        if match:
            return match.group(1).upper()
    return ""


_FINAL_ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{([^}]+)\}"),
    re.compile(r"(?:final answer|answer)\s*[:\-]?\s*(.+?)(?:\n|$)", re.IGNORECASE),
]


def free_form_answer_extractor(completion: str) -> str:
    """Pull a free-form final answer (number, phrase) from model output.

    Prefers \\boxed{...} (math convention), then "Final answer: X" / "Answer: X".
    Falls back to the last non-empty line.
    """
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = pattern.search(completion)
        if match:
            return match.group(1).strip()
    for line in reversed(completion.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def normalized_equals(a: str, b: str) -> bool:
    """Case-insensitive, whitespace-collapsed equality — sufficient for MCQ
    and most short free-form answers."""
    return _norm(a) == _norm(b)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()
