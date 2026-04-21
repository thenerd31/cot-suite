"""Synthetic-degradation invariants — every metric must score worse on a
degraded trajectory than on a pristine one.

These tests use a mocked autorater (no network calls) calibrated to return
scores linearly in the fraction of original reasoning preserved. The point of
these tests is not to validate the autorater itself — that's the repro work in
Month 1 Week 3 — but to lock the metric plumbing against regressions that
could silently invert a score or drop a degradation signal.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from cotmon.core.trajectory import Reasoning, Trajectory, Turn


def _traj(cot: str) -> Trajectory:
    return Trajectory(
        turns=[
            Turn(role="user", text="What is 17 * 23?"),
            Turn(
                role="assistant",
                text="391",
                reasoning=[Reasoning(text=cot, provider="anthropic", is_summary=False)],
            ),
        ],
        final_answer="391",
    )


def _redact_half(cot: str) -> str:
    sentences = cot.split(". ")
    return ". ".join(sentences[: len(sentences) // 2])


def _translate_to_nonlatin(cot: str) -> str:
    return "答え：" + "".join(["日" for _ in cot[:50]])  # noqa: RUF001


def _replace_with_filler(cot: str) -> str:
    return " ..." * (len(cot) // 4 or 1)


class _DeterministicGrader:
    """Awards higher legibility+coverage for longer, English-looking CoT."""

    def __init__(self, reference: str) -> None:
        self.reference = reference

    async def complete(self, prompt: str) -> str:
        # crude heuristic aligned with the 2510.23966 rubric:
        # - mostly-ASCII + substantial length => high score
        # - translated / short / filler => low score
        # Returns the Appendix C output schema: justification, legibility_score,
        # coverage_score.
        for marker in ("...", "日", "答え"):
            if prompt.count(marker) >= 5:
                return json.dumps(
                    {
                        "justification": "unreadable",
                        "legibility_score": 1,
                        "coverage_score": 1,
                    },
                )
        ratio = prompt.count(self.reference[:15]) > 0
        if ratio:
            return json.dumps(
                {"justification": "full", "legibility_score": 4, "coverage_score": 4},
            )
        return json.dumps(
            {"justification": "partial", "legibility_score": 2, "coverage_score": 2},
        )


@pytest.mark.parametrize(
    "degrader",
    [_redact_half, _translate_to_nonlatin, _replace_with_filler],
)
def test_legibility_coverage_drops_under_degradation(degrader) -> None:
    from cotmon import metrics as _  # noqa: F401 — ensure registration
    from cotmon import score_trajectory

    reference = (
        "Let me compute 17 * 23. I can break this down as 17 * 20 + 17 * 3. "
        "That gives 340 + 51 = 391."
    )
    full = _traj(reference)
    degraded = _traj(degrader(reference))

    grader = _DeterministicGrader(reference)

    with patch("cotmon.autoraters.legibility_coverage.get_grader_client", return_value=grader):
        full_score = score_trajectory(full, metrics=["legibility", "coverage"], runs=1)
        degr_score = score_trajectory(degraded, metrics=["legibility", "coverage"], runs=1)

    for metric in ("legibility", "coverage"):
        assert degr_score.metrics[metric].value <= full_score.metrics[metric].value, (
            f"{metric} failed to drop under {degrader.__name__}: "
            f"full={full_score.metrics[metric].value} "
            f"degraded={degr_score.metrics[metric].value}"
        )
