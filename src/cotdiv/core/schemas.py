"""Result schemas for metrics and tests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Classification = Literal["computational", "mixed", "rationalization", "unknown"]


class MetricValue(BaseModel):
    """One metric's scored value plus dispersion and provenance."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    stderr: float | None = None
    stddev: float | None = None
    n_runs: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class TestResult(BaseModel):
    """Output of a Lanham-style causal-intervention test.

    Fields:
        name: canonical test name, e.g. ``lanham.early_answering``.
        aoc: the paper's length-weighted AOC, when the test's scalar IS
            Lanham's AOC. Set to ``None`` for tests whose primary output is
            a curve (paraphrasing, filler_tokens) — see ``raw_curve``.
        per_fraction: headline retention/accuracy curve keyed by prefix
            fraction (or length). Shape matches the paper's canonical plot
            for that test.
        raw_curve: paper-equivalent curve data. Same shape as per_fraction
            but semantically tagged as "this is what the paper reports."
            Empty dict when per_fraction is already the paper-equivalent curve.
        synthesis: CoT-Divergence-invented scalar summaries keyed by versioned
            name (e.g. ``cotdiv_paraphrasing_gap_v1``). These are NOT from
            the paper — the paper reports raw curves. See individual tests
            for definitions.
        raw: untyped debug payload (completions, per-index info, etc.).
    """

    __test__ = False  # tell pytest this is not a test class

    model_config = ConfigDict(frozen=True)

    name: str
    aoc: float | None = None
    per_fraction: dict[float, float] = Field(default_factory=dict)
    raw_curve: dict[float, float] = Field(default_factory=dict)
    synthesis: dict[str, float] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class Divergence(BaseModel):
    """A single CoT↔action or CoT↔activation divergence event."""

    model_config = ConfigDict(frozen=True)

    turn_index: int
    divergence_type: Literal["cot_action_mismatch", "cot_activation_mismatch", "restoration_error"]
    score: float
    evidence: str | None = None


class ScoreResult(BaseModel):
    """Aggregated output of `score_trajectory`."""

    model_config = ConfigDict(frozen=True)

    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    tests: dict[str, TestResult] = Field(default_factory=dict)
    divergences: list[Divergence] = Field(default_factory=list)
    classification: Classification = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __getattr__(self, name: str) -> MetricValue:  # pragma: no cover - sugar
        if name in self.metrics:
            return self.metrics[name]
        if name in self.tests:
            return self.tests[name]  # type: ignore[return-value]
        raise AttributeError(name)
