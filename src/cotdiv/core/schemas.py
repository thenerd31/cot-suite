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
    """Output of a Lanham-style causal-intervention test."""

    __test__ = False  # tell pytest this is not a test class

    model_config = ConfigDict(frozen=True)

    name: str
    aoc: float | None = None
    per_fraction: dict[float, float] = Field(default_factory=dict)
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
