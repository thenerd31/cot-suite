"""Metric and test registration + top-level orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from cotdiv.core.schemas import MetricValue, ScoreResult, TestResult
from cotdiv.core.trajectory import Trajectory

MetricFn = Callable[..., Awaitable[MetricValue]]
TestFn = Callable[..., Awaitable[TestResult]]

_METRICS: dict[str, MetricFn] = {}
_TESTS: dict[str, TestFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator — register an async metric function under a stable name."""

    def inner(fn: MetricFn) -> MetricFn:
        if name in _METRICS:
            raise ValueError(f"metric already registered: {name}")
        _METRICS[name] = fn
        return fn

    return inner


def register_test(name: str) -> Callable[[TestFn], TestFn]:
    """Decorator — register an async causal-intervention test."""

    def inner(fn: TestFn) -> TestFn:
        if name in _TESTS:
            raise ValueError(f"test already registered: {name}")
        _TESTS[name] = fn
        return fn

    return inner


def list_metrics() -> list[str]:
    return sorted(_METRICS)


def list_tests() -> list[str]:
    return sorted(_TESTS)


async def _run_async(
    trajectory: Trajectory,
    metric_names: list[str],
    autorater: str,
    model_for_tests: str | None,
    runs: int,
) -> ScoreResult:
    metric_coros: dict[str, Awaitable[MetricValue]] = {}
    for name in metric_names:
        if name not in _METRICS:
            raise KeyError(f"unknown metric: {name!r}. Registered: {list_metrics()}")
        metric_coros[name] = _METRICS[name](
            trajectory,
            autorater=autorater,
            model_for_tests=model_for_tests,
            runs=runs,
        )
    results = await asyncio.gather(*metric_coros.values(), return_exceptions=True)
    metrics: dict[str, MetricValue] = {}
    errors: dict[str, str] = {}
    for name, res in zip(metric_coros.keys(), results, strict=True):
        if isinstance(res, BaseException):
            errors[name] = f"{type(res).__name__}: {res}"
        else:
            metrics[name] = res
    metadata: dict[str, Any] = {"autorater": autorater, "runs": runs}
    if errors:
        metadata["errors"] = errors
    return ScoreResult(metrics=metrics, metadata=metadata)


def run(
    trajectory: Trajectory,
    metric_names: list[str],
    autorater: str = "google/gemini-2.5-pro",
    model_for_tests: str | None = None,
    runs: int = 5,
) -> ScoreResult:
    """Synchronous entrypoint used by `cotdiv.score_trajectory`."""
    return asyncio.run(
        _run_async(trajectory, metric_names, autorater, model_for_tests, runs),
    )
