"""cot-suite: Chain-of-thought monitorability and faithfulness evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cotsuite.core.registry import register_metric, register_test
from cotsuite.core.trajectory import Action, Reasoning, Trajectory, Turn

if TYPE_CHECKING:
    from cotsuite.core.schemas import ScoreResult

# Version is injected at build time by hatch-vcs (writes
# src/cotsuite/_version.py from the latest git tag). When running from
# a checkout without a build (e.g. `pip install -e .` with hatch-vcs
# auto-generation) the file is created on first install. The fallback
# string keeps editable installs from a fresh clone working before a
# `pip install -e .` has run.
try:
    from cotsuite._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "Action",
    "Reasoning",
    "Trajectory",
    "Turn",
    "register_metric",
    "register_test",
    "score_trajectory",
]


def score_trajectory(
    trajectory: Trajectory,
    metrics: list[str],
    autorater: str = "google/gemini-2.5-pro",
    model_for_tests: str | None = None,
    runs: int = 5,
) -> ScoreResult:
    """Run the configured metrics against a Trajectory.

    Thin wrapper over the registry — see `cotsuite.core.registry.run` for the full
    surface. Returns a `ScoreResult` with per-metric `Value ± SD` and metadata.
    """
    from cotsuite.core.registry import run

    return run(
        trajectory,
        metric_names=metrics,
        autorater=autorater,
        model_for_tests=model_for_tests,
        runs=runs,
    )
