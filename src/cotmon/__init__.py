"""cot-monitor: Chain-of-thought monitorability and faithfulness evaluation."""

from cotmon.core.registry import register_metric, register_test
from cotmon.core.trajectory import Action, Reasoning, Trajectory, Turn

__version__ = "0.0.1"

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
):
    """Run the configured metrics against a Trajectory.

    Thin wrapper over the registry — see `cotmon.core.registry.run` for the full
    surface. Returns a `ScoreResult` with per-metric `Value ± SD` and metadata.
    """
    from cotmon.core.registry import run

    return run(
        trajectory,
        metric_names=metrics,
        autorater=autorater,
        model_for_tests=model_for_tests,
        runs=runs,
    )
