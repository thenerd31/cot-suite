"""Metric modules. Importing registers each metric under its canonical name."""

from cotsuite.autoraters import legibility_coverage  # noqa: F401 — registers legibility, coverage
from cotsuite.metrics.verbosity import reasoning_surface_health, verbosity

__all__ = ["reasoning_surface_health", "verbosity"]
