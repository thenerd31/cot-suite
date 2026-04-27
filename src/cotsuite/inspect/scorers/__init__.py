"""Inspect AI scorers wrapping cotsuite metrics."""

from cotsuite.inspect.scorers.legibility_coverage import cot_legibility_coverage
from cotsuite.inspect.scorers.post_hoc_rationalization import (
    cot_post_hoc_rationalization,
)

__all__ = [
    "cot_legibility_coverage",
    "cot_post_hoc_rationalization",
]
