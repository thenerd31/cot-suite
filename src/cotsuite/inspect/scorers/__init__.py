"""Inspect AI scorers wrapping cotsuite metrics."""

from cotsuite.inspect.scorers.chen_cue_injection import cot_chen_cue_injection
from cotsuite.inspect.scorers.legibility_coverage import cot_legibility_coverage
from cotsuite.inspect.scorers.post_hoc_rationalization import (
    cot_post_hoc_rationalization,
)

__all__ = [
    "cot_chen_cue_injection",
    "cot_legibility_coverage",
    "cot_post_hoc_rationalization",
]
