"""Inspect AI scorers wrapping cotsuite metrics."""

from cotsuite.inspect.scorers.chen_cue_injection import cot_chen_cue_injection
from cotsuite.inspect.scorers.lanham_early_answering import cot_lanham_early_answering_aoc
from cotsuite.inspect.scorers.legibility_coverage import cot_legibility_coverage
from cotsuite.inspect.scorers.post_hoc_rationalization import (
    cot_post_hoc_rationalization,
)
from cotsuite.inspect.scorers.turpin_counterfactual import cot_turpin_counterfactual

__all__ = [
    "cot_chen_cue_injection",
    "cot_lanham_early_answering_aoc",
    "cot_legibility_coverage",
    "cot_post_hoc_rationalization",
    "cot_turpin_counterfactual",
]
