"""Lanham et al. 2307.13702 — four-test faithfulness suite.

Ships:
- early_answering: prefix-ablation retention curve + length-weighted AOC.
- mistake_injection: sentence-corruption + optional tail-resample AOC.

Planned:
- paraphrasing (Task #12)
- filler_tokens (Task #12)
"""

from cotdiv.tests.lanham.early_answering import early_answering
from cotdiv.tests.lanham.mistake_injection import mistake_injection

__all__ = ["early_answering", "mistake_injection"]
