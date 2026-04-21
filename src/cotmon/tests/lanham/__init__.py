"""Lanham et al. 2307.13702 — four-test faithfulness suite.

Ships:
- early_answering: prefix-ablation retention curve + length-weighted AOC.
- mistake_injection: sentence-corruption + optional tail-resample AOC.
- paraphrasing: steganography-gap between original and paraphrased prefixes.
- filler_tokens: extra-test-time-compute sanity sweep.
"""

from cotmon.tests.lanham.early_answering import early_answering
from cotmon.tests.lanham.filler_tokens import filler_tokens
from cotmon.tests.lanham.mistake_injection import mistake_injection
from cotmon.tests.lanham.paraphrasing import paraphrasing

__all__ = [
    "early_answering",
    "filler_tokens",
    "mistake_injection",
    "paraphrasing",
]
