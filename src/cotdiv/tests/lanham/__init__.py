"""Lanham et al. 2307.13702 — four-test faithfulness suite.

Ships:
- early_answering: prefix-ablation retention curve + length-weighted AOC.

Planned:
- mistake_injection (Task #11)
- paraphrasing (Task #12)
- filler_tokens (Task #12)
"""

from cotdiv.tests.lanham.early_answering import early_answering

__all__ = ["early_answering"]
