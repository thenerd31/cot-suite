"""Inspect AI solvers.

* ``cot_cue_injection_solver`` — inject a Chen 2505.05410 cue into the prompt
  (pairs with ``cotsuite.inspect.scorers.cot_chen_cue_injection``).
* ``cot_bias_injection_solver`` — inject a Turpin 2305.04388 counterfactual bias
  into the prompt (pairs with ``cotsuite.inspect.scorers.cot_turpin_counterfactual``).
* ``cot_lanham_early_answering`` — Lanham 2307.13702 early-answering intervention
  (re-elicits the model-under-test at CoT-prefix fractions, computes AOC; pairs
  with ``cotsuite.inspect.scorers.cot_lanham_early_answering_aoc``). The Lanham POC.

Planned (v0.2): the other three Lanham tests — ``mistake_injection`` and
``paraphrasing`` (each needs a second model role), ``filler_tokens``; and
factored_decomposition.py (Radhakrishnan 2307.11768).
"""

from cotsuite.inspect.solvers.bias_injection import cot_bias_injection_solver
from cotsuite.inspect.solvers.cue_injection import cot_cue_injection_solver
from cotsuite.inspect.solvers.lanham_early_answering import cot_lanham_early_answering

__all__ = [
    "cot_bias_injection_solver",
    "cot_cue_injection_solver",
    "cot_lanham_early_answering",
]
