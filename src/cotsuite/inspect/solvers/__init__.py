"""Inspect AI solvers.

* ``cot_cue_injection_solver`` — inject a Chen 2505.05410 cue into the prompt
  (pairs with ``cotsuite.inspect.scorers.cot_chen_cue_injection``).
* ``cot_bias_injection_solver`` — inject a Turpin 2305.04388 counterfactual bias
  into the prompt (pairs with ``cotsuite.inspect.scorers.cot_turpin_counterfactual``).

Planned: factored_decomposition.py (Radhakrishnan 2307.11768).
"""

from cotsuite.inspect.solvers.bias_injection import cot_bias_injection_solver
from cotsuite.inspect.solvers.cue_injection import cot_cue_injection_solver

__all__ = ["cot_bias_injection_solver", "cot_cue_injection_solver"]
