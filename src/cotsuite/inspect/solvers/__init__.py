"""Inspect AI solvers.

* ``cot_cue_injection_solver`` — inject a Chen 2505.05410 cue into the prompt
  (the Solver half of the cue-injection scorer; pairs with
  ``cotsuite.inspect.scorers.cot_chen_cue_injection``).

Planned: factored_decomposition.py (Radhakrishnan 2307.11768); a Turpin
bias-injection solver (Phase 2).
"""

from cotsuite.inspect.solvers.cue_injection import cot_cue_injection_solver

__all__ = ["cot_cue_injection_solver"]
