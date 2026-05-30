"""Inspect AI solver: inject a Turpin 2305.04388 counterfactual bias into the prompt.

The **Solver** half of the Turpin bias-injection method (the recon's "Solver +
Scorer" framing). The solver rewrites the user prompt to embed a bias from
``BIAS_CATALOG`` (default: ``suggested_answer`` — "I personally think the answer
is (X)"), the eval's ``generate()`` produces the biased CoT, and
``cotsuite.inspect.scorers.cot_turpin_counterfactual`` judges the result. Compose::

    from inspect_ai import Task, task
    from inspect_ai.solver import generate
    from cotsuite.inspect.solvers import cot_bias_injection_solver
    from cotsuite.inspect.scorers import cot_turpin_counterfactual

    @task
    def turpin_suggested_answer():
        return Task(
            dataset=my_dataset,               # Sample.metadata["bias_target"] = "B" (optional)
            solver=[cot_bias_injection_solver(bias="suggested_answer"), generate()],
            scorer=cot_turpin_counterfactual(),
        )

The bias *target* (the answer the cue points to) is read per-sample from
``state.metadata["bias_target"]``; when absent, the bias's ``default_target`` is
used (and the fixed-target injector, matching the native path's behaviour). The
effective target, the bias name, and the bias description (what the verbalization
judge reads) are stashed into ``state.metadata`` for the paired scorer.

**This is a NEW Inspect-facing surface.** It does not touch
``cotsuite.tests.turpin_counterfactual`` (the ``@register_test`` path) or the
±0.08pp-validated B2 reproduction (``scripts/validate_b2_turpin_*``); the
cross-arm ``accuracy_drop`` lives there, not here.
"""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver

from cotsuite.tests.turpin_counterfactual import BIAS_CATALOG, BiasConfig

# state.metadata keys shared with cot_turpin_counterfactual (the paired scorer).
BIAS_TARGET_KEY = "bias_target"
BIAS_DESCRIPTION_KEY = "bias_description"
BIAS_NAME_KEY = "bias_name"


@solver
def cot_bias_injection_solver(bias: str | BiasConfig = "suggested_answer") -> Solver:
    """Inject a counterfactual bias into the user prompt and record bias metadata.

    Args:
        bias: a ``BIAS_CATALOG`` key (``"suggested_answer"``, ``"always_a_fewshot"``)
            or a ``BiasConfig``. Defaults to ``"suggested_answer"`` — Turpin's
            suggested-answer cue and the bias B2 validated.

    Returns:
        An Inspect ``Solver`` that rewrites ``state.user_prompt`` via the bias's
        target injector (when the sample carries ``metadata["bias_target"]``) or
        its fixed injector (falling back to the bias's ``default_target``), and
        sets ``state.metadata`` keys ``bias_name`` / ``bias_target`` (the
        *effective* target) / ``bias_description``.
    """
    bias_cfg = BIAS_CATALOG[bias] if isinstance(bias, str) else bias

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        explicit_target = state.metadata.get(BIAS_TARGET_KEY)
        target = explicit_target or bias_cfg.default_target
        prompt = state.user_prompt
        if bias_cfg.target_injector is not None and explicit_target is not None:
            prompt.text = bias_cfg.target_injector(prompt.text, target)
        else:
            prompt.text = bias_cfg.injector(prompt.text)
        state.metadata[BIAS_NAME_KEY] = bias_cfg.name
        state.metadata[BIAS_TARGET_KEY] = target
        state.metadata[BIAS_DESCRIPTION_KEY] = bias_cfg.description
        return state

    return solve
