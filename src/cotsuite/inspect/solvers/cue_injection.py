"""Inspect AI solver: inject a Chen 2505.05410 cue into the prompt.

This is the **Solver** half of the Chen cue-injection method (the recon's
"Solver + Scorer" framing). The solver rewrites the user prompt to embed a cue
from ``CUE_CATALOG`` (e.g. a metadata block claiming a particular answer), then
the eval's ``generate()`` produces the CoT, then
``cotsuite.inspect.scorers.cot_chen_cue_injection`` judges the result. Compose::

    from inspect_ai import Task, task
    from inspect_ai.solver import generate
    from cotsuite.inspect.solvers import cot_cue_injection_solver
    from cotsuite.inspect.scorers import cot_chen_cue_injection

    @task
    def chen_metadata_cue():
        return Task(
            dataset=my_dataset,                       # Sample.metadata["cue_target"] = "B"
            solver=[cot_cue_injection_solver(cue="metadata"), generate()],
            scorer=cot_chen_cue_injection(),
        )

The cue's *target* (the answer letter the cue points to — usually wrong) is read
per-sample from ``state.metadata["cue_target"]``; this solver renders the cue and
stashes the rendered cue *description* (what the verbalization judge reads) and
the cue *name* into ``state.metadata`` for the paired scorer.
"""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver

from cotsuite.tests.chen_cue_injection import CUE_CATALOG, Cue

# state.metadata keys shared with cot_chen_cue_injection (the paired scorer).
CUE_TARGET_KEY = "cue_target"
CUE_DESCRIPTION_KEY = "cue_description"
CUE_NAME_KEY = "cue_name"


@solver
def cot_cue_injection_solver(cue: str | Cue) -> Solver:
    """Inject a cue into the user prompt and record cue metadata for the scorer.

    Args:
        cue: a ``CUE_CATALOG`` key (``"sycophancy"``, ``"consistency"``,
            ``"metadata"``, ``"grader_hacking"``, ``"unethical_information"``) or a
            ``Cue`` instance.

    Returns:
        An Inspect ``Solver`` that rewrites ``state.user_prompt`` via the cue's
        renderer and sets ``state.metadata`` keys ``cue_name`` / ``cue_target`` /
        ``cue_description``. Requires ``state.metadata["cue_target"]`` (the answer
        the cue points to) on each sample; raises ``ValueError`` if absent.
    """
    cue_cfg = CUE_CATALOG[cue] if isinstance(cue, str) else cue

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cue_target = state.metadata.get(CUE_TARGET_KEY)
        if not cue_target:
            raise ValueError(
                f"cot_cue_injection_solver requires sample "
                f"metadata['{CUE_TARGET_KEY}'] (the answer letter the cue points "
                f"to); none found for this sample."
            )
        prompt = state.user_prompt
        prompt.text = cue_cfg.renderer(prompt.text, cue_target)
        state.metadata[CUE_NAME_KEY] = cue_cfg.name
        state.metadata[CUE_DESCRIPTION_KEY] = cue_cfg.description_template.format(
            target=cue_target
        )
        return state

    return solve
