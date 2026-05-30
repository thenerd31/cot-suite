"""Inspect AI solver: Lanham 2307.13702 early-answering AOC (proof-of-concept).

The Lanham POC (Phase 3, option iii). Lanham's tests are **mid-trajectory
interventions** — early-answering truncates the CoT at each prefix fraction and
**re-elicits the model-under-test**, then measures the area-over-the-curve (AOC).
That re-elicitation does not fit Inspect's ``score(state, target)`` (which scores
one completed trajectory), so the work lives **here, in a solver** — running the
N re-elicitations as a custom loop (Inspect has no N-re-elicit primitive) and
stashing the AOC into ``state.metadata`` for the thin
``cotsuite.inspect.scorers.cot_lanham_early_answering_aoc`` scorer to surface.

**Reuses the native logic untouched.** The AOC is computed by the tested
``cotsuite.tests.lanham.early_answering`` ``@register_test`` function — invoked,
not reimplemented — with the eval's primary model (``get_model()``, the
model-under-test) adapted to a ``GraderClient`` via ``InspectGraderAdapter``.

**Single model only.** early-answering needs no second model. The other three
Lanham tests are deferred to v0.2: ``mistake_injection`` and ``paraphrasing``
require a *second* model role (a mistake generator / a paraphraser); ``filler_tokens``
is single-model but deferred with them.

Compose (the solver runs AFTER ``generate()`` so it sees the model's CoT)::

    from inspect_ai.solver import generate
    from cotsuite.inspect.solvers import cot_lanham_early_answering
    from cotsuite.inspect.scorers import cot_lanham_early_answering_aoc

    @task
    def lanham_early_answering():
        return Task(
            dataset=ds,
            solver=[generate(), cot_lanham_early_answering()],
            scorer=cot_lanham_early_answering_aoc(),
        )
"""

from __future__ import annotations

from collections.abc import Sequence

from inspect_ai.model import get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from cotsuite.judges import InspectGraderAdapter
from cotsuite.tests.lanham._extractors import AnswerExtractor, mcq_answer_extractor
from cotsuite.tests.lanham.early_answering import DEFAULT_FRACTIONS
from cotsuite.tests.lanham.early_answering import early_answering as _native_early_answering

# state.metadata key shared with cot_lanham_early_answering_aoc (the thin scorer).
LANHAM_EARLY_ANSWERING_KEY = "lanham_early_answering"


@solver
def cot_lanham_early_answering(
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    answer_extractor: AnswerExtractor = mcq_answer_extractor,
    length_weighted: bool = True,
) -> Solver:
    """Re-elicit the model-under-test at CoT-prefix fractions; stash Lanham AOC.

    Runs after ``generate()``: reads the model's CoT from ``state.messages`` and
    the question from ``state.input_text``, then invokes the native
    ``early_answering`` (re-eliciting ``get_model()`` once per prefix fraction)
    and writes ``{aoc, per_fraction, sentence_count, full_answer}`` to
    ``state.metadata['lanham_early_answering']``.

    Args:
        fractions: CoT-prefix fractions to probe (default: Lanham's 5-point sweep).
        answer_extractor: ``(completion: str) -> str`` for the re-elicited answer
            (default: MCQ letter extraction).
        length_weighted: weight per-fraction AOC contribution by cumulative
            sentence count (the paper's convention).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cot = _extract_reasoning(state.messages)
        if not cot.strip():
            state.metadata[LANHAM_EARLY_ANSWERING_KEY] = {
                "aoc": None,
                "skip_reason": "empty_reasoning",
            }
            return state

        # The eval's primary model = the model-under-test, adapted to GraderClient.
        adapter = InspectGraderAdapter(get_model())
        result = await _native_early_answering(
            model=adapter,
            question=state.input_text,
            cot=cot,
            fractions=fractions,
            answer_extractor=answer_extractor,
            length_weighted=length_weighted,
        )
        state.metadata[LANHAM_EARLY_ANSWERING_KEY] = {
            "aoc": result.aoc,
            "per_fraction": result.per_fraction,
            "sentence_count": result.raw.get("sentence_count"),
            "full_answer": result.raw.get("full_answer"),
        }
        return state

    return solve


def _extract_reasoning(messages) -> str:  # type: ignore[no-untyped-def]
    """Pull ``ContentReasoning`` text from an Inspect message stream.

    Mirrors ``cotsuite.inspect.scorers.post_hoc_rationalization._extract_reasoning``.
    """
    parts: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if type(block).__name__ == "ContentReasoning":
                    text = getattr(block, "reasoning", None) or getattr(block, "text", "")
                    if text:
                        parts.append(text)
    return "\n\n".join(parts)
