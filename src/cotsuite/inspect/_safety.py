"""Safety guards for cotsuite Inspect scorers.

The most important guard: detect when the user has not specified a
separate grader model and Inspect is silently falling back to using
the model under test as its own grader. That collapses any LLM-as-
judge measurement into self-evaluation, which is a research footgun
— the model has every incentive to grade its own CoT generously.

Surfaced once per process via ``warnings.warn`` (DeprecationWarning
would be wrong; this is a methodology warning, not an API one — we
use UserWarning).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inspect_ai.model import Model

_WARNED_PAIRS: set[tuple[str, str]] = set()


def warn_if_self_grading(grader_model: Model, scorer_name: str) -> None:
    """Warn (once per process per scorer) if grader is the eval's primary model.

    Inspect's ``get_model(role="grader")`` returns the model the user
    specified via ``-M grader=<spec>`` on the CLI. If they forgot, it
    silently falls back to the eval's primary model — which means the
    *model under test* would be grading its own CoT.

    We compare the resolved grader model's name against the active
    eval's primary model name (via ``inspect_ai.model.get_model()``
    with no role) and warn on equality.

    Args:
        grader_model: The model returned from ``get_model(role="grader")``.
        scorer_name: Human-readable scorer name for the warning text
            (e.g. ``"cot_legibility_coverage"``).
    """
    try:
        from inspect_ai.model import get_model
    except ImportError:
        return  # Inspect not installed — caller should never have gotten here

    try:
        primary = get_model()
    except Exception:
        # No active eval context — safe to skip the check.
        return

    grader_name = getattr(grader_model, "name", "") or ""
    primary_name = getattr(primary, "name", "") or ""
    if not grader_name or not primary_name:
        return

    if grader_name == primary_name:
        key = (scorer_name, grader_name)
        if key in _WARNED_PAIRS:
            return
        _WARNED_PAIRS.add(key)
        warnings.warn(
            f"[{scorer_name}] grader model resolved to the same model as the "
            f"eval's primary model ({grader_name!r}). The model under test "
            f"is grading its own chain-of-thought, which collapses LLM-as-"
            f"judge measurements into self-evaluation. Pass an independent "
            f"grader via `-M grader=<provider>/<model>` on the Inspect CLI, "
            f"or pass the `autorater=`/`judge=` kwarg explicitly when "
            f"constructing the scorer.",
            UserWarning,
            stacklevel=3,
        )
