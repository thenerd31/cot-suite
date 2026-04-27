"""Smoke tests for cotsuite's Inspect AI integration.

We don't run live Inspect evals here — that requires the Inspect CLI
+ a model. Instead we check three integration invariants:

1. The two scorer factories (`cot_legibility_coverage`,
   `cot_post_hoc_rationalization`) are importable and produce Scorer
   objects with the expected registry names.
2. The `[project.entry-points.inspect_ai]` entry point is registered
   under the name `cotsuite` and points at `cotsuite.inspect`.
3. Importing `cotsuite.inspect` triggers both scorer registrations
   (the entry-point loader does this in production; we exercise the
   same import path here).

The actual end-to-end Inspect smoke test (run a tiny eval against
mockllm/model) is deferred to v0.2 when we have more scorers and
the Solver+Scorer pattern in flight.
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import entry_points


def test_entry_point_registered() -> None:
    eps = entry_points(group="inspect_ai")
    matches = [ep for ep in eps if ep.name == "cotsuite"]
    assert matches, "cotsuite entry-point not registered under group=inspect_ai"
    ep = matches[0]
    assert ep.value == "cotsuite.inspect", (
        f"expected cotsuite.inspect, got {ep.value!r}"
    )


def test_entry_point_loadable() -> None:
    """Loading the entry-point should import cotsuite.inspect, which
    transitively imports both scorers and fires their `@scorer`
    decorators."""
    # Strip cached imports so we re-trigger the decorator side-effects.
    for cached in [k for k in sys.modules if k.startswith("cotsuite.inspect")]:
        del sys.modules[cached]
    # Loading the entry-point exec's the target module.
    eps = entry_points(group="inspect_ai")
    [ep] = [e for e in eps if e.name == "cotsuite"]
    module = ep.load()
    # The module should expose its scorers submodule.
    assert hasattr(module, "scorers")


def test_legibility_coverage_registers_with_inspect() -> None:
    importlib.import_module("cotsuite.inspect")
    from inspect_ai._util.registry import registry_info

    from cotsuite.inspect.scorers import cot_legibility_coverage

    info = registry_info(cot_legibility_coverage())
    assert info.name == "cot_legibility_coverage", (
        f"expected registry name 'cot_legibility_coverage', got {info.name!r}"
    )


def test_post_hoc_rationalization_registers_with_inspect() -> None:
    importlib.import_module("cotsuite.inspect")
    from inspect_ai._util.registry import registry_info

    from cotsuite.inspect.scorers import cot_post_hoc_rationalization

    info = registry_info(cot_post_hoc_rationalization())
    assert info.name == "cot_post_hoc_rationalization", (
        f"expected registry name 'cot_post_hoc_rationalization', got {info.name!r}"
    )


def test_phr_default_answer_extractor() -> None:
    from cotsuite.inspect.scorers.post_hoc_rationalization import (
        _default_final_answer_extractor as ext,
    )

    assert ext("Answer: A") == "A"
    assert ext("answer is (B).") == "B"
    assert ext("\\boxed{C}") == "C"
    assert ext("the answer: D") == "D"
    assert ext("no letter here") == ""
    # Boxed should win over a later 'answer:' if both present.
    assert ext("\\boxed{A} (incorrect; answer: B)") == "A"


def test_self_grading_guard_module_importable() -> None:
    """The guard helper itself shouldn't import-error when Inspect is
    not in an active eval context."""
    from cotsuite.inspect import _safety

    assert callable(_safety.warn_if_self_grading)
