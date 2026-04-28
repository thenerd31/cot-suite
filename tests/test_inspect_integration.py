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
    assert ep.value == "cotsuite.inspect", f"expected cotsuite.inspect, got {ep.value!r}"


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
    """Sanity-check the default extractor wired in from cotsuite.parsing.

    Post-fix (2026-04-27, EVAL_VERSION 1.0.0 → 1.1.0) the colon-or-dash
    is MANDATORY — prose like "answer is (B)" without a colon is
    correctly unscorable. See tests/test_parser_bug_diagnosis.py for
    the bug + fix history.
    """
    from cotsuite.inspect.scorers.post_hoc_rationalization import (
        _default_final_answer_extractor as ext,
    )

    assert ext("Answer: A") == "A"
    assert ext("\\boxed{C}") == "C"
    assert ext("the answer: D") == "D"
    assert ext("no letter here") == ""
    # Mandatory colon — "answer is X" without a colon is now unscorable.
    assert ext("answer is (B).") == ""
    # Boxed (layer 1) wins over a later generic 'answer:' (layer 3).
    assert ext("\\boxed{A} (incorrect; answer: B)") == "A"


def test_self_grading_guard_module_importable() -> None:
    """The guard helper itself shouldn't import-error when Inspect is
    not in an active eval context."""
    from cotsuite.inspect import _safety

    assert callable(_safety.warn_if_self_grading)


def test_scorer_modules_export_eval_version() -> None:
    """Both Inspect scorers must expose a top-level EVAL_VERSION constant.

    Bumped on any methodology change that breaks numeric comparability
    of results across runs. Stamped into Score.metadata so downstream
    consumers can filter / partition results by evaluation version.
    """
    from cotsuite.inspect.scorers import legibility_coverage as lc
    from cotsuite.inspect.scorers import post_hoc_rationalization as phr

    assert hasattr(lc, "EVAL_VERSION"), "legibility_coverage missing EVAL_VERSION"
    assert hasattr(phr, "EVAL_VERSION"), "post_hoc_rationalization missing EVAL_VERSION"
    # Sanity-check format: must be parseable as semver-style major.minor.patch
    for mod_name, version in [
        ("legibility_coverage", lc.EVAL_VERSION),
        ("post_hoc_rationalization", phr.EVAL_VERSION),
    ]:
        parts = version.split(".")
        assert len(parts) == 3, f"{mod_name} EVAL_VERSION not semver: {version!r}"
        assert all(p.isdigit() for p in parts), f"{mod_name} EVAL_VERSION not numeric: {version!r}"


def test_scorer_eval_version_pinned_snapshot() -> None:
    """SHA-style snapshot test: the v0.1 launch values are PINNED.

    To bump either scorer's EVAL_VERSION, update this snapshot in the
    same commit. The intent is to force a deliberate decision — bumping
    EVAL_VERSION breaks numeric comparability with prior runs and should
    be conscious, not accidental. This test plays the same role syrupy
    snapshots would (and is simpler to audit in code review than a
    binary __snapshots__/ file).
    """
    from cotsuite.inspect.scorers import legibility_coverage as lc
    from cotsuite.inspect.scorers import post_hoc_rationalization as phr

    # Bumped 2026-04-27: post_hoc_rationalization 1.0.0 → 1.1.0 when
    # the answer-extractor parser was rewritten to fix a colon-optional
    # / first-match-wins bug. PHR numeric outputs are NOT comparable
    # across the version boundary. See AUDIT.md and the methodology
    # note in cotsuite.inspect.scorers.post_hoc_rationalization for
    # the full bug + fix narrative.
    expected = {
        "legibility_coverage": "1.0.0",
        "post_hoc_rationalization": "1.1.0",
    }
    actual = {
        "legibility_coverage": lc.EVAL_VERSION,
        "post_hoc_rationalization": phr.EVAL_VERSION,
    }
    assert actual == expected, (
        f"EVAL_VERSION drift detected: {actual!r}. "
        "If intentional, bump this snapshot in the same commit."
    )


def test_scorers_carry_cotsuite_version_stamp() -> None:
    """Reproducibility primitive: every scorer module must import and
    reference the package version, so Score.metadata can be back-traced
    to the cotsuite release that produced it.

    Per the v0.1 reproducibility-primitives audit, the metadata
    contract for every Inspect scorer is: eval_version, cotsuite_version,
    prompt_version, prompt_sha256, plus the autorater/judge model spec.
    These five together let any consumer reproduce the eval bit-for-bit
    given the same input trajectories and grader seeds.
    """
    from cotsuite import __version__
    from cotsuite.inspect.scorers import legibility_coverage as lc
    from cotsuite.inspect.scorers import post_hoc_rationalization as phr

    assert lc._cotsuite_version == __version__
    assert phr._cotsuite_version == __version__
