"""Verify the ``cotdiv`` → ``cotsuite`` deprecation shim works.

Original install: 2026-04-21 (cot-divergence → cot-monitor rename).
Updated 2026-04-26 to skip the intermediate cotmon shim and alias
directly to cotsuite. Exists to guarantee existing user imports
(``from cotdiv.X import Y``) continue to work for 90 days post-rename.
"""

from __future__ import annotations

import importlib
import sys
import warnings


def _fresh_import(name: str):
    """Re-import a module after stripping any cached copy from sys.modules."""
    for cached in [k for k in sys.modules if k == name or k.startswith(f"{name}.")]:
        del sys.modules[cached]
    return importlib.import_module(name)


def test_importing_cotdiv_emits_deprecation_warning() -> None:
    # Strip any cached cotdiv so the warning fires fresh.
    for cached in [k for k in sys.modules if k == "cotdiv" or k.startswith("cotdiv.")]:
        del sys.modules[cached]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("cotdiv")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning on cotdiv import"
        msg = str(dep_warnings[0].message)
        assert "cotdiv" in msg
        assert "cotsuite" in msg


def test_cotdiv_trajectory_is_same_class_as_cotsuite_trajectory() -> None:
    import cotdiv.core.trajectory as cd

    import cotsuite.core.trajectory as cs

    assert cd.Trajectory is cs.Trajectory
    assert cd.Reasoning is cs.Reasoning
    assert cd.Turn is cs.Turn
    assert cd.Action is cs.Action


def test_deep_submodule_imports_via_cotdiv() -> None:
    # Exercise a few deep paths that downstream users might hit.
    from cotdiv.autoraters.legibility_coverage import LegibilityCoveragePrompt
    from cotdiv.core.classify import classify
    from cotdiv.core.provenance import Provenance
    from cotdiv.tests.chen_cue_injection import CUE_CATALOG
    from cotdiv.tests.lanham import early_answering, filler_tokens, paraphrasing

    assert LegibilityCoveragePrompt.load().sha256  # smoke
    assert classify({}) == "unknown"
    assert Provenance(arxiv_id=None).is_extension()
    assert "sycophancy" in CUE_CATALOG
    assert callable(early_answering)
    assert callable(paraphrasing)
    assert callable(filler_tokens)


def test_top_level_cotdiv_reexports() -> None:
    import cotdiv

    assert cotdiv.Trajectory.__name__ == "Trajectory"
    assert cotdiv.__version__


def test_cotdiv_and_cotsuite_share_registry() -> None:
    # A metric registered via one path should be visible via the other —
    # they're the same module object after aliasing.
    import cotdiv.core.registry as cd

    import cotsuite.core.registry as cs

    assert cd.list_metrics() == cs.list_metrics()
    assert cd.list_tests() == cs.list_tests()
