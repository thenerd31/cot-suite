"""Verify the ``cotmon`` → ``cotsuite`` deprecation shim works.

Installed 2026-04-26 with the cot-monitor → cot-suite rename. Exists
to guarantee existing user imports (``from cotmon.X import Y``)
continue to work for 90 days post-rename.
"""

from __future__ import annotations

import importlib
import sys
import warnings


def test_importing_cotmon_emits_deprecation_warning() -> None:
    for cached in [k for k in sys.modules if k == "cotmon" or k.startswith("cotmon.")]:
        del sys.modules[cached]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        importlib.import_module("cotmon")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "expected DeprecationWarning on cotmon import"
        msg = str(dep_warnings[0].message)
        assert "cotmon" in msg
        assert "cotsuite" in msg


def test_cotmon_trajectory_is_same_class_as_cotsuite_trajectory() -> None:
    import cotmon.core.trajectory as cm

    import cotsuite.core.trajectory as cs

    assert cm.Trajectory is cs.Trajectory
    assert cm.Reasoning is cs.Reasoning
    assert cm.Turn is cs.Turn
    assert cm.Action is cs.Action


def test_deep_submodule_imports_via_cotmon() -> None:
    from cotmon.autoraters.legibility_coverage import LegibilityCoveragePrompt
    from cotmon.core.classify import classify
    from cotmon.core.provenance import Provenance
    from cotmon.tests.chen_cue_injection import CUE_CATALOG
    from cotmon.tests.lanham import early_answering, filler_tokens, paraphrasing

    assert LegibilityCoveragePrompt.load().sha256
    assert classify({}) == "unknown"
    assert Provenance(arxiv_id=None).is_extension()
    assert "sycophancy" in CUE_CATALOG
    assert callable(early_answering)
    assert callable(paraphrasing)
    assert callable(filler_tokens)


def test_top_level_cotmon_reexports() -> None:
    import cotmon

    assert cotmon.Trajectory.__name__ == "Trajectory"
    assert cotmon.__version__


def test_cotmon_and_cotsuite_share_registry() -> None:
    import cotmon.core.registry as cm

    import cotsuite.core.registry as cs

    assert cm.list_metrics() == cs.list_metrics()
    assert cm.list_tests() == cs.list_tests()
