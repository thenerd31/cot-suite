"""``cotmon`` — deprecated alias for ``cotsuite``.

The package was renamed from ``cot-monitor`` to ``cot-suite`` on
2026-04-26 (PyPI distribution + Python import). The library
ships five bundled evaluations (Lanham early-answering /
mistake-injection / paraphrasing / filler_tokens, Turpin
counterfactual, Chen cue-injection, Arcuschin PHR, Emmons &
Zimmermann autorater) — ``cot-suite`` is more accurate to that
shape than ``cot-monitor`` was. The earlier rename history:
``cot-divergence`` (initial) → ``cot-monitor`` (2026-04-21) →
``cot-suite`` (2026-04-26).

This shim aliases the new ``cotsuite`` package tree into the old
``cotmon`` namespace so existing imports keep working. Plan: remove
this shim 2026-07-26 (90 days from rename).

Any ``from cotmon.X import Y`` resolves to ``cotsuite.X.Y`` via
``sys.modules`` aliasing. Triggering this shim emits a single
``DeprecationWarning`` per process; update imports to silence.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

import cotsuite

warnings.warn(
    "`cotmon` has been renamed to `cotsuite`. The old name still works "
    "but will be removed after 2026-07-26. Migrate `from cotmon...` → "
    "`from cotsuite...` to silence this warning.",
    DeprecationWarning,
    stacklevel=2,
)


def _install_cotmon_aliases() -> None:
    """Register every ``cotsuite.*`` submodule under ``cotmon.*`` in sys.modules.

    Eager import so that ``from cotmon.core.trajectory import Trajectory``
    resolves — Python looks up the full dotted path in ``sys.modules``
    before checking disk, and we need to pre-populate it.
    """
    for _finder, name, _ispkg in pkgutil.walk_packages(
        cotsuite.__path__,
        prefix="cotsuite.",
    ):
        importlib.import_module(name)

    for name, mod in list(sys.modules.items()):
        if name == "cotsuite" or name.startswith("cotsuite."):
            alias = "cotmon" + name[len("cotsuite") :]
            sys.modules.setdefault(alias, mod)


_install_cotmon_aliases()

# Re-export top-level names so ``from cotmon import Trajectory`` still works.
from cotsuite import (  # noqa: E402
    Action,
    Reasoning,
    Trajectory,
    Turn,
    __version__,
    register_metric,
    register_test,
    score_trajectory,
)

__all__ = [
    "Action",
    "Reasoning",
    "Trajectory",
    "Turn",
    "__version__",
    "register_metric",
    "register_test",
    "score_trajectory",
]
