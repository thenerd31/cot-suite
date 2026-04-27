"""``cotdiv`` — deprecated alias for ``cotsuite``.

The package was renamed twice:
  - ``cot-divergence`` → ``cot-monitor`` on 2026-04-21
  - ``cot-monitor``    → ``cot-suite``   on 2026-04-26

This shim aliases the current ``cotsuite`` package tree into the
oldest ``cotdiv`` namespace, skipping the intermediate ``cotmon``
shim to avoid double-warning chains. Plan: remove this shim
2026-07-21 (90 days from the original ``cot-divergence`` →
``cot-monitor`` rename).

Any ``from cotdiv.X import Y`` resolves to ``cotsuite.X.Y`` via
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
    "`cotdiv` has been renamed (twice) to `cotsuite`. The old name "
    "still works but will be removed after 2026-07-21. Migrate "
    "`from cotdiv...` → `from cotsuite...` to silence this warning.",
    DeprecationWarning,
    stacklevel=2,
)


def _install_cotdiv_aliases() -> None:
    """Register every ``cotsuite.*`` submodule under ``cotdiv.*`` in sys.modules.

    Eager import so that ``from cotdiv.core.trajectory import Trajectory``
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
            alias = "cotdiv" + name[len("cotsuite") :]
            sys.modules.setdefault(alias, mod)


_install_cotdiv_aliases()

# Re-export top-level names so ``from cotdiv import Trajectory`` still works.
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
