"""``cotdiv`` — deprecated alias for ``cotmon``.

The package was renamed from ``cot-divergence`` to ``cot-monitor`` on
2026-04-21 to reflect the broader positioning around monitorability +
faithfulness (not faithfulness alone). This shim module aliases the new
``cotmon`` package tree into the old ``cotdiv`` namespace so existing
imports continue to work. Plan: remove this shim 2026-07-21 (90 days).

Any ``from cotdiv.X import Y`` resolves to ``cotmon.X.Y`` via
:mod:`sys.modules` aliasing. Triggering this shim emits a single
``DeprecationWarning`` per process; update imports to silence.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

import cotmon

warnings.warn(
    "`cotdiv` has been renamed to `cotmon`. The old name still works "
    "but will be removed after 2026-07-21. Migrate `from cotdiv...` → "
    "`from cotmon...` to silence this warning.",
    DeprecationWarning,
    stacklevel=2,
)


def _install_cotdiv_aliases() -> None:
    """Register every ``cotmon.*`` submodule under ``cotdiv.*`` in sys.modules.

    Eager import is required so that ``from cotdiv.core.trajectory import
    Trajectory`` resolves — Python looks up the full dotted path in
    ``sys.modules`` before checking disk, and we need to pre-populate it.
    """
    # First, ensure cotmon's own submodules are all importable.
    for _finder, name, _ispkg in pkgutil.walk_packages(
        cotmon.__path__,
        prefix="cotmon.",
    ):
        importlib.import_module(name)

    # Now alias every cotmon.* module under cotdiv.*
    for name, mod in list(sys.modules.items()):
        if name == "cotmon" or name.startswith("cotmon."):
            alias = "cotdiv" + name[len("cotmon") :]
            sys.modules.setdefault(alias, mod)


_install_cotdiv_aliases()

# Re-export top-level names so ``from cotdiv import Trajectory`` still works.
from cotmon import (  # noqa: E402
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
