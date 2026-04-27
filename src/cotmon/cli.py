"""Deprecated CLI shim. Forwards to ``cotsuite.cli:app``."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "The `cot-monitor` and `cotmon` console scripts are deprecated; "
    "use `cot-suite` (or `cotsuite`) instead. Forwarding to cotsuite.cli. "
    "Removal target: 2026-07-26.",
    DeprecationWarning,
    stacklevel=2,
)

from cotsuite.cli import app  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
