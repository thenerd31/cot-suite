"""scripts/resync_gmean.py drift detection — pure-function tests, no network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_ROOT = Path(__file__).resolve().parents[1]
_RESYNC = _ROOT / "scripts" / "resync_gmean.py"
_VENDORED = _ROOT / "src" / "cotsuite" / "metrics" / "gmean.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resync_gmean", _RESYNC)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["resync_gmean"] = mod  # register before exec (safe module loading)
    spec.loader.exec_module(mod)
    return mod


def test_strip_header_removes_vendoring_block() -> None:
    mod = _load()
    text = (
        "# Vendored from x at commit abc.\n"
        "# See LICENSES/...\n"
        "# Resync via scripts/resync_gmean.py.\n"
        "from __future__ import annotations\nimport os\n"
    )
    body = mod.strip_header(text)
    assert body.startswith("from __future__ import annotations")
    assert "# Vendored" not in body


def test_parse_pinned_sha() -> None:
    mod = _load()
    sha = "806dabf3ecc4e4ed466b6e26c1689e1e5e404c92"
    assert mod.parse_pinned_sha(f"# Vendored from x at commit {sha}.\n") == sha
    assert mod.parse_pinned_sha("no sha here") is None


def test_detect_drift_none_when_identical() -> None:
    mod = _load()
    drifted, diff = mod.detect_drift("from __future__\nx = 1\n", "from __future__\nx = 1\n")
    assert drifted is False
    assert diff == ""


def test_detect_drift_flags_modification() -> None:
    mod = _load()
    drifted, diff = mod.detect_drift("from __future__\nx = 1\n", "from __future__\nx = 2\n")
    assert drifted is True
    assert "x = 1" in diff and "x = 2" in diff


def test_drift_on_real_vendored_file_fixture() -> None:
    # The committed vendored body vs itself: no drift. A one-token change to a
    # fixture copy IS flagged — the plan's "modify the vendored file" check.
    mod = _load()
    body = mod.strip_header(_VENDORED.read_text())
    assert mod.detect_drift(body, body)[0] is False
    mutated = body.replace("BootstrapConfig", "BootstrapConfigX", 1)
    drifted, diff = mod.detect_drift(body, mutated)
    assert drifted is True
    assert "BootstrapConfigX" in diff
