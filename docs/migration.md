# Migration from cot-divergence / cot-monitor

If you've been using cot-suite under one of its earlier names, this page covers the alias deprecation timeline.

## Distribution name aliases

| Legacy distribution | Resolves to | Aliased until | Status |
|---|---|---|---|
| `cot-divergence` | `cot-suite` | **2026-07-21** | Active alias on PyPI |
| `cot-monitor` | `cot-suite` | **2026-07-26** | Active alias on PyPI |

Both still install cleanly:

```bash
pip install cot-divergence  # aliased to cot-suite
pip install cot-monitor     # aliased to cot-suite
```

After the dates above, both aliases will stop being published; `pip install cot-suite` will be the only canonical name.

## Python import aliases

| Legacy import | New import | Removed | Behavior today |
|---|---|---|---|
| `import cotdiv` | `import cotsuite` | **2026-07-21** | Works, emits `DeprecationWarning` on first import |
| `import cotmon` | `import cotsuite` | **2026-07-26** | Works, emits `DeprecationWarning` on first import |

The `cotdiv` and `cotmon` shim packages re-export every public symbol from `cotsuite` via `sys.modules.setdefault` + `pkgutil.walk_packages`, so even submodule imports work:

```python
import cotmon.core.trajectory      # works, emits DeprecationWarning once
import cotsuite.core.trajectory    # canonical
```

After removal, replace the legacy imports with the canonical ones — `cotsuite.core.trajectory.Trajectory` is the same class as the deprecated `cotmon.core.trajectory.Trajectory` (the shim re-exports the canonical types, so user code that holds references across the boundary continues to type-check).

## CLI command aliases

| Legacy command | Canonical | Removed |
|---|---|---|
| `cotdiv` | `cot-suite` (or `cotsuite`) | **2026-07-21** |
| `cotmon` | `cot-suite` (or `cotsuite`) | **2026-07-26** |
| `cot-monitor` | `cot-suite` | **2026-07-26** |

The canonical CLI entry-points are `cot-suite` (matches the PyPI distribution name) and `cotsuite` (compact alias). Both run the same Typer app.

## Why the renames

- **2026-04-21:** `cot-divergence` → `cot-monitor`. The original name implied a focus on divergence-as-target; the actual scope is monitorability evaluation broadly.
- **2026-04-26:** `cot-monitor` → `cot-suite`. "Monitor" overloaded with the Korbak / Baker / Apollo CoT-monitor methodology; "suite" more accurately reflects the bundle scope.

Both renames kept 90-day deprecation windows. v0.1 ships with all aliases active.
