"""Drift detection for the vendored g-mean² metric.

Compares ``src/cotsuite/metrics/gmean.py`` (its body, below the cot-suite
vendoring header) against the current upstream
``openai/monitorability-evals/main/metric/intervention_gmean_metric.py``.

Usage:
    python scripts/resync_gmean.py                  # read-only; prints diff, exits 1 on drift
    python scripts/resync_gmean.py --apply-upstream  # prompts, then overwrites + re-pins the SHA
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import urllib.request
from pathlib import Path

REPO = "openai/monitorability-evals"
UPSTREAM_PATH = "metric/intervention_gmean_metric.py"
VENDORED = Path(__file__).resolve().parents[1] / "src" / "cotsuite" / "metrics" / "gmean.py"
_BODY_SENTINEL = "from __future__ import annotations"
_SHA_RE = re.compile(r"at commit ([0-9a-f]{7,40})")


def strip_header(vendored_text: str) -> str:
    """Return the upstream body — everything from the first __future__ import on."""
    idx = vendored_text.find(_BODY_SENTINEL)
    if idx < 0:
        raise ValueError(f"sentinel {_BODY_SENTINEL!r} not found in vendored file")
    return vendored_text[idx:]


def parse_pinned_sha(vendored_text: str) -> str | None:
    """Return the SHA pinned in the vendoring header, or None if absent."""
    match = _SHA_RE.search(vendored_text)
    return match.group(1) if match else None


def detect_drift(vendored_body: str, upstream_text: str) -> tuple[bool, str]:
    """Return (drifted, unified_diff) comparing the vendored body to upstream."""
    if vendored_body == upstream_text:
        return False, ""
    diff = "".join(
        difflib.unified_diff(
            vendored_body.splitlines(keepends=True),
            upstream_text.splitlines(keepends=True),
            fromfile="vendored (gmean.py body)",
            tofile=f"upstream ({REPO}/main)",
        )
    )
    return True, diff


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def fetch_upstream(ref: str = "main") -> str:
    """Fetch the raw upstream metric file at the given ref."""
    return _fetch(f"https://raw.githubusercontent.com/{REPO}/{ref}/{UPSTREAM_PATH}")


def fetch_main_sha() -> str:
    """Fetch the current HEAD SHA of the upstream main branch."""
    raw = _fetch(f"https://api.github.com/repos/{REPO}/commits/main")
    return str(json.loads(raw)["sha"])


def _header(sha: str) -> str:
    return (
        f"# Vendored from {REPO} at commit {sha}.\n"
        "# See LICENSES/Apache-2.0-monitorability-evals.txt.\n"
        "# Resync via scripts/resync_gmean.py.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect/resync drift in vendored gmean.py")
    parser.add_argument(
        "--apply-upstream",
        action="store_true",
        help="overwrite the vendored file with upstream main and re-pin the SHA (prompts first)",
    )
    args = parser.parse_args(argv)

    vendored_text = VENDORED.read_text()
    pinned = parse_pinned_sha(vendored_text)
    body = strip_header(vendored_text)
    upstream = fetch_upstream("main")
    drifted, diff = detect_drift(body, upstream)
    main_sha = fetch_main_sha()

    print(f"pinned SHA:  {pinned}")
    print(f"main SHA:    {main_sha}")
    if not drifted:
        print("No drift: vendored body matches upstream main.")
        return 0

    print("DRIFT DETECTED:\n")
    print(diff)
    if not args.apply_upstream:
        print("\nRe-run with --apply-upstream to update (will prompt).")
        return 1

    reply = input(f"\nOverwrite gmean.py with upstream main ({main_sha})? [y/N] ").strip().lower()
    if reply != "y":
        print("Aborted; no changes written.")
        return 1
    VENDORED.write_text(_header(main_sha) + upstream)
    print(f"Updated. Re-pinned to {main_sha}. Review the diff and re-run the test suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
