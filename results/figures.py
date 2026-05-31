"""Figure 1 — the κ-degeneracy plot, from the committed E-Z cross-judge re-analysis.

Reads ``benchmarks/results/ez_cross_judge/degeneracy_reanalysis.json`` (committed;
no model calls, $0) and renders ``results/kappa_degeneracy.png``: per axis × judge
pair, Cohen's κ beside raw observed agreement ``p_o`` and prevalence-robust Gwet
AC2. The point at a glance — on saturated **legibility**, κ collapses to ~0.2 while
p_o/AC2 stay ~0.97 (the gap is a base-rate artifact, not judge disagreement); on
**coverage** (real variance), κ ≈ AC2.

matplotlib is a figure-only dependency, NOT a cot-suite runtime dep. Run:

    uv run --with matplotlib python results/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path("benchmarks/results/ez_cross_judge/degeneracy_reanalysis.json")
OUT = Path("results/kappa_degeneracy.png")

PAIR_LABELS = {
    "haiku__sonnet-4.6": "Haiku↔Sonnet",
    "haiku__gemini-2.5-pro": "Haiku↔Gemini",
    "sonnet-4.6__gemini-2.5-pro": "Sonnet↔Gemini",
}
# κ is the coefficient that misleads (muted red); p_o + AC2 are what's actually true.
SERIES_COLOR = {
    "κ (Cohen)": "#c0392b",
    "$p_o$ (raw agreement)": "#27ae60",
    "AC2 (Gwet)": "#2980b9",
}


def _domfrac_range(axis: dict) -> str:  # type: ignore[type-arg]
    fr = list(axis["per_judge_dominant_fraction"].values())
    return f"{min(fr) * 100:.0f}–{max(fr) * 100:.0f}%"


def main() -> None:
    d = json.loads(DATA.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    titles = {"leg": "Legibility — saturated", "cov": "Coverage — real variance"}

    for ax, key in zip(axes, ("leg", "cov"), strict=True):
        a = d["axes"][key]
        pairs = list(a["pairwise_kappa"].keys())
        x = np.arange(len(pairs))
        width = 0.26
        series = [
            ("κ (Cohen)", [a["pairwise_kappa"][p] for p in pairs]),
            ("$p_o$ (raw agreement)", [a["pairwise_p_o"][p] for p in pairs]),
            ("AC2 (Gwet)", [a["pairwise_ac2"][p] for p in pairs]),
        ]
        for i, (name, vals) in enumerate(series):
            ax.bar(x + (i - 1) * width, vals, width, label=name, color=SERIES_COLOR[name])
        ax.set_xticks(x)
        ax.set_xticklabels([PAIR_LABELS[p] for p in pairs], fontsize=9)
        ax.set_title(f"{titles[key]}\n(dominant category {_domfrac_range(a)})", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Agreement coefficient")
    # Figure-level legend in a strip below the panels — clear of every bar.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=9)
    fig.suptitle(
        "Cohen's κ collapses to ~0.2 on legibility while raw agreement ($p_o$) and "
        "prevalence-robust AC2\nstay ~0.97 — the gap is a base-rate artifact, not judge "
        "disagreement; on coverage, κ ≈ AC2.",
        fontsize=9,
        y=1.04,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
