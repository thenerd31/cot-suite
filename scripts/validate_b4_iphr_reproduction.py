"""B4a — Arcuschin 2503.08679 IPHR against-release reproduction (integer-exact).

Runs cot-suite's INDEPENDENT IPHR metric (``cotsuite.tests.iphr``) on ChainScope's
released per-question dataframe and reproduces the paper's per-model IPHR
unfaithful-pair counts — integer-exact (±0) — for the 9 non-oversampled models.
This is a $0 metric-replay: pure parse + aggregate of the vendored CSV, no LLM
calls, no API calls.

Verification is two-target per model:
  (a) cot-suite's flagged count == ChainScope's own computed count
      (``notebooks/plots_for_writeup.py:813`` Fig-2 numerators), and
  (b) the resulting rate (count / 4892) == the pinned Fig-2 cell, which equals the
      paper abstract's headline for the 4 headline models.

If ANY of the 9 models is not ±0, the script exits non-zero (HALT) — that would
mean cot-suite's criteria port is wrong on a case we claim is correct.

Denominator note (a finding about the paper, not a choice we hide): ChainScope's
Fig 2 uses ``n_pairs = 4892``; the paper TEXT says 4,834 (= 9668 questions / 2).
We reproduce the FIGURE (4892), which is what produced the published rates, and
surface the inconsistency.

The 7 oversampled models (claude-3.6-sonnet, claude-3.7-sonnet, claude-3.7-sonnet_64k,
deepseek-r1, gemini-2.5-pro-preview, chatgpt-4o-latest, gpt-4o-2024-08-06) are
NOT reproduced here: their flagging required an adaptive two-pass oversampling
(``--unfaithful-only -n 100``) whose pass-1 candidate-selection state is absent
from the released aggregate df. See ``src/cotsuite/tests/iphr.py`` + ``AUDIT.md``.

Inputs (vendored — see ``validation/chainscope_iphr/PROVENANCE.md``):
  - ``validation/chainscope_iphr/df_wm_non_ambiguous_hard2_9models.csv.gz``
    (subset of ChainScope ``df-wm-non-ambiguous-hard-2.pkl.gz`` @ bb128ac0)

Output: ``validation/b4_iphr_reproduction_results.json``

Usage:
    PYTHONPATH=. .venv/bin/python scripts/validate_b4_iphr_reproduction.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from cotsuite.tests.iphr import (
    CHAINSCOPE_N_PAIRS,
    count_unfaithful_pairs,
    iphr_questions_from_rows,
    iphr_unfaithful_rate,
)

CHAINSCOPE_COMMIT = "bb128ac0f22dd60ab3d876a78e1c6c22fff7e830"
N_PAIRS = CHAINSCOPE_N_PAIRS  # 4892 (figure); paper text says 4834
N_PAIRS_PAPER_TEXT = 4834

VENDORED_CSV = Path("validation/chainscope_iphr/df_wm_non_ambiguous_hard2_9models.csv.gz")
VENDORED_CSV_SHA256 = "ad7fd888c6710a34a91a8e17bd1597e9bc1d2569cf7b7bd087794b26f5de7c1d"
OUTPUT_PATH = Path("validation/b4_iphr_reproduction_results.json")

# ChainScope's own computed unfaithful-pair counts (plots_for_writeup.py:813),
# keyed by df model_id. Verification target (a). These are the Fig-2 numerators.
CHAINSCOPE_DF_COUNTS: dict[str, int] = {
    "openai/gpt-4o-mini": 660,
    "anthropic/claude-3.5-haiku": 363,
    "google/gemini-pro-1.5": 320,
    "qwen/qwq-32b": 220,
    "meta-llama/Llama-3.1-70B": 159,
    "google/gemini-2.5-flash-preview": 106,
    "meta-llama/Llama-3.3-70B-Instruct": 102,
    "deepseek/deepseek-chat": 60,
    "anthropic/claude-3.7-sonnet_1k": 2,
}

# Paper abstract headline cells (the 4 of the 9 that are headline figures).
# Verification target (b): count/4892 must round to these.
PAPER_HEADLINE_PCT: dict[str, str] = {
    "openai/gpt-4o-mini": "13%",
    "anthropic/claude-3.5-haiku": "7%",
    "google/gemini-2.5-flash-preview": "2.17%",
    "anthropic/claude-3.7-sonnet_1k": "0.04%",  # Sonnet-3.7 w/ thinking (1k)
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not VENDORED_CSV.exists():
        print(f"ERROR: vendored data not found: {VENDORED_CSV}", file=sys.stderr)
        return 2

    actual_sha = _sha256(VENDORED_CSV)
    if actual_sha != VENDORED_CSV_SHA256:
        print(
            f"WARNING: vendored CSV SHA-256 drift: actual={actual_sha[:12]} "
            f"expected={VENDORED_CSV_SHA256[:12]} — provenance broken.",
            file=sys.stderr,
        )

    df = pd.read_csv(VENDORED_CSV)

    rows: list[dict] = []
    all_exact = True
    for model_id, expected in CHAINSCOPE_DF_COUNTS.items():
        model_df = df[df.model_id == model_id]
        questions = iphr_questions_from_rows(model_df.to_dict("records"))
        count = count_unfaithful_pairs(questions)
        rate = iphr_unfaithful_rate(count, N_PAIRS)
        delta = count - expected
        exact = delta == 0
        all_exact = all_exact and exact
        headline = PAPER_HEADLINE_PCT.get(model_id)
        rows.append(
            {
                "model_id": model_id,
                "model_dir": model_id.split("/")[-1],
                "cotsuite_count": count,
                "chainscope_df_count": expected,
                "delta": delta,
                "exact": exact,
                "rate_pct": round(rate, 2),
                "n_pairs": N_PAIRS,
                "paper_headline": headline,
            }
        )

    # Report
    print("=" * 78)
    print("B4a — Arcuschin 2503.08679 IPHR against-release reproduction (metric-replay)")
    print("=" * 78)
    print(f"ChainScope @ {CHAINSCOPE_COMMIT[:12]} | n_pairs={N_PAIRS} (figure)  "
          f"[paper text says {N_PAIRS_PAPER_TEXT}]")
    print(f"data: {VENDORED_CSV} (sha {actual_sha[:12]})  | $0 — no LLM calls")
    print("-" * 78)
    print(f"{'model':<28}{'cot-suite':>10}{'chainscope':>11}{'Δ':>5}{'rate':>9}  headline")
    print("-" * 78)
    for r in rows:
        hl = r["paper_headline"] or ""
        print(
            f"{r['model_dir']:<28}{r['cotsuite_count']:>10}{r['chainscope_df_count']:>11}"
            f"{r['delta']:>+5}{r['rate_pct']:>8.2f}%  {hl}"
        )
    print("-" * 78)
    n_headline = sum(1 for r in rows if r["paper_headline"])
    print(f"{'INTEGER-EXACT (±0): ALL 9 MODELS' if all_exact else '*** MISMATCH — HALT ***'}"
          f"  | {n_headline} of 7 paper-headline cells reproduced")
    print("blocked (oversampled, not reconstructable): claude-3.6-sonnet, claude-3.7-sonnet, "
          "claude-3.7-sonnet_64k, deepseek-r1, gemini-2.5-pro-preview, chatgpt-4o-latest, "
          "gpt-4o-2024-08-06")

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "reproduction": "B4a Arcuschin IPHR against-release (integer-exact)",
                "chainscope_commit": CHAINSCOPE_COMMIT,
                "n_pairs_figure": N_PAIRS,
                "n_pairs_paper_text": N_PAIRS_PAPER_TEXT,
                "vendored_csv": str(VENDORED_CSV),
                "vendored_csv_sha256": actual_sha,
                "all_integer_exact": all_exact,
                "models": rows,
                "blocked_models_oversampled": [
                    "claude-3.6-sonnet", "claude-3.7-sonnet", "claude-3.7-sonnet_64k",
                    "deepseek-r1", "gemini-2.5-pro-preview", "chatgpt-4o-latest",
                    "gpt-4o-2024-08-06",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {OUTPUT_PATH}")

    if not all_exact:
        print("HALT: at least one of the 9 models is not ±0 — criteria port is wrong.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
