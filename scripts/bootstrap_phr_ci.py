"""Bootstrap 95% CIs on per-model PHR-strict rates from the v2 corrected JSONLs.

1000-sample bootstrap with replacement on binary PHR=True/False outcomes per
trajectory. Reports point estimates, CIs, and per-model cluster-membership
verdicts against the thinking-mode max.

Reproduction: PYTHONPATH=. python scripts/bootstrap_phr_ci.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 0
B = 1000
THINKING_MAX = 4.72  # Qwen3-Thinking-14B v2 corrected point estimate

MODELS: list[tuple[str, str, str]] = [
    ("qwen3_8b_gpqa_full", "Qwen3-Thinking-8B", "thinking"),
    ("qwen3_14b_gpqa_full", "Qwen3-Thinking-14B", "thinking"),
    ("qwen3_32b_gpqa_full", "Qwen3-Thinking-32B", "thinking"),
    ("ds_r1_distill_qwen_14b_full", "DS-R1-Distill-Qwen-14B", "thinking"),
    ("ds_r1_distill_llama_70b_full", "DS-R1-Distill-Llama-70B", "thinking"),
    ("qwen25_7b_instruct_full", "Qwen2.5-7B-Instruct", "non-thinking"),
    ("qwen25_72b_instruct_full", "Qwen2.5-72B-Instruct", "non-thinking"),
    ("llama31_8b_instruct_full", "Llama-3.1-8B-Instruct", "non-thinking"),
]


def load_outcomes(model_dir: str) -> list[int]:
    """Read v2 PHR judgments and return the binary strict-PHR outcomes for the correct subset."""
    p = Path(f"benchmarks/results/{model_dir}/post_hoc_rationalization_v2.jsonl")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    correct = [r for r in rows if r.get("is_correct")]
    return [
        1 if (r.get("diverged") and not r.get("acknowledged")) else 0
        for r in correct
    ]


def bootstrap_ci(outcomes: list[int], b: int = B, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point, ci_low, ci_high) — point estimate plus (1-alpha) bootstrap CI."""
    n = len(outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = sum(outcomes) / n
    samples = sorted(
        sum(random.choice(outcomes) for _ in range(n)) / n
        for _ in range(b)
    )
    return point, samples[int(b * alpha / 2)], samples[int(b * (1 - alpha / 2))]


def main() -> None:
    random.seed(SEED)
    print(
        f"{'model':<28} {'mode':<13} {'n':>4} {'phr':>4} "
        f"{'pt_est':>7} {'CI_low':>7} {'CI_high':>7}"
    )
    print("-" * 80)
    nonthinking_results = []
    for d, name, mode in MODELS:
        outcomes = load_outcomes(d)
        n, phr = len(outcomes), sum(outcomes)
        point, lo, hi = bootstrap_ci(outcomes)
        print(
            f"{name:<28} {mode:<13} {n:>4} {phr:>4} "
            f"{100 * point:>6.2f}% {100 * lo:>6.2f}% {100 * hi:>6.2f}%"
        )
        if mode == "non-thinking":
            nonthinking_results.append((name, n, point, lo, hi))

    print()
    print("Cluster-membership verdicts (does CI lower bound clear thinking-max=4.72%?):")
    for name, n, point, lo, hi in nonthinking_results:
        margin = 100 * lo - THINKING_MAX
        if margin > 2:
            verdict = "ROBUST"
        elif margin > 0:
            verdict = "NEAR-BOUNDARY"
        else:
            verdict = "NOT-ROBUST"
        print(
            f"  {name}: n={n}, point={100 * point:.2f}%, "
            f"CI_lo={100 * lo:.2f}%, margin {margin:+.2f}pp → {verdict}"
        )


if __name__ == "__main__":
    main()
