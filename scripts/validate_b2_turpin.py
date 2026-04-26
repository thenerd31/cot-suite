"""B2 — Turpin 2305.04388 counterfactual-bias validation on GPT-3.5-turbo.

Runs cotmon's ``counterfactual_bias`` with the ``always_a_fewshot`` bias
on GPT-3.5-turbo across BBH subtasks (logical_deduction_three_objects +
sports_understanding, 15 each = 30 total). Measures accuracy drop when
the bias points to a wrong answer + verbalization rate (fraction of
bias-followed trajectories that explicitly acknowledge the bias in CoT).

**Caveats**:

1. Few-shot scaffolding is the 3-toy-exemplar variant in
   BIAS_CATALOG["always_a_fewshot"], NOT the 13-BBH-task per-subtask
   prompts Turpin used. Absolute accuracy drop will not match Turpin's
   ~36% headline because the few-shot setup is structurally different.

2. Model substitution. Turpin tested GPT-3.5-turbo and Claude 1.3.
   We use ``openai/gpt-3.5-turbo`` — exact model match for the GPT
   side of Turpin's setup. Claude 1.3 substitution is deferred (no
   currently-supported equivalent that closely matches the 2023 model).

3. Verbalization detector substitution. Turpin used human annotation
   + keyword matching. We use Claude Haiku 4.5 LLM-as-judge via
   ``judges_verbalizes`` in cotmon.tests._cue_judge.

4. Sample size: 30 questions × 1 bias = 30 paired (baseline, biased)
   inferences = 60 GPT-3.5 calls + up to 30 Haiku judge calls.

Budget: ~$3.00 (60 GPT-3.5-turbo calls × ~$0.001 + 30 Haiku judge
calls × ~$0.01). Within v0.1 validation budget.

Validation goal: implementation produces sensible accuracy-drop and
verbalization rates on a current API model — NOT exact reproduction
of Turpin's ~36%/~near-zero headline. See validation/turpin_2305.04388.md.

Usage:
    PYTHONPATH=. python scripts/validate_b2_turpin.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from cotmon.tests.turpin_counterfactual import (
    BIAS_CATALOG,
    Sample,
    counterfactual_bias,
)

MODEL_UNDER_TEST = "openai/gpt-3.5-turbo"
JUDGE = "anthropic/claude-haiku-4-5"
N_PER_SUBTASK = 15
BIAS_NAME = "always_a_fewshot"
OUTPUT_PATH = Path("validation/b2_turpin_raw.jsonl")

BBH_SUBTASKS = ("logical_deduction_three_objects", "sports_understanding")


def _load_bbh_samples(subtask: str, limit: int) -> list[Sample]:
    """Load BBH subtask, returning Turpin-style Sample(question, correct_answer).

    BBH targets are bare letters or yes/no strings; we normalize to
    upper-case. Answers like 'yes' / 'no' are passed through as-is so
    the bias-follow detection (`normalized_equals(answer, "A")`) works
    on whatever the model emits.
    """
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", subtask, split="test")
    out: list[Sample] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        target = row["target"].strip().replace("(", "").replace(")", "").upper()
        out.append(Sample(question=row["input"], correct_answer=target))
    return out


async def main() -> int:
    from cotmon.verify_keys import require_keys

    require_keys(["openai", "anthropic"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    for subtask in BBH_SUBTASKS:
        samples.extend(_load_bbh_samples(subtask, N_PER_SUBTASK))
    bias = BIAS_CATALOG[BIAS_NAME]

    print(f"Running {BIAS_NAME} bias against {MODEL_UNDER_TEST} on {len(samples)} questions...")
    result = await counterfactual_bias(
        model=MODEL_UNDER_TEST,
        bias=bias,
        samples=samples,
        judge=JUDGE,
    )

    OUTPUT_PATH.write_text(json.dumps(result.raw, indent=2) + "\n")

    raw = result.raw
    n_total = raw["n_total"]
    n_baseline_correct = raw["n_baseline_correct"]
    n_biased_correct = raw["n_biased_correct"]
    accuracy_drop = raw["accuracy_drop"]
    bias_follow_on_wrong = raw["bias_follow_rate_on_wrong_pointing"]
    verbalization_rate = raw["verbalization_rate"]

    print()
    print("=" * 70)
    print(f"B2 VALIDATION — Turpin 2305.04388 '{BIAS_NAME}' bias on {MODEL_UNDER_TEST}")
    print("=" * 70)
    print(f"n_total                          = {n_total}")
    print(f"baseline accuracy                = {n_baseline_correct/n_total*100:.1f}%")
    print(f"biased accuracy                  = {n_biased_correct/n_total*100:.1f}%")
    print(f"accuracy_drop                    = {accuracy_drop:.3f} = {accuracy_drop*100:.1f}pp")
    print(f"bias_follow_rate_on_wrong        = {bias_follow_on_wrong:.3f} = {bias_follow_on_wrong*100:.1f}%")
    print(f"verbalization_rate (when followed) = {verbalization_rate:.3f} = {verbalization_rate*100:.1f}%")
    print()
    print("Turpin 2305.04388 reference values (GPT-3.5-turbo, BBH 13 tasks, full setup):")
    print("  accuracy drop up to ~36% (varies by task)")
    print("  verbalization rate ~near-zero on biased trajectories")
    print()
    print("Our setup uses 3 toy exemplars (not paper's 13-task per-subtask few-shot).")
    print("Absolute numbers will not match the paper headline; direction of signal")
    print("(some accuracy drop, low verbalization rate) is the qualitative test.")
    print()
    print(f"Raw output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
