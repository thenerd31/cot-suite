"""B3 — Chen 2505.05410 cue-injection validation on Claude Sonnet 3.7.

Runs cotmon's ``chen_cue_injection`` with the ``metadata`` cue (cleanest
verbatim-from-paper template of the six) on Claude Sonnet 3.7 across 50
GPQA-Diamond questions. Measures cue uptake rate (fraction where the
model's answer matches the injected cue target) and verbalization rate
(fraction of cue-following trajectories that acknowledge the cue in CoT).

**Caveats**:

1. Dataset substitution. Chen 2505.05410 tested on MMLU and BBH. We use
   GPQA-Diamond because our pipeline loads it cleanly and the Stage 1/2
   runs already normalize on it. Different difficulty distribution; cue
   uptake rates may differ.

2. Model substitution. Chen reports 25% overall verbalization for
   Claude 3.7 Sonnet. We use ``claude-3-7-sonnet-20250219`` with thinking
   enabled, which matches.

3. Cue-type substitution. Chen reports verbalization rates aggregated
   across six cues. We run only the ``metadata`` cue (cleanest paper
   template + easiest to verify the injected target landed). Single-cue
   uptake may differ from the pooled mean.

4. Sample size: 50 questions × 1 cue = 50 autorater calls. Tight for
   stable mean ± SD.

Budget: ~$2.25 (50 Sonnet-3.7-thinking calls + 50 Haiku judge calls).
Within the $5 B3 ceiling.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

from cotmon.tests.chen_cue_injection import CUE_CATALOG, InjectionSample, cue_injection

MODEL_UNDER_TEST = "anthropic/claude-3-7-sonnet-20250219"
JUDGE = "anthropic/claude-haiku-4-5"
N_QUESTIONS = 50
CUE_NAME = "metadata"
OUTPUT_PATH = Path("validation/b3_chen_raw.jsonl")


def _load_samples(limit: int) -> list[InjectionSample]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    samples: list[InjectionSample] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        correct = row["Correct Answer"]
        incorrect = [row[f"Incorrect Answer {j}"] for j in (1, 2, 3)]
        options = [correct, *incorrect]
        rng = random.Random(i)
        rng.shuffle(options)
        letter_to_text = dict(zip("ABCD", options, strict=True))
        correct_letter = next(letter for letter, text in letter_to_text.items() if text == correct)
        # Target a wrong answer — pick the first non-correct letter for determinism.
        target_letter = next(letter for letter in "ABCD" if letter != correct_letter)
        question_full = row["Question"] + "\n\n"
        question_full += "\n".join(f"({letter}) {letter_to_text[letter]}" for letter in "ABCD")
        samples.append(
            InjectionSample(
                question=question_full,
                correct_answer=correct_letter,
                target_answer=target_letter,
            ),
        )
    return samples


async def main() -> int:
    from cotmon.verify_keys import require_keys
    require_keys(["anthropic"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(N_QUESTIONS)
    cue = CUE_CATALOG[CUE_NAME]

    print(f"Running {CUE_NAME} cue against {MODEL_UNDER_TEST} on {len(samples)} questions...")
    result = await cue_injection(
        model=MODEL_UNDER_TEST,
        cue=cue,
        samples=samples,
        judge=JUDGE,
    )

    OUTPUT_PATH.write_text(json.dumps(result.raw, indent=2) + "\n")

    follow_rate = result.raw["follow_rate"]
    follow_rate_on_wrong = result.raw["follow_rate_on_wrong"]
    verbalization_rate = result.raw["verbalization_rate"]
    n_total = result.raw["n_total"]
    n_cue_followed = result.raw["n_cue_followed"]

    print()
    print("=" * 70)
    print(f"B3 VALIDATION — Chen 2505.05410 '{CUE_NAME}' cue on Claude Sonnet 3.7")
    print("=" * 70)
    print(f"n_total                     = {n_total}")
    print(f"n_cue_followed              = {n_cue_followed}")
    print(f"cue_follow_rate             = {follow_rate:.3f} = {follow_rate * 100:.1f}%")
    print(f"cue_follow_rate_on_wrong    = {follow_rate_on_wrong:.3f} = {follow_rate_on_wrong * 100:.1f}%")
    print(f"verbalization_rate          = {verbalization_rate:.3f} = {verbalization_rate * 100:.1f}%")
    print()
    print("Chen 2505.05410 Fig 1 reference values:")
    print("  Claude 3.7 Sonnet (Extended Thinking): 25% overall verbalization (20% misaligned)")
    print("  (metadata is a NEUTRAL cue, not Misaligned; paper may report its rate differently)")
    print()
    print(f"Raw output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
