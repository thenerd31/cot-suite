# B1 — Lanham 2307.13702 validation

**Paper:** "Measuring Faithfulness in Chain-of-Thought Reasoning" — Lanham
et al., Anthropic, 2023. arXiv: 2307.13702.

**Validation goal:** Our implementation of `early_answering` and
`mistake_injection` produces sensible AOC (area-over-the-curve) numbers
on a current reasoning model — NOT exact reproduction of Lanham's
published AOCs on Claude 1.3. Paper numbers are a sanity reference, not
an acceptance gate. See §"Qualitative agreement" below for what counts
as "sensible."

---

## Paper setup

| axis | Lanham 2023 |
|---|---|
| model | Claude 1.3 (June 2023 vintage) |
| datasets | AQuA, LogiQA, MMLU, ARC Easy, ARC Challenge, HellaSwag, OpenBookQA, TruthfulQA (8 MCQ benchmarks) |
| CoT format | native few-shot prompting; ``<response><scratchpad>...</scratchpad>...</response>`` |
| tests | early_answering, mistake_injection, paraphrasing, filler_tokens |
| sample size | 250 questions per benchmark |
| mistake generator | the same Claude 1.3 (paper §3.2 does not call out a separate base model, but the "pretrained LM" framing in the paper implies a non-RLHF model would be strictly correct) |
| AOC weighting | token count through sentence k (§3.1) |

---

## Our setup

| axis | This validation |
|---|---|
| model | **`claude-sonnet-4-6`** — current-tier Sonnet |
| dataset | BBH `logical_deduction_three_objects` + BBH `sports_understanding` (15 each, 30 total) |
| CoT format | Sonnet 4.6 native thinking-mode emit, parsed for sentences |
| tests | `early_answering` + `mistake_injection` only (B1 scope) |
| sample size | 30 questions total |
| mistake generator | `claude-haiku-4-5` (also RLHF-tuned — pragmatic substitute) |
| AOC weighting | `round(f * n)` where n is sentence count at prefix fraction |

---

## What differs

1. **Model substitution chain.** Lanham's Claude 1.3 is long retired
   (model deprecations page, Anthropic). First substitute was
   `claude-3-5-sonnet-20241022` (authored into the script on
   2026-04-21 when it was the current Sonnet); that model reached
   end-of-life and began returning HTTP 404 on 2026-04-23 (observed
   live). Second substitute is `claude-sonnet-4-6`, the currently
   supported Sonnet tier as of April 2026. Every substitution step
   changes training regime, instruction tuning, and reasoning-mode
   implementation — **absolute AOC values will NOT match Lanham's
   Table 2.**

2. **Dataset substitution.** BBH instead of the eight MCQ benchmarks
   Lanham used. BBH subsets are gathered from an existing cotmon
   fixture and bias toward logic-heavy questions where mistake
   injection produces more visible signal. Different difficulty
   distribution than Lanham's averaged-across-8 mean.

3. **AOC weighting formula.** We weight by sentence count at prefix
   fraction rather than token count through sentence k. See
   BLOCKERS.md "Length-weighted AOC alignment" — this is a known
   reproducibility gap, logged for v0.2+.

4. **Sample size.** 30 questions vs. Lanham's 250 per benchmark. Our
   budget ceiling is $5 for B1; Lanham-scale N would blow it by 8×.
   Per-question statistical power is lower; trend direction is still
   informative.

5. **Mistake generator model class.** We use Haiku 4.5 (RLHF-tuned).
   Paper implies a non-RLHF base LM should be used to avoid the RLHF
   model over-correcting its own proposed corruption. Logged in
   BLOCKERS as a gap blocking exact reproduction claims.

---

## Our numbers

**Result: pending.** Run scheduled on Modal-detached driver after the
Sonnet 4.6 migration + $200 Modal cap bump on 2026-04-23.

When B1 lands, this section will contain:
- `early_answering_aoc` mean ± SD across the 30 questions
- `mistake_injection_aoc` mean ± SD across the 30 questions
- count of rows that errored in the `gen` / `ea` / `mi` phases

---

## Qualitative agreement assessment

The validation passes if all three hold:

1. **Runs without infrastructure error.** No HTTP 404s, no credit-
   balance errors, no unhandled parse failures. ≥80% of the 30 rows
   complete both tests.
2. **AOC values are in the plausible range [0, 1].** AOC is a
   normalized curve area; out-of-range would signal a pipeline bug in
   our implementation rather than a legitimate disagreement with the
   paper.
3. **Direction of signal is consistent with Lanham's central finding:**
   `early_answering` and `mistake_injection` AOCs are both > 0 on a
   current reasoning model. Lanham's central claim is that CoT has
   *some* causal effect on the final answer — AOCs > 0 reproduce that
   directional finding even if absolute values differ.

What this validation does **NOT** claim:
- That we reproduce Lanham's Table 2 headline numbers on any model.
- That our AOCs are comparable to Lanham's across the
  token-weighting-vs.-sentence-weighting gap.
- That Haiku-4.5 as mistake generator is methodologically equivalent
  to Lanham's implicit non-RLHF choice.
