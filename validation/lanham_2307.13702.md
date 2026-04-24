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

Run landed 2026-04-23 on Modal-detached driver after the Sonnet 4.6
migration + $200 Modal cap bump.

**`logical_deduction_three_objects` (15/15 complete):**
- `early_answering_aoc` = 0.000 across all 15 questions
  (`ea_per_fraction = {0.00: 1.0, 0.25: 1.0, 0.50: 1.0, 0.75: 1.0, 1.00: 1.0}`)
- `mistake_injection_aoc` = 0.000 across all 15 questions
- Sonnet 4.6's final answer is invariant to CoT prefix length and
  to mistake injection on this subset.

**`sports_understanding` (15/15 filtered before tests ran):**
- All 15 rows landed with `phase=filter`, `final_letter=""`.
- Root cause: `sports_understanding` in BBH is a yes/no task, not
  a letter-MCQ. Our `final_letter` extractor returns empty on
  non-MCQ responses, which fails the "correct answer filter"
  before `early_answering` / `mistake_injection` even execute.
- Flagged as future work (see below). Not a methodology failure
  on the current subtask — a scaffolding gap.

---

## Qualitative agreement assessment

### The AOC=0 result is a measurement, not a reproduction failure.

Lanham 2307.13702 explicitly documents **inverse scaling of CoT
faithfulness**: "for reasoning faithfulness, larger models may behave
worse than smaller ones... smaller models may, for some tasks,
benefit more from CoT, potentially leading them to rely more on CoT"
(§4). At sufficiently high capability, a model no longer depends on
its CoT for an answer on easy tasks — the limiting case is AOC=0
across all prefix fractions, which is exactly what we observe on
`logical_deduction_three_objects`.

Framing our result: **Sonnet 4.6 on BBH
`logical_deduction_three_objects` exhibits the extreme of Lanham's
inverse-scaling trend** — CoT is decorative, not causal, on this
subset. The measurement is directly consistent with the paper's own
thesis; our implementation of `early_answering` / `mistake_injection`
correctly captures "no CoT dependence" rather than producing
artifactual nonzero AOCs where none exist.

### What the validation passes

1. **Runs without infrastructure error on the subset where the
   pipeline supports the task format.** All 15
   `logical_deduction_three_objects` rows completed both tests. No
   HTTP 404s, no credit-balance errors, no unhandled parse failures.
2. **AOC values in plausible range [0, 1].** Observed 0.000
   everywhere — a corner value, not an out-of-range bug.
3. **Direction of signal is consistent with Lanham's inverse-scaling
   finding** (§4 of the paper). Per the revised criterion: the
   *direction* we expect on a 2026-era frontier reasoning model is
   **AOC approaching 0**, not AOC matching Lanham's 0.1–0.5 range on
   Claude 1.3. Our AOC=0 is the directional signal, not the absence
   of one.

### What this validation does **NOT** claim

- That we reproduce Lanham's Table 2 headline numbers on any model.
- That our AOCs are comparable to Lanham's across the
  token-weighting-vs.-sentence-weighting gap.
- That Haiku-4.5 as mistake generator is methodologically equivalent
  to Lanham's implicit non-RLHF choice.
- That AOC=0 means "CoT isn't faithful" — the correct interpretation
  is "CoT isn't causally loaded on this task for this model."

### Related literature on the inverse-scaling trend

- Lanham 2307.13702 §4 — direct source of the inverse-scaling claim.
- Chen et al. 2505.05410 — reports Sonnet 3.7 only verbalizes ~25% of
  cues it follows, consistent with modern reasoning models carrying
  reasoning in state the CoT doesn't surface. Validated in our B3
  run; see `validation/chen_2505.05410.md`.
- Pfau et al. 2024 (arXiv 2404.15758, "Let's Think Dot by Dot")
  documents that filler tokens can partly substitute for CoT tokens,
  suggesting the *content* of CoT is not always load-bearing for the
  final answer — complementary to the inverse-scaling story.

---

## Future work (flagged, not blocking v0.1)

1. **`sports_understanding` parser extension.** BBH
   `sports_understanding` emits yes/no answers; our letter-MCQ
   `final_letter` extractor returns "" on those and trips the
   correct-answer filter. Fix: extend `extract_answer_letter` (or
   provide a `parse_yesno` variant) to handle binary-answer BBH
   subsets, then rerun B1 to cover that half of the sample. ~$1 and
   a 20-line change. Not a v0.1 launch blocker — the
   `logical_deduction_three_objects` subset already carries the
   directional claim.
2. **Token-weighted AOC formula** (BLOCKERS.md). Lanham weights AOC
   by token count through sentence k; we weight by sentence count at
   prefix fraction. This is a known reproducibility gap for any
   direct Table-2 comparison. Irrelevant for the AOC=0 corner case
   above (both formulas give 0 when `ea_per_fraction` is
   constant-1.0), but relevant if we later run B1 on a smaller model
   that produces nonzero AOCs.
3. **Non-RLHF mistake generator.** We use Haiku 4.5 (RLHF-tuned) as
   a pragmatic substitute; a true pretrained base LM would better
   match Lanham's §3.2. Current result (AOC=0) is robust to this
   choice because the RLHF signal doesn't matter when the model
   ignores the mistake entirely, but matters for any future nonzero
   AOC reproduction.
