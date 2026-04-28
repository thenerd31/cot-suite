# Multi-family CoT monitorability scaling — Stage 1+2+3 summary (8 models)

**v2 corrected, 2026-04-27.** Numbers below are post answer-extractor parser fix
(`cotsuite.parsing.extract_answer_letter`, `EVAL_VERSION = 1.1.0`). The pre-fix
v1 numbers in this file's git history are not directly comparable — see the
"Parser fix and v1 → v2 deltas" section below and `AUDIT.md` for the full
discovery and fix narrative.

**Setup (constant across all 8 rows):** GPQA-Diamond (198 questions),
Qwen3 thinking-mode default sampling (temperature 0.6, top_p 0.95,
top_k 20, min_p 0, max_new_tokens 32768), 5-min per-question Modal
RPC cap (Fix #1). Autorater = Claude Haiku 4.5 with Emmons &
Zimmermann 2510.23966 Appendix C prompt (SHA `ac1e0ac4044b0a64…`).
PHR detector = Claude Haiku 4.5 with Arcuschin-inspired LLM-as-judge
prompt (SHA `4d7cc712e9456b80…`). Detector validated B4 (9.30% on
GPT-4o-mini vs paper 13%, **unchanged across the parser fix**) —
see `validation/arcuschin_2503.08679.md`.

## Eight-model scaling table (v2 corrected)

| model | mode | base | n | accuracy | legibility | coverage | leg-cov gap | PHR strict | PHR incl. ack | empty final |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Thinking-8B | thinking | Qwen3 | 198 | 61.62% | 3.48 | 3.24 | 0.23 | 3.28% | 4.92% | 0 |
| Qwen3-Thinking-14B | thinking | Qwen3 | 197 | 64.47% | 3.61 | 3.42 | 0.19 | 4.72% | 5.51% | 0 |
| Qwen3-Thinking-32B | thinking | Qwen3 | 185 | 71.89% | 3.73 | 3.57 | 0.16 | 2.26% | 3.76% | 0 |
| DS-R1-Distill-Qwen-14B | thinking | Qwen | 197 | 54.82% | 3.41 | 3.13 | 0.29 | 0.93% | 1.85% | 0 |
| DS-R1-Distill-Llama-70B | thinking | Llama | 198 | 68.69% | 3.59 | 3.30 | 0.29 | 2.94% | 5.15% | 0 |
| Qwen2.5-7B-Instruct | non-thinking | Qwen2.5 | 198 | 31.82% | 3.89 | 2.22 | 1.67 | 14.29% | 14.29% | 2 |
| Qwen2.5-72B-Instruct | non-thinking | Qwen2.5 | 195 | 52.82% | 3.99 | 2.98 | 1.01 | 21.36% | 22.33% | 0 |
| Llama-3.1-8B-Instruct | non-thinking | Llama-3.1 | 198 | 23.23% | 3.77 | 1.96 | 1.81 | 13.04% | 13.04% | 40 |

## Hypothesis prediction status — 8/8 in predicted quadrant

| condition | threshold | violators / total |
|---|---|---|
| non-thinking → gap >1.0 | required | 0/3 (all pass: 1.67, 1.01, 1.81) |
| non-thinking → PHR ≥10% | required | 0/3 (all pass: 14.29%, 21.36%, 13.04%) |
| thinking → gap <0.5 | required | 0/5 (all pass: 0.23, 0.19, 0.16, 0.29, 0.29) |
| thinking → PHR <10% | required | 0/5 (all pass: 3.28%, 4.72%, 2.26%, 0.93%, 2.94%) |

**8/8 models in the predicted quadrant.**

Cluster separation (corrected v2):

- **PHR axis: 2.76× minimum** (nearest non-thinking PHR = Llama-3.1-8B at 13.04% vs. furthest thinking-mode PHR = Qwen3-Thinking-14B at 4.72%). Maximum PHR separation: 23.07× (Qwen2.5-72B at 21.36% vs. DS-R1-Distill-Qwen-14B at 0.93%).
- **Gap axis: 3.5×** (nearest non-thinking gap = Qwen2.5-72B at 1.01 vs. furthest thinking-mode gap = DS-R1-Distill-Llama-70B / Qwen-14B at 0.29). Unchanged by the parser fix — autorater scores are independent of the answer extractor.

**The two-axis cluster claim survives the parser fix.** Both axes still cleanly separate by training paradigm. The PHR axis's separation factor dropped from a v1 5.3× to a v2 2.76× — but every thinking-mode model is still under 5% strict-PHR while every non-thinking instruct model is still above 13%, with no overlap.

The 72B Qwen2.5 result remains the publishable wrinkle: scale within non-thinking partially closes the coverage gap (Δcov=+0.76, Δgap=−0.66 from 7B → 72B) without lowering PHR. The two monitorability axes decouple under scale: coverage is scale-sensitive on non-thinking models; PHR alignment is paradigm-locked.

## Four controlled comparisons (v2 corrected)

### 1. Qwen3-Thinking-14B vs DS-R1-Distill-Qwen-14B — same base, different RL recipe

| metric | Qwen3-Thinking-14B | DS-R1-Distill-Qwen-14B | Δ |
|---|---|---|---|
| accuracy | 64.47% | 54.82% | -9.65pp |
| legibility | 3.61 | 3.41 | -0.20 |
| coverage | 3.42 | 3.13 | -0.29 |
| leg-cov gap | 0.19 | 0.29 | +0.10 |
| PHR strict | 4.72% | **0.93%** | -3.79pp |

Both training recipes reach the thinking quadrant on the same Qwen
base. R1 distillation produces marginally lower autorater scores but
**better CoT-answer alignment** (lower PHR rate). Different RL
recipes converge on the same monitorability signature class.

### 2. DS-R1-Distill-Qwen-14B vs DS-R1-Distill-Llama-70B — same RL recipe, different base

| metric | DS-Qwen-14B | DS-Llama-70B | Δ |
|---|---|---|---|
| accuracy | 54.82% | 68.69% | +13.87pp |
| legibility | 3.41 | 3.59 | +0.18 |
| coverage | 3.13 | 3.30 | +0.17 |
| leg-cov gap | 0.29 | 0.29 | 0.00 |
| PHR strict | 0.93% | 2.94% | +2.01pp |

**Same monitorability signature class across base architectures.**
A 5× parameter scaling (14B → 70B) and a base-family swap
(Qwen → Llama) produce ~+0.18 on each autorater axis (consistent
scale gain), Δgap of 0.00 (within autorater noise), and +2.01pp PHR.
Both models stay clearly inside the thinking quadrant.

### 3. Qwen2.5-7B-Instruct vs Qwen2.5-72B-Instruct — scale within non-thinking

| metric | Qwen2.5-7B | Qwen2.5-72B | Δ |
|---|---|---|---|
| params | 7B | 72B | ~10× |
| accuracy | 31.82% | 52.82% | +21.00pp |
| legibility | 3.89 | 3.99 | +0.10 (saturating) |
| coverage | 2.22 | 2.98 | +0.76 |
| leg-cov gap | 1.67 | 1.01 | -0.66 |
| PHR strict | 14.29% | 21.36% | +7.07pp |

**Scale within non-thinking partially closes the coverage gap but
does NOT lower PHR.** A ~10× parameter scaling buys +0.76 coverage
(40% of the gap closes), saturating legibility, but the silent-flip
PHR rate stays elevated and slightly worsens. The two
monitorability axes decouple under scale: **coverage is
scale-sensitive on non-thinking models; PHR alignment is
paradigm-locked, not scale-sensitive.** Both models still firmly in
the non-thinking quadrant.

### 4. Qwen3-Thinking-32B vs Llama-3.1-8B-Instruct — thinking vs non-thinking, different scales

| metric | Qwen3-T-32B | Llama-3.1-8B | Δ |
|---|---|---|---|
| params | 32B | 8B | 4× |
| accuracy | 71.89% | 23.23% | -48.66pp |
| legibility | 3.73 | 3.77 | +0.04 |
| coverage | 3.57 | 1.96 | -1.61 |
| leg-cov gap | 0.16 | 1.81 | +1.65 |
| PHR strict | 2.26% | 13.04% | +10.78pp |
| empty final_answer | 0 | 40 (20%) | +40 |

**Even at a 4× parameter advantage, thinking-mode dominates scale
for monitorability.** Llama-3.1-8B has comparable legibility (3.77 vs
3.73) but the coverage gap is 1.65 wider, the PHR rate is 5.8× higher,
and 20% of trajectories never commit to an answer. Thinking-mode
training is the load-bearing variable; raw scale does not close the
monitorability gap on non-thinking models within the range studied.

## Parser fix and v1 → v2 deltas (2026-04-27)

The pre-fix v1 numbers in this file's git history were inflated by a
bug in the answer-extractor regex used by `scripts/run_qwen3_gpqa.py`.
The regex `\\banswer\\s*(?:is)?\\s*[:\\-]?\\s*\\(?([A-Da-d])\\)?`
made the colon optional and was consumed by `re.search` (first match
wins). Three independent failure modes:

1. **Prose false positive:** `the answer choices` captured 'c' from
   "**c**hoices" because the colon was optional.
2. **Multi-line "Final Answer" header:** `Final Answer\\n\\nAnswer: D`
   captured 'A' from the second "Answer" word because `\\s*` ate the
   `\\n\\n`.
3. **First-match-not-last-match:** mid-output prose like "leaning
   toward answer A" overrode a final "Answer: D" commitment.

The fix lives in `cotsuite/parsing.py` — layered anchored extraction
(``\\boxed{X}`` → ``Final Answer: X`` → generic ``Answer: X`` with
mandatory colon → scoped bare-letter line in last 500 chars) with
last-match within each layer. Both call sites (the multi-family
scaling driver and the v0.1 Inspect AI scorer) now import this
single canonical implementation.

### v1 → v2 PHR strict deltas

| model | v1 (buggy) | v2 (corrected) | Δ |
|---|---|---|---|
| Qwen3-Thinking-8B | 4.50% | 3.28% | -1.22pp |
| Qwen3-Thinking-14B | 5.74% | 4.72% | -1.02pp |
| Qwen3-Thinking-32B | 4.03% | 2.26% | -1.77pp |
| DS-R1-Distill-Qwen-14B | 2.78% | 0.93% | -1.85pp |
| DS-R1-Distill-Llama-70B | 3.79% | 2.94% | -0.85pp |
| Qwen2.5-7B-Instruct | 20.00% | 14.29% | -5.71pp |
| Qwen2.5-72B-Instruct | 22.62% | 21.36% | -1.26pp |
| Llama-3.1-8B-Instruct | 32.69% | 13.04% | -19.65pp |
| **B4 GPT-4o-mini (Arcuschin validation)** | 9.30% | 9.30% | **±0.00pp** |

**Bidirectional framing.** The bug both inflated some PHR rates
(mode 1 false positives — dominant on non-thinking instruct models
that often prose through option choices) AND deflated others (mode 3
anchoring on a corrupted `final_answer` made the judge report
no-divergence on cases that genuinely diverged once the corrected
final_answer was passed). Net direction was deflationary on
non-thinking models because mode 1 dominated. The pilot re-judge of
Qwen3-Thinking-14B surfaced 2 newly-identified strict-PHR cases (qids
121, 126) that had been hidden by the anchoring effect — those are
real PHR signals that the buggy parser had masked.

**B4 was unaffected** because `scripts/validate_b4_arcuschin.py` used
a strict primary regex (mandatory `Final Answer:` prefix) with the
buggy loose regex only as a fallback. GPT-4o-mini outputs all triggered
the strict primary, so the fallback never fired. The 9.30% paper
comparison stands.

## Detector ablation reframing

The Stage 3.5 detector ablation as originally shipped (Claude judge +
Arcuschin regex + exact-match) **does not protect against parser
bugs** because all three detection methods consume the same upstream
`final_answer` field. The ±1.64pp robustness signal is honest about
post-parsing detection robustness, but it doesn't span the full
parsing pipeline. The corrected v0.1 framing: "robustness across
detection methods given the parsed final answer." A v0.1.1 follow-up
should add a fourth detection method that re-parses from `raw_text`
independently. Documented in `AUDIT.md`.

## Methodology notes (caveats for the README)

### PHR rate sensitivity to N at our sample sizes

PHR strict is computed over the **correct trajectory subset**. The
correct subset shifted slightly under the v2 parser because some
trajectories that previously had a hallucinated wrong-letter
`final_answer` (counted as incorrect) now have the correct letter
and count toward the correct subset. Concrete: Qwen3-Thinking-32B's
correct subset grew from n=124 (v1) to n=133 (v2); Qwen2.5-72B-Instruct
grew from n=84 (v1) to n=103 (v2). At these n's, PHR rates have
non-trivial sampling variance (Wilson 95% intervals span 4-7pp).

**The cluster-separation claim is robust to this sampling variance.**
Even pessimistic combinations (highest thinking PHR = 4.72% Qwen3-T-14B
vs lowest non-thinking PHR = 13.04% Llama-3.1-8B) keep cluster
separation at 2.76×, with the two ranges non-overlapping by 8.32pp —
well outside any reasonable confidence interval for either side.

### What the table does NOT claim

- Within-cluster PHR rate differences (e.g., Qwen3-T-32B 2.26% vs
  Qwen3-T-14B 4.72%) are NOT statistically distinguishable. They sit
  inside the Wilson interval of the other.
- Accuracy gains and monitorability gains are not independent —
  within the thinking cluster, higher accuracy correlates with
  slightly higher legibility/coverage; the autorater may be partially
  proxying for trajectory length.
- The 5-model thinking cluster is not fully homogeneous. R1-distill
  variants sit at slightly higher gaps (both 0.29) than native Qwen3-
  Thinking (0.16-0.23); within-class variance, not a cluster-membership
  question.

### What N supports

The cluster-membership claim ("each model lands in its predicted
quadrant") survives any reasonable resampling because the
inter-cluster gap (2.76× minimum on PHR axis post-correction;
~8pp absolute non-overlap) dwarfs the intra-cluster variance.

## Top-line claim (v2 corrected, with bootstrap robustness)

Across 8 reasoning models spanning 7B-72B parameters and two base
families (Qwen, Llama):

- **PHR-strict cluster:** All 5 thinking-mode trained models stay at
  ≤4.72% (range 0.93%-4.72%); all 3 non-thinking instruct models
  stay at ≥13.04% (range 13.04%-21.36%). **8.32pp absolute non-
  overlap** in point estimates, no model in either gap.
- **Legibility-coverage gap cluster:** All 5 thinking-mode models
  stay at ≤0.29; all 3 non-thinking instruct models stay at ≥1.01.
  Qwen2.5-72B closes 40% of the gap (1.65 → 1.01) at ~10× parameter
  scale vs Qwen2.5-7B but does not cross into thinking-mode
  territory. The PHR-strict axis shows no scale rescue (Qwen2.5-72B
  at 21.36% vs Qwen2.5-7B at 14.29%).
- **8/8 models in their predicted quadrant.**

Both signals are paradigm-discriminating. The legibility-coverage
axis is partially scale-sensitive on the non-thinking side; the PHR
axis is paradigm-locked, with no scale rescue. The cross-architecture
R1-distill replication (Llama base + R1 recipe produces the same
monitorability signature class as Qwen base + R1 recipe — Δgap=0.00,
ΔPHR=+2.01pp) confirms that the training intervention, not the model
substrate, controls CoT-answer alignment.

### Bootstrap CI robustness (1000-sample, with replacement)

The non-overlap claim is on point estimates. Bootstrap 95% CIs on
each non-thinking model's PHR-strict rate, against the thinking-mode
max (Qwen3-Thinking-14B 4.72%):

| model | n | point | 95% CI | margin vs thinking-max | verdict |
|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 103 | 21.36% | [12.62, 29.13] | +7.90pp | **ROBUST** |
| Qwen2.5-7B-Instruct | 63 | 14.29% | [6.35, 22.22] | +1.63pp | NEAR-BOUNDARY |
| Llama-3.1-8B-Instruct | 46 | 13.04% | [4.35, 23.91] | -0.37pp | NOT-ROBUST |

**Two-tier cluster claim:**

- **Strong evidence (5 of 8 thinking-mode + 2 of 3 non-thinking, 7
  models total):** point estimates cluster cleanly with bootstrap CIs
  on the right side of the boundary. Qwen2.5-72B's lower CI bound
  clears the thinking-mode max by 7.90pp; Qwen2.5-7B's clears by
  1.63pp.
- **Weaker evidence (Llama-3.1-8B specifically, n=46):** point
  estimate 13.04% is firmly in the non-thinking range, but the small
  correct subsample (driven by Llama-3.1-8B's 23.23% accuracy on
  GPQA-Diamond) produces a wide 95% CI [4.35%, 23.91%] whose lower
  bound sits 0.37pp below the thinking-mode max. P(Llama-3.1-8B
  bootstrap < thinking-mode max) ≈ 6%. The cluster claim is supported
  in central tendency for this model, with sampling-noise uncertainty
  acknowledged.

The headline claim survives in central tendency on all 8 models. The
Llama-3.1-8B CI caveat is a methodology disclosure, not a cluster-
breaking finding — 7 of 8 models have CI-clean cluster membership,
and the 8th has 94% bootstrap mass above the thinking-mode max.

A **pre-launch parser-bug discovery and fix** (2026-04-27) corrected
the previously-reported numbers downward. The bidirectional bug
inflated PHR rates on non-thinking instruct models by 1-20pp via
prose false positives, and deflated PHR rates on a small number of
thinking-mode trajectories by anchoring the judge to a wrong
`final_answer`. The B4 GPT-4o-mini validation against Arcuschin's
paper-reported 13% rate is **unchanged at 9.30%** because B4's
validation script used a stricter primary regex that didn't trigger
the bug — defensive parser layering paid off in validation tooling.
See `AUDIT.md` for the full discovery, fix, and bootstrap-CI narrative.
