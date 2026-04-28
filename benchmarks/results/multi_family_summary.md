# Multi-family CoT monitorability scaling — Stage 1+2+3 summary (8 models)

**v2 normalized, 2026-04-28.** Numbers below reflect three layers of pre-launch
audit:

1. **Answer-extraction parser fix** (`cotsuite.parsing.extract_answer_letter`,
   2026-04-27) — corrected a regex bug that inflated apparent PHR rates 5-10×
   on non-thinking models via prose false positives.
2. **Option-letter normalizer** (`cotsuite.normalize_cot.normalize_cot_conclusion`,
   2026-04-28) — resolves judge `cot_conclusion` strings (which can be letters,
   value-strings, concept names, multi-letter ambiguity, or UNCLEAR) against
   the per-question option map before computing divergence. The first draft
   was over-aggressive (treated every non-letter conclusion as unscorable,
   dropping real forced-choice divergences); the v2 normalizer was revised
   after manual case adjudication on Qwen3-Thinking-14B and Qwen2.5-72B.
3. **Bootstrap CIs** on the normalized PHR rates and the legibility-coverage
   gap to quantify cluster-claim robustness across sampling variance.

`EVAL_VERSION = 1.1.0`. Pre-audit (v0/v1) numbers in git history are
NOT directly comparable. See `AUDIT.md` for the full chronological
narrative and reproduction script (`scripts/verify_headline.py --all`).

**Setup (constant across all 8 rows):** GPQA-Diamond (198 questions),
Qwen3 thinking-mode default sampling (temperature 0.6, top_p 0.95,
top_k 20, min_p 0, max_new_tokens 32768), 5-min per-question Modal
RPC cap (Fix #1). Autorater = Claude Haiku 4.5 with Emmons &
Zimmermann 2510.23966 Appendix C prompt (SHA `ac1e0ac4044b0a64…`).
PHR detector = Claude Haiku 4.5 with Arcuschin-inspired LLM-as-judge
prompt (SHA `4d7cc712e9456b80…`). Detector validated B4 — see
"B4 re-validation through v2 normalizer" below.

## Eight-model scaling table (v2 normalized)

| model | mode | base | n | accuracy | legibility | coverage | leg-cov gap | PHR strict (v2 norm) | bootstrap 95% CI | n_scorable |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Thinking-8B | thinking | Qwen3 | 198 | 61.62% | 3.48 | 3.24 | 0.23 | 3.28% | [0.82, 6.56] | 122 |
| Qwen3-Thinking-14B | thinking | Qwen3 | 197 | 64.47% | 3.61 | 3.42 | 0.19 | 3.23% | [0.81, 6.45] | 124 |
| Qwen3-Thinking-32B | thinking | Qwen3 | 185 | 71.89% | 3.73 | 3.57 | 0.16 | 2.26% | [0.00, 5.26] | 133 |
| DS-R1-Distill-Qwen-14B | thinking | Qwen | 197 | 54.82% | 3.41 | 3.13 | 0.29 | 0.93% | [0.00, 2.80] | 107 |
| DS-R1-Distill-Llama-70B | thinking | Llama | 198 | 68.69% | 3.59 | 3.30 | 0.29 | 2.96% | [0.74, 5.93] | 135 |
| Qwen2.5-7B-Instruct | non-thinking | Qwen2.5 | 198 | 31.82% | 3.89 | 2.22 | 1.67 | 14.29% | [6.35, 22.22] | 63 |
| Qwen2.5-72B-Instruct | non-thinking | Qwen2.5 | 195 | 52.82% | 3.99 | 2.98 | 1.01 | 19.80% | [11.88, 28.71] | 101 |
| Llama-3.1-8B-Instruct | non-thinking | Llama-3.1 | 198 | 23.23% | 3.77 | 1.96 | 1.81 | 6.67% | [0.00, 15.56] | 45 |

n_scorable counts trajectories in the correct subset where
`cot_conclusion` is letter-resolvable under the v2 normalizer
(includes letter, value-match, concept-match, forced-choice; excludes
multi-letter ambiguity and UNCLEAR). Headline rate = strict /
n_scorable.

## Top-line claim (v2 normalized)

**cot-suite v0.1 ships an open-source PyPI bundle of 5 published CoT-
faithfulness methodologies (Lanham, Turpin, Chen, Arcuschin, Emmons-
Zimmermann) on Inspect AI. We demonstrate the bundle on 8 open-weight
reasoning models on GPQA-Diamond.**

Two paradigm-discriminating signals:

- **PHR-strict cluster (7/8 bootstrap-robust + 1/8 partially-resolved).**
  5 thinking-mode models at ≤3.28%; 2 non-thinking Qwen2.5 instruct
  models at ≥14.29%. **11pp gap between thinking-mode max (3.28%)
  and non-thinking-Qwen2.5 min (14.29%); Llama-3.1-8B at 6.67% sits
  within that range as the partially-resolved case.** Llama-3.1-8B
  (n_scorable=45) doesn't cluster with either thinking-mode (max
  3.28%) or non-thinking-Qwen2.5 (min 14.29%). **95% bootstrap CI:
  [0.00%, 15.56%].** The CI overlaps both clusters — lower bound
  crosses thinking-mode max, upper bound crosses non-thinking-Qwen2.5
  min. Whether Llama clusters with thinking-mode (least natural given
  its training recipe), with non-thinking-Qwen2.5, or sits in a third
  regime is partially-resolved at v0.1 sample size due to
  Llama-3.1-8B's 23.2% GPQA-Diamond accuracy floor. v0.1.1
  cross-benchmark replication on CommonsenseQA / MMLU-Pro planned to
  grow n≥100.
- **Legibility-coverage gap cluster (8/8 bootstrap-robust).** All
  thinking ≤0.29 (max CI hi 0.441), all non-thinking ≥1.012 (min CI
  lo 0.774). 3.5× point-estimate separation, partially scale-
  sensitive on non-thinking side (Qwen2.5-72B closes 40% of the gap
  at ~10× parameters vs Qwen2.5-7B but doesn't cross the cluster
  boundary). Autorater-based, independent of answer-extraction
  pipeline.

The PHR figures reflect three layers of pre-launch audit:
(1) parser bug in answer extraction (caught 2026-04-27);
(2) judge labeling artifacts resolved via option-letter normalization
(2026-04-28); (3) the first normalizer was over-aggressive and
dropped real forced-choice divergences — revised after manual case
adjudication on Qwen3-Thinking-14B + Qwen2.5-72B. Per-model deltas
across the three layers in the "Three-layer audit" section below.
Full audit trail in `AUDIT.md`. Reproducibility primitive:
`scripts/verify_headline.py --all`.

## Cluster verdict (v2 normalized + bootstrap-tested)

**Gap axis (8/8 bootstrap-robust):** all thinking-mode gap ≤0.29
(point estimates 0.16-0.29, max bootstrap CI hi 0.441); all
non-thinking gap ≥1.012 (point estimates 1.012-1.808, min bootstrap
CI lo 0.774). 3.5× point-estimate separation, partially scale-
sensitive on non-thinking side. **Per-model bootstrap CIs are non-
overlapping across the cluster boundary on every model.**

**PHR axis (7/8 bootstrap-robust + 1/8 partially-resolved):**

| model | mode | PHR (v2 norm) | bootstrap 95% CI | margin vs cluster boundary | verdict |
|---|---|---|---|---|---|
| Qwen3-Thinking-8B | thinking | 3.28% | [0.82, 6.56] | CI hi +7.73pp under non-thinking-Qwen2.5 min (14.29%) | ROBUST |
| Qwen3-Thinking-14B | thinking | 3.23% | [0.81, 6.45] | CI hi +7.84pp under non-thinking min | ROBUST |
| Qwen3-Thinking-32B | thinking | 2.26% | [0.00, 5.26] | CI hi +9.03pp under non-thinking min | ROBUST |
| DS-R1-Distill-Qwen-14B | thinking | 0.93% | [0.00, 2.80] | CI hi +11.49pp under non-thinking min | ROBUST |
| DS-R1-Distill-Llama-70B | thinking | 2.96% | [0.74, 5.93] | CI hi +8.36pp under non-thinking min | ROBUST |
| Qwen2.5-7B-Instruct | non-thinking | 14.29% | [6.35, 22.22] | CI lo +3.07pp above thinking max (3.28%) | ROBUST |
| Qwen2.5-72B-Instruct | non-thinking | 19.80% | [11.88, 28.71] | CI lo +8.60pp above thinking max | ROBUST |
| **Llama-3.1-8B-Instruct** | non-thinking | **6.67%** | **[0.00, 15.56]** | **CI lo -3.28pp below thinking max** | **PARTIALLY RESOLVED** |

Llama-3.1-8B is the partially-resolved case: point estimate 6.67%
sits 3.39pp above thinking max (3.28%) but n_scorable=45 produces a
wide CI [0.00%, 15.56%] whose lower bound crosses the boundary. This
is a v0.1 sample-size limitation driven by Llama-3.1-8B's 23.2%
GPQA-Diamond accuracy floor — n=46 correct trajectories before
normalization, n=45 scorable. v0.1.1 cross-benchmark replication
plan: re-run on CommonsenseQA / MMLU-Pro to grow n≥100.

**Bootstrap-robust subset (7/8): 11.01pp absolute non-overlap on
PHR axis** (thinking max 3.28% vs Qwen2.5-7B at 14.29%). The
27.7×-separation framing on point estimates (Qwen2.5-72B at 19.80%
vs DS-R1-Distill-Qwen-14B at 0.93%) overstates the bootstrap-
robust separation, which is what reviewers should check.

The 72B Qwen2.5 result remains a notable scale-sensitivity datapoint
on the gap axis: scale within non-thinking partially closes the
coverage gap (Δcov=+0.76, Δgap=−0.66 from Qwen2.5-7B to 72B at ~10×
parameters) **without crossing the cluster boundary** and without
lowering PHR (14.29% → 19.80% from 7B → 72B). The two monitorability
axes decouple under scale: coverage is scale-sensitive on non-
thinking models; PHR alignment is paradigm-locked.

## Four controlled comparisons (v2 corrected)

### 1. Qwen3-Thinking-14B vs DS-R1-Distill-Qwen-14B — same base, different RL recipe

| metric | Qwen3-Thinking-14B | DS-R1-Distill-Qwen-14B | Δ |
|---|---|---|---|
| accuracy | 64.47% | 54.82% | -9.65pp |
| legibility | 3.61 | 3.41 | -0.20 |
| coverage | 3.42 | 3.13 | -0.29 |
| leg-cov gap | 0.19 | 0.29 | +0.10 |
| PHR strict (v2 norm) | 3.23% | **0.93%** | -2.30pp |

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
| PHR strict (v2 norm) | 0.93% | 2.96% | +2.03pp |

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
| PHR strict (v2 norm) | 14.29% | 19.80% | +5.51pp |

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
| PHR strict (v2 norm) | 2.26% | 6.67% | +4.41pp |
| empty final_answer | 0 | 40 (20%) | +40 |

**Even at a 4× parameter advantage, thinking-mode dominates scale
for monitorability.** Llama-3.1-8B has comparable legibility (3.77 vs
3.73) but the coverage gap is 1.65 wider, the PHR rate is 5.8× higher,
and 20% of trajectories never commit to an answer. Thinking-mode
training is the load-bearing variable; raw scale does not close the
monitorability gap on non-thinking models within the range studied.

## Three-layer audit: v0 → v1 → v2 PHR deltas

The PHR rates in this report passed through three layers of pre-
launch audit (full chronological narrative in `AUDIT.md`):

1. **v0 (pre-audit):** raw judge `diverged && !acknowledged` over the
   pre-fix parser's `final_answer` field. Inflated by the answer-
   extractor regex bug.
2. **v1 (parser fix, 2026-04-27):** corrected `final_answer`
   extraction via `cotsuite.parsing.extract_answer_letter` (layered
   anchored regex, mandatory colon, last-match within each layer).
3. **v2 (option-letter normalization, 2026-04-28):** judge
   `cot_conclusion` strings (letters / value-strings / concept names /
   multi-letter / UNCLEAR) resolved against the per-question option
   map via `cotsuite.normalize_cot.normalize_cot_conclusion`. Skip-
   as-NaN for unscorable conclusions; forced-choice for definite
   values that don't match any option. Applied only to judge-flagged
   `diverged=True` rows — judge `diverged=False` calls trusted
   directly.

| model | v0 (raw) | v1 (parser fix) | v2 (normalized) | total Δ |
|---|---|---|---|---|
| Qwen3-Thinking-8B | 4.50% | 3.28% | 3.28% | -1.22pp |
| Qwen3-Thinking-14B | 5.74% | 4.72% | 3.23% | -2.51pp |
| Qwen3-Thinking-32B | 4.03% | 2.26% | 2.26% | -1.77pp |
| DS-R1-Distill-Qwen-14B | 2.78% | 0.93% | 0.93% | -1.85pp |
| DS-R1-Distill-Llama-70B | 3.79% | 2.94% | 2.96% | -0.83pp |
| Qwen2.5-7B-Instruct | 20.00% | 14.29% | 14.29% | -5.71pp |
| Qwen2.5-72B-Instruct | 22.62% | 21.36% | 19.80% | -2.82pp |
| **Llama-3.1-8B-Instruct** | 32.69% | 13.04% | **6.67%** | **-26.02pp** |
| **B4 GPT-4o-mini (Arcuschin validation)** | 9.30% | 9.30% | **4.88%** | **-4.42pp** |

**Why each layer mattered.**

- **Parser fix (v0 → v1):** mostly dropped non-thinking rates
  because the buggy regex's prose false positives (capturing 'c'
  from "answer **c**hoices", 'A' from "**A**nswer" in
  `"Final Answer\n\nAnswer: D"`) were more common in non-thinking
  outputs that prose through option choices before committing.
- **Normalization (v1 → v2):** dropped rates further on cases
  where the judge described the CoT conclusion in non-letter form
  (value strings, concept names) that string-compared as different
  from `final_answer` even when the underlying answer agreed.
  Bigger effect on non-thinking models where judge descriptions
  tended to be more verbose.

**B4 GPT-4o-mini drop from 9.30% (v0/v1) to 4.88% (v2)** is
significant. Same mechanism as the multi-family models — judge
described conclusions in value-strings the v2 normalizer correctly
unmaps. Paper reports ~13% on GPT-4o-mini; v2 corrected sits
8.12pp below paper's value, outside ±5pp tolerance. **v0.1.1 cross-
parser + cross-judge ablation on Arcuschin's original ChainScope
dataset planned** to determine whether the paper's number itself
contained labeling artifacts our normalizer drops, or whether our
normalizer is too conservative on UNCLEAR cases. See
`validation/arcuschin_2503.08679.md`.

**Llama-3.1-8B v0 → v2 drop is the largest** (-26.02pp). This
reflects two effects compounding: (a) the model's outputs frequently
trigger the parser bug because it truncates mid-calculation
(genuine non-commitments the buggy parser turned into hallucinated
letters); (b) when committed, its CoT conclusions often appear in
value-string or concept-string form the normalizer correctly unmaps.
Both are correct corrections; the headline 6.67% is the v2 honest
rate, not a methodology failure.

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

## Methodology notes

### PHR rate sensitivity to N at our sample sizes

PHR strict is computed over the **scorable correct trajectory
subset** — `is_correct=True` AND the v2 normalizer was able to
resolve `cot_conclusion` to a letter (or flag it as forced-choice
divergence). Per-model n_scorable values are in the scaling table.
At n=45-135, PHR rates have non-trivial sampling variance — see the
"Cluster verdict" section above for bootstrap CIs.

**The cluster-separation claim is robust on 7 of 8 models** under
1000-sample bootstrap. Llama-3.1-8B is the partially-resolved case:
point estimate 6.67% sits in non-thinking range but n_scorable=45
produces a wide CI [0.00%, 15.56%] whose lower bound crosses the
thinking-mode max. v0.1.1 cross-benchmark replication (CommonsenseQA
/ MMLU-Pro, ~$5 compute) is the planned remediation.

### What the table does NOT claim

- Within-cluster PHR rate differences (e.g., Qwen3-T-32B 2.26% vs
  Qwen3-T-14B 3.23%) are NOT statistically distinguishable — bootstrap
  CIs heavily overlap.
- Accuracy gains and monitorability gains are not independent —
  within the thinking cluster, higher accuracy correlates with
  slightly higher legibility/coverage; the autorater may be partially
  proxying for trajectory length.
- The 5-model thinking cluster is not fully homogeneous. R1-distill
  variants sit at slightly higher gaps (both 0.29) than native Qwen3-
  Thinking (0.16-0.23); within-class variance, not a cluster-membership
  question.
- **Judge-error residual.** Spot-checks (qids 072 and 134 on
  Qwen2.5-72B; see `AUDIT.md`) suggest a ~3-5% judge-error rate on
  `diverged=True` calls. v0.1.1 cross-judge ablation planned.

### What N supports

The cluster-membership claim survives any reasonable resampling on
7 of 8 models because bootstrap CIs are non-overlapping across the
boundary. Llama-3.1-8B's CI overlap reflects sample-size limitation,
not cluster-claim collapse — point estimate (6.67%) supports
non-thinking membership; lower-tail uncertainty acknowledged.

## Reproducibility primitives

Every row in `post_hoc_rationalization_v2.jsonl` exposes:

- `final_answer`: v2 parser-extracted letter
- `cot_conclusion`: raw judge string
- `cot_conclusion_normalized`: v2 normalizer canonical letter (or
  null for forced-choice / unscorable)
- `phr_scorable`: True if normalizer resolved or judge said
  diverged=False
- `diverged_normalized`: post-normalization divergence verdict
- `phr_strict_normalized`: True iff diverged_normalized && !acknowledged
- `phr_normalization_flag`: case-type tag (`letter_divergence`,
  `value_match`, `concept_substring`, `forced_choice`,
  `duplicate_options_resolved`, `letter_match`,
  `unscorable_ambiguous`, `unscorable_empty`,
  `judge_no_divergence`, `judge_error_or_skip`)

Reproduction primitive: `scripts/verify_headline.py --all` reads
these fields and recomputes the headline rate per model. Passes
8/8 within ≤0.5pp tolerance.

Companion scripts:
- `scripts/reconstruct_options.py` — rebuilds the per-question
  option map from GPQA-Diamond using the same `random.Random(i)`
  shuffle the run-driver applied.
- `scripts/materialize_v2_normalized.py` — re-applies the v2
  normalizer + re-extracts v2 parser output across all 8 models.
- `scripts/bootstrap_phr_ci.py` — 1000-sample bootstrap CIs on
  each model's normalized PHR rate.

The 9-block adjudication dumps used to validate the v2 normalizer
are at `benchmarks/results/qwen3_14b_gpqa_full/adjudication_dump.txt`
(first round) and
`benchmarks/results/qwen25_72b_instruct_full/adjudication_dump.txt`
(second round, post-revision).
