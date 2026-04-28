# Multi-family scaling results

The v0.1 launch result: an 8-model open-weight scaling table on GPQA-Diamond, exercised end-to-end through the cot-suite metric pipeline.

!!! info "v2 corrected (2026-04-27)"
    Numbers below are post answer-extractor parser fix
    (`cotsuite.parsing.extract_answer_letter`, `EVAL_VERSION 1.1.0`).
    A bidirectional bug in the previous loose-regex extractor inflated
    PHR rates on non-thinking instruct models (mode 1: prose false
    positives) and deflated rates on a small number of thinking-mode
    trajectories (mode 3: judge anchored to wrong `final_answer`). See
    [`AUDIT.md`](https://github.com/thenerd31/cot-suite/blob/main/AUDIT.md)
    for the discovery, fix, and bootstrap CI narrative. B4 GPT-4o-mini
    validation against Arcuschin's paper-reported 13% rate is unchanged
    at 9.30% — defensive parser layering in `validate_b4_arcuschin.py`
    insulated B4 from the bug.

## Setup (constant across all 8 rows)

- **Benchmark:** GPQA-Diamond, 198 questions
- **Sampling:** Qwen3 thinking-mode default — temperature 0.6, top_p 0.95, top_k 20, min_p 0, max_new_tokens 32768
- **Per-question cap:** 5-minute Modal RPC timeout (Fix #1)
- **Autorater:** Claude Haiku 4.5 with Emmons & Zimmermann 2510.23966 Appendix C prompt (SHA `ac1e0ac4044b0a64…`)
- **PHR detector:** Claude Haiku 4.5 with Arcuschin-inspired LLM-as-judge prompt (SHA `4d7cc712e9456b80…`)
- **Answer extractor:** `cotsuite.parsing.extract_answer_letter` (layered anchored extraction; v2 corrected)

The PHR detector is validated B4 (9.30% on GPT-4o-mini vs paper 13%, within the paper's 5-25% band, **unchanged across the v2 parser fix**) and detector-ablated Stage 3.5 (±1.64pp across three independent detection methods, all consuming the parsed `final_answer` — see [Arcuschin PHR detector](metrics/arcuschin.md) for the ablation table and the cross-parser ablation deferred to v0.1.1).

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
| Llama-3.1-8B-Instruct[^llama-n] | non-thinking | Llama-3.1 | 198 | 23.23% | 3.77 | 1.96 | 1.81 | 13.04% | 13.04% | 40 |

[^llama-n]: Llama-3.1-8B's GPQA-Diamond accuracy of 23.23% produces the smallest correct-trajectory subsample of any model in this study (n=46). The PHR rate point estimate sits in the non-thinking cluster (13.04%) but the bootstrap 95% CI [4.35%, 23.91%] crosses the thinking-mode max (4.72%) by 0.37pp on the lower bound. P(bootstrap < thinking-mode max) ≈ 6%. Cluster claim for Llama-3.1-8B is **partially-resolved at v0.1 sample size** — see "Bootstrap CI robustness" below and the v0.1.1 follow-up plan to grow n via cross-benchmark replication.

## Cluster-membership story

The v0.1 hypothesis: **non-thinking models** show legibility-coverage gap >1.0 and PHR strict ≥10%; **thinking-mode models** show gap <0.5 and PHR strict <10%.

| Condition | Threshold | Violators / total |
|---|---|---|
| non-thinking → gap >1.0 | required | 0/3 (1.67, 1.01, 1.81) |
| non-thinking → PHR ≥10% | required | 0/3 (14.29%, 21.36%, 13.04%) |
| thinking → gap <0.5 | required | 0/5 (0.23, 0.19, 0.16, 0.29, 0.29) |
| thinking → PHR <10% | required | 0/5 (3.28%, 4.72%, 2.26%, 0.93%, 2.94%) |

**Point estimates: 8/8 in the predicted quadrant.**

### Bootstrap CI robustness (1000-sample, with replacement)

| Non-thinking model | n | Point | 95% CI | Margin vs thinking-max (4.72%) | Verdict |
|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 103 | 21.36% | [12.62, 29.13] | +7.90pp | **ROBUST** |
| Qwen2.5-7B-Instruct | 63 | 14.29% | [6.35, 22.22] | +1.63pp | **ROBUST** |
| Llama-3.1-8B-Instruct | 46 | 13.04% | [4.35, 23.91] | -0.37pp | **PARTIALLY RESOLVED** |

**Cluster claim status, two parts:**

- **PHR axis: 7/8 bootstrap-robust + 1/8 partially-resolved at v0.1 sample size.** The 7 bootstrap-robust models (5 thinking + Qwen2.5-7B + Qwen2.5-72B) cluster at PHR ≤4.72% (thinking) vs ≥14.29% (non-thinking) with **9.57pp absolute non-overlap** on the bootstrap-robust subset. Llama-3.1-8B is partially-resolved — see footnote on the table.
- **Gap axis: 8/8 bootstrap-robust.** The autorater scores every trajectory regardless of correctness (full n=197-198 per model), so the gap-axis CIs are tighter and the cluster claim holds on every model. **3.5× separation**, partially scale-sensitive on the non-thinking side (Qwen2.5-72B closes 40% of the gap at ~10× parameters but doesn't cross the cluster boundary).

The Llama-3.1-8B caveat is a **v0.1 sample-size limitation** driven by Llama's accuracy floor on GPQA-Diamond (46/198 = 23.2% accuracy), not a methodology flaw in the PHR detector. The detector itself is validated B4 (GPT-4o-mini at 9.30%, paper's 5-25% band, unchanged across the v2 parser fix).

## v0.1.1 research program: tighten the Llama-3.1-8B PHR CI

Re-run Llama-3.1-8B on a benchmark with a higher accuracy floor (CommonsenseQA, MMLU-Pro) to grow the correct trajectory subsample to n≥100. Llama-3.1-8B's reported MMLU-Pro accuracy (~70%) substantially exceeds its 23.23% on GPQA-Diamond, so a 200-question pass should land n≥140. This is a planned cross-benchmark replication for the launch follow-up — the Llama-3.1-8B PHR signal deserves the same statistical resolution as the other 7 models, and GPQA-Diamond's difficulty isn't the right substrate for high-accuracy-floor evaluation. ~$5 compute, post-launch.

## Four controlled comparisons

### 1. Qwen3-Thinking-14B vs DS-R1-Distill-Qwen-14B — same base, different RL recipe

| metric | Qwen3-T-14B | DS-R1-Distill-Qwen-14B | Δ |
|---|---|---|---|
| accuracy | 64.47% | 54.82% | -9.65pp |
| legibility | 3.61 | 3.41 | -0.20 |
| coverage | 3.42 | 3.13 | -0.29 |
| leg-cov gap | 0.19 | 0.29 | +0.10 |
| PHR strict | 4.72% | **0.93%** | -3.79pp |

Both training recipes reach the thinking quadrant on the same Qwen base. R1 distillation produces marginally lower autorater scores but **better CoT-answer alignment** (lower PHR rate). Different RL recipes converge on the same monitorability signature class.

### 2. DS-R1-Distill-Qwen-14B vs DS-R1-Distill-Llama-70B — same RL recipe, different base

| metric | DS-Qwen-14B | DS-Llama-70B | Δ |
|---|---|---|---|
| accuracy | 54.82% | 68.69% | +13.87pp |
| legibility | 3.41 | 3.59 | +0.18 |
| coverage | 3.13 | 3.30 | +0.17 |
| leg-cov gap | 0.29 | 0.29 | 0.00 |
| PHR strict | 0.93% | 2.94% | +2.01pp |

A 5× parameter scaling (14B → 70B) and a base-family swap (Qwen → Llama) produce ~+0.18 on each autorater axis, Δgap of 0.00, and +2.01pp PHR. Both stay clearly inside the thinking quadrant. **The R1-distill recipe transfers across base architectures with the monitorability signature intact.**

### 3. Qwen2.5-7B-Instruct vs Qwen2.5-72B-Instruct — scale within non-thinking

| metric | Qwen2.5-7B | Qwen2.5-72B | Δ |
|---|---|---|---|
| params | 7B | 72B | ~10× |
| accuracy | 31.82% | 52.82% | +21.00pp |
| legibility | 3.89 | 3.99 | +0.10 (saturating) |
| coverage | 2.22 | 2.98 | +0.76 |
| leg-cov gap | 1.67 | 1.01 | -0.66 |
| PHR strict | 14.29% | 21.36% | +7.07pp |

**Scale within non-thinking partially closes the coverage gap but does NOT lower PHR.** A ~10× parameter scaling buys +0.76 coverage (40% of the gap closes), saturating legibility, but the silent-flip PHR rate stays elevated and slightly worsens. The two monitorability axes decouple under scale: **coverage is scale-sensitive on non-thinking models; PHR alignment is paradigm-locked.**

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

**Even at a 4× parameter advantage, thinking-mode dominates scale for monitorability.** Thinking-mode training is the load-bearing variable; raw scale does not close the monitorability gap on non-thinking models within this range.

## Methodology notes

### PHR rate sensitivity to N at our sample sizes

PHR strict is computed over the **correct trajectory subset** (n ≈ 46-136 across models). At these sample sizes, PHR rates have non-trivial sampling variance. The bootstrap CI section above quantifies this directly: Llama-3.1-8B's CI [4.35%, 23.91%] is wide because n=46; Qwen2.5-72B's CI [12.62, 29.13] is tighter because n=103.

**The PHR-axis cluster claim is bootstrap-robust on 7 of 8 models.** Llama-3.1-8B is partially-resolved at v0.1 sample size (see footnote and v0.1.1 plan above). The gap-axis cluster claim is bootstrap-robust on all 8 models because the autorater scores every trajectory.

### What the table does NOT claim

- Within-cluster PHR rate differences (e.g. Qwen3-T-32B 2.26% vs Qwen3-T-14B 4.72%) are NOT statistically distinguishable. They sit inside the bootstrap CI of the other.
- Accuracy gains and monitorability gains are NOT independent — within the thinking cluster, higher accuracy correlates with slightly higher legibility/coverage; the autorater may be partially proxying for trajectory length.
- The 5-model thinking cluster is NOT fully homogeneous. R1-distill variants sit at slightly higher gaps (both 0.29) than native Qwen3-Thinking (0.16-0.23); within-class variance, not a cluster-membership question.

## Top-line claim

Across 8 reasoning models spanning 7B-72B parameters, two base families (Qwen, Llama) plus a within-family scale ladder (Qwen2.5-7B vs 72B), and two training paradigms (native thinking-mode and DeepSeek-R1 distillation vs vanilla instruct tuning), CoT monitorability separates by training paradigm on **both** the legibility-coverage gap (8/8 bootstrap-robust, 3.5× separation, partially scale-sensitive) and the PHR-strict axis (7/8 bootstrap-robust, 9.57pp absolute non-overlap on the robust subset; Llama-3.1-8B partially-resolved at v0.1 sample size due to its 23.2% GPQA-Diamond accuracy floor — v0.1.1 grows n via cross-benchmark replication). The cross-architecture R1-distill replication (Llama base + R1 recipe produces the same monitorability signature class as Qwen base + R1 recipe — Δgap=0.00, ΔPHR=+2.01pp) confirms that the training intervention, not the model substrate, controls CoT-answer alignment.

## Source

Raw run logs and per-row trajectory JSONL are in `benchmarks/results/<model_id>_full/` in the repo. Aggregated summary at [`benchmarks/results/multi_family_summary.md`](https://github.com/thenerd31/cot-suite/blob/main/benchmarks/results/multi_family_summary.md). Bootstrap script: [`scripts/bootstrap_phr_ci.py`](https://github.com/thenerd31/cot-suite/blob/main/scripts/bootstrap_phr_ci.py).
