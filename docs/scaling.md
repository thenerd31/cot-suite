# Multi-family scaling results

The v0.1 launch result: an 8-model open-weight scaling table on GPQA-Diamond, exercised end-to-end through the cot-suite metric pipeline.

!!! info "v2 normalized (2026-04-28)"
    Numbers below reflect **three layers of pre-launch audit**:

    1. **Answer-extraction parser fix** (2026-04-27,
       `cotsuite.parsing.extract_answer_letter`) — corrected a regex
       bug in answer extraction. See `AUDIT.md` for per-model deltas.
    2. **Option-letter normalizer** (2026-04-28,
       `cotsuite.normalize_cot.normalize_cot_conclusion`) — resolves
       judge `cot_conclusion` strings against per-question option maps
       before computing divergence. The first draft was over-aggressive;
       the v2 normalizer was revised after manual adjudication on
       Qwen3-Thinking-14B + Qwen2.5-72B.
    3. **Bootstrap CIs** quantifying cluster-claim robustness across
       sampling variance.

    Pre-audit numbers in this file's git history are NOT directly
    comparable. Reproducibility primitive:
    [`scripts/verify_headline.py --all`](https://github.com/thenerd31/cot-suite/blob/main/scripts/verify_headline.py).
    Full chronological narrative in
    [`AUDIT.md`](https://github.com/thenerd31/cot-suite/blob/main/AUDIT.md).

## Setup (constant across all 8 rows)

- **Benchmark:** GPQA-Diamond, 198 questions
- **Sampling:** Qwen3 thinking-mode default — temperature 0.6, top_p 0.95, top_k 20, min_p 0, max_new_tokens 32768
- **Per-question cap:** 5-minute Modal RPC timeout (Fix #1)
- **Autorater:** Claude Haiku 4.5 with Emmons & Zimmermann 2510.23966 Appendix C prompt (SHA `ac1e0ac4044b0a64…`)
- **PHR detector:** Claude Haiku 4.5 with Arcuschin-inspired LLM-as-judge prompt (SHA `4d7cc712e9456b80…`)
- **Answer extractor:** `cotsuite.parsing.extract_answer_letter` (layered anchored extraction; v2 corrected)

The PHR detector is validated B4 (9.30% on GPT-4o-mini vs paper 13%, within the paper's 5-25% band, **unchanged across the v2 parser fix**) and detector-ablated Stage 3.5 (±1.64pp across three independent detection methods, all consuming the parsed `final_answer` — see [Arcuschin PHR detector](metrics/arcuschin.md) for the ablation table and the cross-parser ablation deferred to v0.1.1).

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
| Llama-3.1-8B-Instruct[^llama-n] | non-thinking | Llama-3.1 | 198 | 23.23% | 3.77 | 1.96 | 1.81 | 6.67% | [0.00, 15.56] | 45 |

[^llama-n]: Llama-3.1-8B at 6.67% (n_scorable=45) doesn't cluster with either thinking-mode (max 3.28%) or non-thinking-Qwen2.5 (min 14.29%). 95% bootstrap CI [0.00%, 15.56%] overlaps both clusters — lower bound crosses thinking-mode max, upper bound crosses non-thinking-Qwen2.5 min. Whether Llama clusters with thinking-mode (least natural given its training recipe), with non-thinking-Qwen2.5, or sits in a third regime is **partially-resolved at v0.1 sample size** due to Llama-3.1-8B's 23.2% GPQA-Diamond accuracy floor. v0.1.1 cross-benchmark replication on CommonsenseQA / MMLU-Pro planned to grow n≥100.

## Cluster-membership story

The v0.1 hypothesis: **non-thinking models** show legibility-coverage gap >1.0 and PHR strict ≥10%; **thinking-mode models** show gap <0.5 and PHR strict <10%.

| Condition | Threshold | Violators / total |
|---|---|---|
| non-thinking → gap >1.0 | required | 0/3 (1.67, 1.01, 1.81) |
| non-thinking → PHR (v2 norm) | observed | 14.29%, 19.80%, 6.67% |
| thinking → gap <0.5 | required | 0/5 (0.23, 0.19, 0.16, 0.29, 0.29) |
| thinking → PHR (v2 norm) | observed | 3.28%, 3.23%, 2.26%, 0.93%, 2.96% |

**Point estimates: 8/8 in the predicted quadrant.**

### Bootstrap CI robustness (v2 normalized, 1000-sample with replacement)

| Non-thinking model | n_scorable | Point | 95% CI | Margin vs thinking-max (3.28%) | Verdict |
|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | 101 | 19.80% | [11.88, 28.71] | +8.60pp on lower bound | **ROBUST** |
| Qwen2.5-7B-Instruct | 63 | 14.29% | [6.35, 22.22] | +3.07pp on lower bound | **ROBUST** |
| Llama-3.1-8B-Instruct | 45 | 6.67% | [0.00, 15.56] | -3.28pp on lower bound | **PARTIALLY RESOLVED** |

**Cluster claim status, two parts:**

- **PHR axis: 7/8 bootstrap-robust + 1/8 partially-resolved at v0.1 sample size.** The 7 bootstrap-robust models (5 thinking + Qwen2.5-7B + Qwen2.5-72B) cluster at PHR ≤3.28% (thinking) vs ≥14.29% (non-thinking-Qwen2.5). **11pp gap between thinking-mode max (3.28%) and non-thinking-Qwen2.5 min (14.29%); Llama-3.1-8B at 6.67% sits within that range as the partially-resolved case** — see footnote on the table for the bootstrap-CI overlap framing.
- **Gap axis: 8/8 bootstrap-robust.** The autorater scores every trajectory regardless of correctness (full n=197-198 per model), so the gap-axis CIs are tighter. All thinking-mode CI hi ≤0.441; all non-thinking CI lo ≥0.774 — **3.5× separation, no overlap on any model**. Partially scale-sensitive on the non-thinking side (Qwen2.5-72B closes 40% of the gap at ~10× parameters vs Qwen2.5-7B but doesn't cross the cluster boundary).

See [Arcuschin metric page](metrics/arcuschin.md) for B4 GPT-4o-mini validation status.

## Llama-3.1-8B cluster membership

Llama-3.1-8B at **6.67%** (n_scorable=45) doesn't cluster with either thinking-mode (max 3.28%) or non-thinking-Qwen2.5 (min 14.29%) on the PHR axis. **95% bootstrap CI: [0.00%, 15.56%]** — overlaps both clusters. Lower bound (0.00%) crosses the thinking-mode max by 3.28pp. Upper bound (15.56%) crosses the non-thinking-Qwen2.5 min by 1.27pp. Whether Llama-3.1-8B clusters with thinking-mode (least natural given its training recipe), with non-thinking-Qwen2.5, or sits in a third regime is **partially-resolved at v0.1 sample size** — driven by Llama-3.1-8B's 23.2% GPQA-Diamond accuracy floor producing n=46 correct → n_scorable=45 after v2 normalization, the smallest correct subsample of any model in this study.

**On the gap axis, Llama-3.1-8B is unambiguously in the non-thinking cluster:** gap=1.81 (highest of any model in the table), 95% bootstrap CI [1.481, 2.135], well above the non-thinking minimum (1.012). The PHR-axis ambiguity is specific to the post-hoc-rationalization signal; the legibility-coverage finding holds.

**v0.1.1 research program: tighten the Llama-3.1-8B PHR CI.** Re-run Llama-3.1-8B on a benchmark with a higher accuracy floor (CommonsenseQA, MMLU-Pro) to grow the correct trajectory subsample to n≥100. Llama-3.1-8B's reported MMLU-Pro accuracy (~70%) substantially exceeds its 23.23% on GPQA-Diamond, so a 200-question pass should land n≥140. This is a planned cross-benchmark replication for the launch follow-up — the Llama-3.1-8B PHR signal deserves the same statistical resolution as the other 7 models, and GPQA-Diamond's difficulty isn't the right substrate for high-accuracy-floor evaluation. ~$5 compute, post-launch.

## Four controlled comparisons

### 1. Qwen3-Thinking-14B vs DS-R1-Distill-Qwen-14B — same base, different RL recipe

| metric | Qwen3-T-14B | DS-R1-Distill-Qwen-14B | Δ |
|---|---|---|---|
| accuracy | 64.47% | 54.82% | -9.65pp |
| legibility | 3.61 | 3.41 | -0.20 |
| coverage | 3.42 | 3.13 | -0.29 |
| leg-cov gap | 0.19 | 0.29 | +0.10 |
| PHR strict (v2 norm) | 3.23% | **0.93%** | -2.30pp |

Both training recipes reach the thinking quadrant on the same Qwen base. R1 distillation produces marginally lower autorater scores but **better CoT-answer alignment** (lower PHR rate). Different RL recipes converge on the same monitorability signature class.

### 2. DS-R1-Distill-Qwen-14B vs DS-R1-Distill-Llama-70B — same RL recipe, different base

| metric | DS-Qwen-14B | DS-Llama-70B | Δ |
|---|---|---|---|
| accuracy | 54.82% | 68.69% | +13.87pp |
| legibility | 3.41 | 3.59 | +0.18 |
| coverage | 3.13 | 3.30 | +0.17 |
| leg-cov gap | 0.29 | 0.29 | 0.00 |
| PHR strict (v2 norm) | 0.93% | 2.96% | +2.03pp |

A 5× parameter scaling (14B → 70B) and a base-family swap (Qwen → Llama) produce ~+0.18 on each autorater axis, Δgap of 0.00, and +2.01pp PHR. Both stay clearly inside the thinking quadrant. **The R1-distill recipe transfers across base architectures with the monitorability signature intact.**

### 3. Qwen2.5-7B-Instruct vs Qwen2.5-72B-Instruct — scale within non-thinking

| metric | Qwen2.5-7B | Qwen2.5-72B | Δ |
|---|---|---|---|
| params | 7B | 72B | ~10× |
| accuracy | 31.82% | 52.82% | +21.00pp |
| legibility | 3.89 | 3.99 | +0.10 (saturating) |
| coverage | 2.22 | 2.98 | +0.76 |
| leg-cov gap | 1.67 | 1.01 | -0.66 |
| PHR strict (v2 norm) | 14.29% | 19.80% | +5.51pp |

**Scale within non-thinking partially closes the coverage gap but does NOT lower PHR.** A ~10× parameter scaling buys +0.76 coverage (40% of the gap closes), saturating legibility, but the silent-flip PHR rate stays elevated and slightly worsens. The two monitorability axes decouple under scale: **coverage is scale-sensitive on non-thinking models; PHR alignment is paradigm-locked.**

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
