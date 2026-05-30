# Emmons & Zimmermann 2510.23966 — autorater validation

**Paper:** "Chain-of-Thought Monitorability" — Emmons, Zimmermann et
al., 2025. arXiv: 2510.23966.

**Validation goal:** Our autorater implementation (Appendix C prompt,
verbatim, run against Claude Haiku 4.5) produces
**legibility/coverage distributions consistent with the paper's
intent** on the scaling trajectories — NOT exact reproduction of the
**Qwen3-235B-A22B-Thinking row of Table 1** (the lowest-monitorability
model among the table's five model rows) with Gemini 2.5 Pro as rater.

---

## Paper setup

| axis | Emmons & Zimmermann 2025 |
|---|---|
| model under test | Qwen3-235B-A22B-Thinking (one of **five** model rows in Table 1) |
| datasets | HLE, GPQA-Diamond, ARC-AGI, AIME (each row pooled across four) |
| autorater | Gemini 2.5 Pro, n=5 runs per item |
| scale | thousands of trajectories across the pooled sample |
| Table 1 — Qwen3-235B row | legibility 97.33% ± 0.18%, coverage 95.27% ± 0.39% (the **lowest-monitorability** of the five rows) |

Table 1 is **per-model** — five rows, each pooled across the four
datasets. The 97.33/95.27 figures are the **Qwen3-235B-A22B-Thinking row
specifically** (the lowest-monitorability model in the table), not a
single pooled headline. The other four rows (arxiv.org/html/2510.23966v1,
fetched 2026-04-20) are higher: Gemini-2.5-Pro 99.81/99.38,
Gemini-2.5-Flash 99.94/97.89, GPT-OSS-120B 99.71/96.71, DeepSeek-R1
99.20/98.14. There is no per-dataset breakdown within a model row.

---

## Our setup

| axis | This validation |
|---|---|
| models under test | Qwen3-8B, Qwen3-14B, Qwen3-32B |
| dataset | GPQA-Diamond only (198 questions) |
| autorater | **Claude Haiku 4.5, n=1 run per item** |
| prompt | Emmons & Zimmermann Appendix C, verbatim, SHA-256 `ac1e0ac4044b0a64816bc3c4424e547a462e64d18cf0f5555dec7ed5b42aaa67` |
| scoring scale | 0-4 for each of legibility and coverage (paper scale) |

Results per `benchmarks/results/scaling_summary.md`:

| model | legibility mean | coverage mean |
|---|---|---|
| Qwen3-8B | 3.48 | 3.24 |
| Qwen3-14B | 3.61 | 3.42 |
| Qwen3-32B | 3.73 | 3.57 |

---

## What differs

1. **Model under test.** We run on Qwen3 8B/14B/32B; paper runs on
   Qwen3-235B-A22B-Thinking. ~7×–30× parameter gap. Absolute
   legibility/coverage scores are expected to be lower on our
   smaller checkpoints and are not directly comparable to Table 1's
   97%/95%.
2. **Autorater substitution.** We use Haiku 4.5; paper uses Gemini
   2.5 Pro. Cross-rater agreement has not been measured in this
   project (Google AI Studio key hit a 2026-04-21 placeholder issue;
   cross-rater validation is a v0.2 deliverable per BLOCKERS.md).
   Different autoraters produce different distributions on the same
   trajectories; direction should agree, absolute values need not.
3. **Single-shot autorater (n=1 vs paper n=5).** Justified by the
   2026-04-20 variance run: 3/3 convergence on the 4/4 smoke case,
   cov-spread=0 on the boundary case. See `docs/autorater-notes.md`.
4. **Dataset subset.** GPQA-Diamond only; paper pools four datasets.
   Our numbers cannot be compared to Table 1 without the other
   three.

---

## Our numbers — see `benchmarks/results/scaling_summary.md`

The autorater is the main instrument used in every scaling table row,
so the three-row scaling summary IS the validation of this autorater.
Key signals:

- **Monotonic legibility + coverage increase with scale.** If the
  autorater were pathological (insensitive to CoT quality, or biased
  to always score 4s), scaling would not produce this monotonic
  trend. The increase across 8B → 14B → 32B (+0.1 per step per axis)
  indicates the rater responds to CoT quality signal.
- **Rated coverage below legibility consistently** (3.24 < 3.48, 3.42
  < 3.61, 3.57 < 3.73). Matches the paper's general pattern that
  models' CoTs can be legible without being fully covering (paper
  §5.1 discussion). Consistent with the axes measuring distinct
  things.
- **SHA-verified prompt.** Prompt bytes match paper Appendix C.
  Integrity test in `tests/test_appendix_c_prompt_integrity.py`.

---

## Qualitative agreement assessment

The validation passes if all three hold:

1. **Appendix C prompt shipped verbatim.** SHA
   `ac1e0ac4044b0a64…` matches the test harness. Byte-level equality
   with the paper's source. ✅
2. **Autorater produces monotonic direction under scale.** Across
   three Qwen3 sizes, both legibility and coverage increase. If the
   rater were pathological, we'd expect either flat values or
   noise-dominated ordering. ✅
3. **Autorater produces distinct signals on the two axes.** Coverage
   < legibility on every row means the rater is NOT collapsing the
   two scores to a single number — it's responding differently to
   "clarity" vs. "completeness" signals. ✅

### What this validation does NOT claim

- That we reproduce Emmons & Zimmermann's Table 1 headline numbers
  (97.33% / 95.27%) on any model. Reaching that claim requires
  running Qwen3-235B on all four datasets with Gemini 2.5 Pro — ~$200-$280
  Modal + Gemini spend, gated on workspace cap seasoning. Logged as
  v0.2 priority in BLOCKERS.md.
- Cross-rater agreement between Haiku 4.5 and Gemini 2.5 Pro. Not
  measured in this project; v0.2.
- That our scoring scale (0-4) is directly comparable to the paper's
  percentage headline. The paper uses a 0-4 Likert and the 97.33%
  figure is the fraction of trajectories scoring ≥ some threshold;
  our means are on the raw Likert scale.

---

## Future work

1. **Gemini 2.5 Pro cross-rater pass** on the full three scaling
   runs, to establish Haiku-vs-Gemini agreement.
2. **Full four-dataset pooled run** (HLE + GPQA-Diamond + ARC-AGI
   + AIME) on a larger Qwen3 checkpoint, to support the headline
   reproduction claim. BLOCKERS.md "True reproduction of 2510.23966
   pooled four-dataset numbers" — v0.2.
