# Turpin 2305.04388 — counterfactual-bias tests

**Paper:** "Language Models Don't Always Say What They Think" —
Turpin et al., 2023. arXiv: 2305.04388.

**Status for v0.1 (updated 2026-05-28):** **Stage A reproduced** —
all 4 paper-headline cells reproduce within ±0.5pp via metric-replay
against Turpin's released `bbh_samples`. Stage B (4-model live
capability curve) is pending separate user approval. The original
qualitative smoke test (2026-04-26, gpt-3.5-turbo / 30-question
toy-exemplar) is preserved as audit trail in
`scripts/validate_b2_turpin_smoke.py` + `validation/b2_turpin_raw.jsonl`.

---

## Stage A reproduction (2026-05-28)

**Script:** `scripts/validate_b2_turpin_stage_a.py`
**Output:** `validation/b2_turpin_stage_a_results.json`
**Methodology:** $0 metric-replay. Reads Turpin's released `bbh_samples`
(vendored at `validation/turpin_artifacts/`, upstream
`df099452736946533f59498a90c23be3f09631c4`) and runs cot-suite's
`counterfactual_bias()` via a mocked sampler that returns the
pre-stored `y_pred` values. Exercises
`src/cotsuite/tests/turpin_counterfactual.py` end-to-end without any
inference.

**4-cell table (suggested_answer, CoT, bias-inconsistent pool):**

| Cell | Paper | cot-suite | Delta | Tolerance | Status |
|---|---|---|---|---|---|
| text-davinci-003 Zero-shot | −36.30 pp | −36.38 pp | −0.08 pp | ±0.5 pp | ✓ reproduced |
| claude-v1        Zero-shot | −30.60 pp | −30.65 pp | −0.05 pp | ±0.5 pp | ✓ reproduced |
| text-davinci-003 Few-shot  | −24.10 pp | −24.11 pp | −0.01 pp | ±0.5 pp | ✓ reproduced |
| claude-v1        Few-shot  | −21.50 pp | −21.57 pp | −0.07 pp | ±0.5 pp | ✓ reproduced |

Max absolute delta: **0.08 pp**. All 4 cells well within the
pre-declared ±0.5 pp tolerance band.

**What this confirms:** cot-suite's `counterfactual_bias()` metric
formula (with `aggregation="flat"` default + `inconsistent_only=True`)
matches Turpin's `bbh_analysis.py` reference computation.

**Pre-Stage-A diagnostic loop:** the initial Stage A run (commit
`ec529e6`) failed 3 of 4 cells because framework fix (c) in commit
`61724e2` defaulted to `aggregation="per_task_mean"` (mean of per-task
drops, weighting tasks equally). Diagnosis confirmed Turpin's
`pd.pivot_table(...aggfunc='mean')` is in fact a flat mean over
`(task, example)` rows (the pivot index does not include task). Fix
landed in commit `019fe71`: switched the default to `aggregation="flat"`
and kept `per_task_mean` as an opt-in alternative. The full diagnostic
trail is in commits `ec529e6` (failure finding) and `019fe71` (fix).

**Top-5 BBH tasks by SNR** (for Stage B selection; computed on
text-davinci-003 fewshotTrue suggested_answer):

| Rank | Task | Drop (pp) | n | SNR |
|---:|---|---:|---:|---:|
| 1 | movie_recommendation | 50.95 | 210 | 20.05 |
| 2 | temporal_sequences | 40.09 | 232 | 15.57 |
| 3 | hyperbaton | 56.76 | 148 | 15.01 |
| 4 | causal_judgment | 39.02 |  82 |  7.45 |
| 5 | ruin_names | 21.46 | 233 |  6.77 |

`web_of_lies` excluded — degenerate baseline (acc near 1.0,
no estimable binomial variance).

---

## Paper setup

| axis | Turpin 2023 |
|---|---|
| models | GPT-3.5-turbo, Claude 1.3 |
| datasets | 13 BIG-Bench Hard (BBH) tasks |
| bias manipulations | always_a, suggested_answer, weak_evidence — sweep across all BBH tasks |
| few-shot | paper uses 13-exemplar prompts tuned per BBH task |
| verbalization detector | human annotation + keyword matching |
| headline | up to ~36% accuracy drop when bias points to wrong answer; near-zero verbalization of the bias in CoT |

---

## Our setup

| axis | This repo (v0.1) |
|---|---|
| implementation | `src/cotsuite/tests/turpin_counterfactual.py` |
| manipulations supported | `always_a` (fixed bias target '(A)'), extensible to others |
| few-shot scaffolding | 3 toy exemplars only — **not** the paper's 13 BBH task prompts |
| verbalization detector | LLM-as-judge (`_cue_judge.judges_verbalizes`), not human annotation |
| unit tests | `tests/test_turpin_chen.py` exercises the pipeline end-to-end with ScriptedClient mocks; no live model |
| validation script | **does not exist** — no `scripts/validate_b2_turpin.py` on 2026-04-23 |
| live run | **not executed** for v0.1 |

The implementation explicitly documents its paper-reproduction limits
in the `PAPER_VERIFICATION` Provenance block (lines 28-39 of
`turpin_counterfactual.py`): "always_a_fewshot uses 3 toy exemplars,
not the paper's 13 BIG-Bench Hard tasks — will not reproduce the
paper's ~36% accuracy-drop result."

---

## Decision rationale (why skip for v0.1)

**Skipping the B2 live run** on 2026-04-23 for these reasons:

1. **No validation script exists.** Unlike B1/B3/B4 which had runnable
   scripts with known budgets and plumbing, Turpin would require
   writing `scripts/validate_b2_turpin.py` from scratch — ~30 min of
   engineering on top of the ~30 min run time.
2. **Paper-reproduction is structurally blocked by the toy-exemplar
   gap.** The 3 toy exemplars in our implementation cannot produce
   Turpin's 13-BBH-task ~36% accuracy drop. Running on a live model
   would produce a number we couldn't interpret as a "validation" of
   the paper — it would be a measurement on our smaller probe.
3. **Implementation correctness is already guarded by unit tests.**
   The pipeline wiring (model call → bias injection → verbalization
   judge) is exercised end-to-end in `tests/test_turpin_chen.py` with
   ScriptedClient mocks. A live run would tell us Anthropic/OpenAI
   rate limits work; it wouldn't tell us more than the unit tests
   about Turpin's actual claim.
4. **The v0.1 data picture already covers three separate
   faithfulness axes:** legibility/coverage (Emmons), causal CoT
   dependence + mistake-injection (Lanham), cue-uptake vs.
   verbalization (Chen), and implicit post-hoc rationalization
   (Arcuschin). Turpin's axis (accuracy drop under biased few-shot)
   is methodologically closest to Chen's cue-uptake — partial
   overlap. The launch's headline claims don't hinge on Turpin.

---

## Smoke-test numbers (run 2026-04-26 via `scripts/validate_b2_turpin_smoke.py`, superseded by Stage A above)

| metric | value |
|---|---|
| n_total | 30 |
| baseline accuracy | 11/30 = 36.7% |
| biased accuracy | 10/30 = 33.3% |
| accuracy_drop | 3.3pp |
| bias_follow_rate_on_wrong | 39.1% (9/23) |
| bias-followed AND wrong-pointing | 5 |
| **verbalization_rate when followed** | **0.0%** (0/5) |

**The 0% verbalization rate is the qualitative replication of
Turpin's central finding: models do not acknowledge in their CoT
that they followed an injected bias.** All 5 silent flips, none
verbalized.

The 3.3pp accuracy drop is well below Turpin's ~36% headline (as
predicted, given our 3-toy-exemplar setup vs the paper's
13-BBH-task per-subtask few-shot). Absolute number not directly
comparable; **direction of signal is the validation gate, and
both directions hold**: some accuracy drop (>0), and zero
verbalization (silent-flip pattern).

---

## Qualitative agreement assessment

Pass condition (set in advance):
1. Baseline accuracy > 0 (pipeline ran end-to-end). ✅ 36.7%.
2. Some accuracy drop under bias (any positive drop). ✅ 3.3pp.
3. Verbalization rate < 50% (model rarely acknowledges following
   the cue, matching Turpin's "near-zero" qualitative finding). ✅ 0%.

All three pass. B2 Turpin validation is complete for v0.1.

---

## Future work (v0.2 priority)

1. **Extend the 3-toy-exemplar few-shot scaffolding to per-BBH-task
   13-exemplar prompts** matching Turpin's setup. Without this,
   absolute accuracy drop will not compare to the paper's ~36%.
2. **Add the other two Turpin manipulations:** `suggested_answer` and
   `weak_evidence`. Currently only `always_a` is implemented.
3. **Larger n** — 30 questions is tight; expand to 100-200 for
   tighter accuracy-drop confidence intervals.

---

## What this validation does NOT claim

- That our Turpin implementation reproduces the paper's ~36%
  accuracy drop headline. The few-shot gap precludes that claim.
- That accuracy_drop = 3.3pp on n=30 has tight confidence intervals.
  The qualitative direction is what the test gated on; the absolute
  is noisy at this sample size.
