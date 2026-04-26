# Turpin 2305.04388 — counterfactual-bias tests

**Paper:** "Language Models Don't Always Say What They Think" —
Turpin et al., 2023. arXiv: 2305.04388.

**Status for v0.1:** **Validated qualitatively** on
GPT-3.5-turbo / BBH (30 samples, 3-toy-exemplar always_a_fewshot
bias) — see "Our numbers" below. Run landed 2026-04-26 via
`scripts/validate_b2_turpin.py`.

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
| implementation | `src/cotmon/tests/turpin_counterfactual.py` |
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

## Our numbers (run 2026-04-26 via `scripts/validate_b2_turpin.py`)

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
