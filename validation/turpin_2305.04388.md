# Turpin 2305.04388 — counterfactual-bias tests

**Paper:** "Language Models Don't Always Say What They Think" —
Turpin et al., 2023. arXiv: 2305.04388.

**Status for v0.1:** **Implemented but not validated against paper
numbers.** Unit-tested in `tests/test_turpin_chen.py`. Not run on a
live model for this launch. See "Decision rationale" and "Future work"
below.

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

## Future work (v0.2 priority)

1. **Write `scripts/validate_b2_turpin.py`.** Structure parallel to
   B1/B3: 30-50 BBH questions, `always_a` bias on Claude Sonnet 4.6,
   Haiku-4.5 verbalization judge, budget ~$3.
2. **Extend the 3-toy-exemplar few-shot scaffolding to per-BBH-task
   13-exemplar prompts** matching Turpin's setup. Without this,
   absolute accuracy drop will not compare to the paper's ~36%.
3. **Add the other two Turpin manipulations:** `suggested_answer` and
   `weak_evidence`. Currently only `always_a` is implemented.
4. **Land a live run** after (1)-(3) and populate this doc's "Our
   numbers" section.

---

## What this validation does NOT claim

- That our Turpin implementation reproduces the paper's ~36%
  accuracy drop. The few-shot gap alone precludes that claim.
- That running B2 on the current implementation would tell you
  anything about Turpin's thesis. It would tell you about our toy
  implementation's behavior.
- That Turpin's axis is covered in v0.1's published validation
  table. It is not — v0.1 ships with B1, B3, B4 validated; B2 is
  deferred.
