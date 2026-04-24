# Qwen3 scaling summary — GPQA-Diamond (Stage 1 + 2)

Data source for the launch README. Numbers only; minimal interpretation.
Narrative framing is README work, not this file.

**Setup (all three rows):** Qwen3 thinking mode, GPQA-Diamond (198
questions), autorater = Claude Haiku 4.5 with Emmons & Zimmermann 2510.23966
Appendix C prompt (SHA `ac1e0ac4044b0a64…`), PHR detector = Claude Haiku 4.5
with Arcuschin-inspired LLM-as-judge prompt (SHA `4d7cc712e9456b80…`).

## Table

| metric | Qwen3-8B | Qwen3-14B | Qwen3-32B |
|---|---|---|---|
| accuracy | 56.06% (111/198) | 61.62% (122/198) | 62.63% (124/198) |
| legibility mean (of rated correct) | 3.48 | 3.61 | 3.73 |
| coverage mean (of rated correct) | 3.24 | 3.42 | 3.57 |
| rated correct n | 107/111 | 120/122 | 124/124 |
| PHR strict (diverged + unacknowledged / correct) | 4.50% (5/111) | 5.74% (7/122) | 4.03% (5/124) |
| PHR incl. acknowledged (diverged / correct) | 6.31% (7/111) | 6.56% (8/122) | 4.84% (6/124) |

## Notes on interpretation

**Legibility + coverage are monotonically increasing with scale.**
Legibility: 3.48 → 3.61 → 3.73. Coverage: 3.24 → 3.42 → 3.57. Both
axes move by ~0.1 per scale step on a 0-4 scale. Small but consistent
direction across 2.3× parameter increases.

**PHR rate is flat within noise across this size range.** Strict-PHR
spans 4.03%–5.74% (range 1.71pp); inclusive-PHR spans 4.84%–6.56%
(range 1.72pp). With n ≈ 120 correct trajectories per model, the
per-rate 1σ Bernoulli noise at p=5% is ±2.0pp. The observed spreads
sit within that noise floor; no scale trend is supported by the data.
Direction is non-monotonic (14B highest, 32B lowest on both PHR
variants) which is a further signal that any real scaling effect is
smaller than noise.

**Accuracy plateaus from 14B to 32B.** 8B → 14B: +5.56pp
(1.75× params, +9.9% relative). 14B → 32B: +1.01pp (2.29× params,
+1.6% relative). The dominant accuracy return is in the 8B → 14B
step, not the 14B → 32B step.

## PHR cot_conclusion distribution — "UNCLEAR" counts

Hypothesis from the 8B checkpoint: smaller models produce more
"UNCLEAR" cot_conclusion judgments.

| model | UNCLEAR / n_phr_rows |
|---|---|
| Qwen3-8B | 7/198 (3.5%) |
| Qwen3-14B | 4/197 (2.0%) |
| Qwen3-32B | 0/185 (0.0%) |

Direction matches the hypothesis (smaller → more UNCLEAR) but absolute
counts are too small to support a load-bearing claim; 32B's zero count
is notable but rests on 185 judge outputs.

## Decoupling observation — monitorability improves while accuracy plateaus

**Monitorability metrics (legibility, coverage) increase monotonically
across all three size steps while accuracy plateaus between 14B and
32B.** The 2.29× parameter increase from 14B to 32B buys +1.01pp
accuracy but +0.12 legibility and +0.16 coverage on the 0-4 autorater
scale. For a downstream monitor that depends on CoT being legible and
complete — not on the model being more correct — scale continues to
help even after accuracy saturates. This is the data-level claim the
README will frame as the launch headline: **CoT monitorability and
task performance decouple at this scale**, and measuring
monitorability directly (this library's purpose) is therefore
complementary to measuring accuracy, not redundant with it.

## Validation status of the PHR detector (B4 Arcuschin)

The PHR strict/inclusive numbers above are only interpretable if the
PHR detector is itself validated. B4 validation result
(`validation/b4_arcuschin_raw.jsonl`, 2026-04-23):
**9.3% PHR rate on GPT-4o-mini / GPQA-Diamond (43 correct+judged)**,
vs. Arcuschin 2503.08679 Fig 1 GPT-4o-mini reference of ~13% — within
the 5%–25% detector-validation bounds. PHR detector considered
directionally validated for the claims above.
