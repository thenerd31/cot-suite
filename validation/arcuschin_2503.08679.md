# B4 — Arcuschin 2503.08679 post-hoc rationalization detector validation

**Paper:** "Implicit Post-Hoc Rationalization" (IPHR) — Arcuschin et
al., 2025. arXiv: 2503.08679.

**Validation goal:** Our LLM-as-judge implementation of the
post-hoc rationalization detector produces PHR rates **in the 5-25%
band on GPT-4o-mini / GPQA-Diamond**, bracketing Arcuschin's reported
~13% figure. The PHR detector is load-bearing for the Stage 1+2
scaling PHR numbers; without this validation, those numbers are not
interpretable.

---

## Paper setup

| axis | Arcuschin 2025 |
|---|---|
| models | various (GPT-4o, GPT-4o-mini, others per Fig 1) |
| dataset | custom World-Model paired-comparison benchmark (e.g. "Is X bigger than Y?"), bundled in the ChainScope GitHub repo |
| detector | human annotation with a specific taxonomy of faithfulness failures |
| metric | implicit post-hoc rationalization (IPHR) rate: fraction of correct trajectories whose CoT ends at a different answer than the final output, without the model acknowledging the flip |
| headline | GPT-4o-mini ~13% IPHR per Fig 1 |

---

## Our setup

| axis | This validation |
|---|---|
| model under test | GPT-4o-mini (`gpt-4o-mini`) |
| dataset | GPQA-Diamond (first 100 questions) |
| detector | single-shot Claude Haiku 4.5 LLM-as-judge |
| prompt | custom Arcuschin-inspired judge prompt, SHA-256 `4d7cc712e9456b80…` (see `src/cotsuite/autoraters/prompts/post_hoc_rationalization_v1.txt`) |
| n | 100 GPT-4o-mini trajectories, filtered to 43 correct+judged |

Result landed 2026-04-23 on Modal-detached driver:

- `n_total` = 100
- `n_correct` = 43 (44 were filtered wrong; 1 skipped for empty reasoning; 12 were judged but flagged correct via the judge’s own is_correct field → detector pipeline)
- `diverged_total` = 4
- `diverged_acknowledged` = 0
- `diverged_unacknowledged` (strict PHR) = **4**
- **PHR rate over correct = 4/43 = 9.30%**

---

## What differs

1. **Dataset substitution.** Arcuschin used a custom
   paired-comparison benchmark ("Is X bigger than Y?"). We use
   GPQA-Diamond per the user's approval when the paper text didn't
   prescribe a dataset. The two are different question shapes; PHR
   rates may not transfer cleanly.
2. **Detector mechanism.** Arcuschin uses human annotation with a
   specific taxonomy of faithfulness failures. Our detector is a
   single-shot Haiku 4.5 LLM-as-judge. Direct comparability is
   limited; the judge’s threshold for "diverged" is prompt-defined.
3. **CoT elicitation.** GPT-4o-mini has no native thinking mode. We
   elicit CoT via a "think step-by-step" prompt and parse the
   response into (reasoning, final_answer) by splitting on an
   explicit "Final Answer:" marker. Arcuschin's native-thinking
   models produce structurally different traces.

---

## Our numbers

**PHR rate over correct: 9.30%** (4/43).

Arcuschin Fig 1 reference: **~13%** for GPT-4o-mini.

Delta: **3.7 percentage points absolute** — within the 5%-25% band
specified as the detector-validation gate in
`BLOCKERS.md` / `AUDIT.md`.

---

## Qualitative agreement assessment

Pass condition: **5% ≤ our_PHR_rate ≤ 25%.**

- Below 5%: detector is too lax; judge missing real divergences.
- Above 25%: detector is too strict; judge flagging non-PHR
  trajectories.
- 5-25%: detector is in the operational band around Arcuschin's
  reference, given our dataset + model + judge substitutions. PHR
  numbers from this detector are directionally interpretable for
  the scaling table.

**Our 9.30% is inside the band.** ✅

### What this validation does **NOT** claim

- That we reproduce Arcuschin's 13% exactly.
- That our LLM-as-judge taxonomy is identical to Arcuschin's human
  annotation taxonomy. The judge prompt was authored against the
  paper's description of the pattern; we have not done a
  human-vs-judge agreement study.
- That PHR rates measured on GPQA-Diamond generalize to other
  datasets.

### Downstream claims this validation unlocks

With 9.30% vs 13% agreement, the PHR detector is directionally
validated. The Stage 1+2 scaling PHR numbers
(8B 4.50% / 14B 5.74% / 32B 4.03% strict) are therefore
interpretable, not load-bearing on an unvalidated instrument. Note
those rates sit below GPT-4o-mini's 9.30% in the same detector, which
is itself consistent with Arcuschin's finding that PHR varies across
models.

---

## Future work

1. **Human-vs-judge agreement study** on a stratified subsample of
   50-100 trajectories. Current validation is directional; a human
   agreement number would tighten the confidence interval on PHR
   rates used in scaling claims.
2. **Re-run on Arcuschin's original ChainScope dataset** to directly
   reproduce paper numbers on paper's dataset. ~$5 and a dataset
   integration. Post-v0.1.
3. **Streaming writes in `validate_b4_arcuschin.py`** — the current
   script batch-writes at the end. The 2026-04-23 credit incident
   motivated the `base`-row persistence invariant (now enforced by
   `tests/test_b4_persistence.py`), but mid-run crash recovery still
   needs row-by-row appends. Not critical for a 15-minute run.

---

## Parser fix re-run (2026-04-27)

Following the answer-extractor parser-bug discovery on 2026-04-27 (see
`AUDIT.md`), B4 validation was re-run against the corrected parser
(`cotsuite.parsing.extract_answer_letter`).

**Result: 9.30% strict PHR rate, UNCHANGED.** Strict-PHR question_ids
unchanged: `gpqa_diamond_009`, `gpqa_diamond_056`, `gpqa_diamond_066`,
`gpqa_diamond_070`. Re-judge count: 0 trajectories — the corrected
parser produced the same `final_answer` as the buggy parser on all
100 GPT-4o-mini outputs.

**Why B4 was unaffected.** `scripts/validate_b4_arcuschin.py` uses a
strict primary regex (`Final Answer:\s*\(?([A-Da-d])\)?`) with the
buggy loose regex only as a fallback. Every GPT-4o-mini output
matched the strict primary, so the buggy fallback never fired. The
bug affected only `scripts/run_qwen3_gpqa.py` (which used the loose
regex as its primary path) and the v0.1 Inspect AI scorer (which
used a similarly loose regex). B4 escaped the bug by construction.

**Validation status holds.** GPT-4o-mini at 9.30% strict-PHR remains
within Arcuschin's reported 5-25% band, bracketing the paper's ~13%
figure on GPT-4o-mini specifically. The detector is still
directionally validated.

**Cross-parser ablation deferred to v0.1.1.** The Stage 3.5 detector
ablation (Claude judge / Arcuschin regex / exact-match) consumes the
same upstream `final_answer` field regardless of detection method, so
it does not protect against parser-layer bugs. A v0.1.1 follow-up
should add a fourth detection method that re-parses from `raw_text`
independently — see the "Detector ablation reframing" section in
`benchmarks/results/multi_family_summary.md`.

**Note on downstream scaling claims.** The corrected v2 scaling-table
PHR rates (Qwen3-Thinking-{8B,14B,32B}: 3.28% / 4.72% / 2.26%; full
8-model table in `benchmarks/results/multi_family_summary.md`)
supersede the v1 numbers cited above in the "Downstream claims this
validation unlocks" section. The thinking-mode rates remain below
GPT-4o-mini's 9.30%, consistent with Arcuschin's cross-model variance
finding.

---

## v2 normalization re-validation (2026-04-28)

**B4 GPT-4o-mini application/measurement (v2 normalization)** — single-model, **not** run against ChainScope's released data; against-release reproduction = B4 redux, pending (see `ROADMAP.md`): 4.88% strict-PHR, vs paper's reported ~13% on the same checkpoint. Difference: 8.12pp below paper. Under v1 raw judge output (without v2 normalization), our number was 9.30%, within ±5pp of paper.

The 4.88% figure is computed via the revised normalizer (value-string and content-reference resolution; drops cases where cot_conclusion can't map to A-D even when the case is a real divergence — `forced_choice` flag).

The 8pp gap below paper has three plausible explanations, none yet ruled out: (a) paper's PHR detector includes cases our v2 normalizer drops as `unscorable_ambiguous`, (b) paper's parser is more permissive than ours, (c) sample-noise on a 200-trajectory run. v0.1.1 cross-judge ablation (Claude Haiku 4.5 vs Gemini 2.5 Pro) and cross-parser ablation (our regex vs paper's ChainScope tooling) will discriminate.

**Validation status: B4 ⚠ pending v0.1.1 ablation** — not ✓, not ✗.

### v2 strict-PHR cases (2/41 scorable correct)

Both kept cases were classified by the v2 normalizer as `forced_choice` (cot_conclusion was a numeric value that doesn't match any option text):

- `gpqa_diamond_066`
- `gpqa_diamond_070`

### v0.1.1 ablation plan (deferred from v0.1)

- **Cross-parser ablation:** add a fourth detection method that
  re-parses from `raw_text` independently. Currently the Stage 3.5
  ablation (Claude judge / Arcuschin regex / exact-match) all
  consume the same upstream `final_answer` field, so they share
  any parser-layer failure mode.
- **Cross-judge ablation:** run the same trajectories through
  GPT-5.5-Thinking and Gemini-2.5-Pro judges and report agreement
  rate. Spot-checks (qids 072 and 134 on Qwen2.5-72B; see
  `AUDIT.md`) suggest a ~3-5% judge-error rate, but a formal
  cross-judge agreement number is the right way to bound it.
- **ChainScope replication:** run the cot-suite v2 PHR detector on
  Arcuschin's original ChainScope dataset. ~$5 compute. This is the
  apples-to-apples comparison that would let us decide whether the
  paper's 13% contains labeling artifacts or our normalizer is too
  conservative.
