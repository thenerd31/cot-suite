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
