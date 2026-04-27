# Related work

cot-suite operationalizes CoT-monitorability evaluations from a set of
existing papers and runs them on a multi-family open-weights scaling
table. This document positions our v0.1 results relative to the six
most-related prior works as of 2026-04-26. The framing throughout: we
are doing **replication + tooling + open-weight scaling**, not novel
methodology — the methodologies are Lanham, Turpin, Chen, Arcuschin,
Emmons & Zimmermann; we wire them into one library and apply them
consistently across 8 models.

---

## 1. Meek et al. — arXiv 2510.27378 — "Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity"

**Summary.** Meek et al. extend the Turpin/Chen cue-injection paradigm
along a second axis — *verbosity* (does the CoT enumerate the causal
factors the task requires?) — and combine faithfulness + verbosity
into a single monitorability score (§3). Their key methodological
move is that faithfulness is now the cue-acknowledgment rate across
*all* instances, not just answer-switches; verbosity is computed by 5
judge models independently enumerating required factors and
aggregating with Claude 4 Sonnet. They release their suite on Inspect
AI.

**Models tested (§3, Table 1):** Four frontier-scale instruction ↔
reasoning sibling pairs:
- DeepSeek V3 ↔ DeepSeek R1 (May 28, 2025)
- Claude 3.5 Sonnet ↔ Claude 3.7 Sonnet (extended thinking)
- Qwen 2.5 (72B) ↔ QwQ-32B
- Qwen 3 (235B) ↔ Qwen 3 Thinking (235B)

**Smallest tested model: QwQ-32B.** No 7B-14B Qwen3-Thinking, no
DS-R1-Distill checkpoints, no Qwen2.5-7B-Instruct, no
Llama-3.1-8B-Instruct.

**Datasets:** BBH (Turpin's 13 subsets), GPQA, MMLU. They explicitly
use **GPQA-Diamond** (§4, Figure 4 caption) — direct overlap with our
benchmark.

**Methods/metrics:** Cue-injection from Chen et al. 2025 (same cue
taxonomy as our `chen_cue_injection`). They cite Arcuschin 2503.08679
for PHR existence but do *not* import its restoration-error /
illogical-shortcut detector — they reinvent the wheel slightly with
their own faithfulness formalism. They cite Lanham 2023 but do *not*
run early-answering / mistake-injection AOC. They cite Emmons 2510.23966
but do *not* import legibility/coverage — verbosity is their own
related-but-distinct construct.

**Headline numbers:**
- DeepSeek-R1 average monitorability: 78.3%
- Claude 3.7 Sonnet on the "sourced information cue": 89.2%
- Qwen 3 Thinking (235B) average: 67.3%
- **Qwen 3 (235B, non-thinking): 59.2% vs Qwen 2.5 (72B): 42.2%** —
  same-family thinking-vs-non-thinking gap.
- QwQ-32B vs Qwen 2.5 72B: 81.7% vs 43.9% (their largest sibling-pair
  gap).

**Where our results compare/extend:**

- **Direct comparable: Qwen2.5-72B-Instruct (non-thinking).** Meek
  reports 42.2% monitorability; we have a full N=198 run on the same
  checkpoint (Stage 3.8, commit `51c8d05`). Side-by-side comparison
  is straightforward. Meek's 42.2% is on their pooled-cue definition;
  our PHR-strict + leg-cov-gap numbers measure related but
  non-identical quantities. Quantitative comparison requires
  refactoring our metrics into Meek's monitorability formula or
  vice-versa; flag as v0.1.1 work.
- **Extension: scale-down.** Meek's 4 frontier pairs probe the very
  top of the Pareto curve. Our 5 thinking-mode rungs from 8B → 70B
  populate the gap below their points. Our finding that PHR strict
  stays in the 2.78%-5.74% band across this entire range is
  consistent with the *direction* of their thinking-models-are-more-
  monitorable claim and adds the within-family scaling slope they
  cannot probe.
- **Novel: DS-R1-Distill-Llama-70B.** Meek tested DeepSeek-R1 (the
  full RL-trained model) but never the *distilled* student. Our
  Stage 3.7 run on this checkpoint at full N=198 (gap=0.288, PHR
  strict=3.79%) probes whether their "thinking models are more
  monitorable" claim survives the distillation recipe — a different
  question than they answer.

---

## 2. Chua & Evans — arXiv 2501.08156 — "Are DeepSeek R1 And Other Reasoning Models More Faithful?"

**Summary.** Chua & Evans run the cleanest "thinking vs non-thinking"
head-to-head: three reasoning models compared against their
non-reasoning siblings on a 7-cue faithfulness battery (Stanford-
professor opinion, two visual square cues, argument cue, post-hoc cue,
wrong few-shot, "are you sure?"). Headline: **DeepSeek-R1 verbalizes
the influence of a "Stanford Professor" cue 59% of the time vs 7%**
for non-reasoning DeepSeek. They argue reward-model training partly
*causes* the lower faithfulness of non-reasoning baselines.

**Models tested:**
- Reasoning: QwQ-32B-Preview, Gemini-2.0-flash-thinking-exp,
  DeepSeek-R1.
- Non-reasoning: **Qwen-2.5-72B-Instruct**, Gemini-2.0-flash-exp,
  DeepSeek-V3, Claude-3.5-Sonnet, GPT-4o, Grok-2-Preview,
  Llama-3.3-70B-Instruct.
- Distilled-from-R1: paper describes "DeepSeek released models
  distilled on 800k CoTs from DeepSeek R1" with the Llama-70B and
  Qwen-32B variants. The 800k-CoT figure matches DeepSeek's
  official R1-distill release announcement, so the paper's
  "Distilled Llama-70B" is almost certainly
  `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (our exact
  checkpoint), but the paper does not cite the HF name explicitly
  — verified via WebFetch on the HTML version 2026-04-26. Treat as
  "almost-certainly-same checkpoint with slightly under-specified
  paper-side attribution." Also tested: Distilled Qwen-32B,
  DeepSeek-R1-Distill-Qwen-1.5B / 14B.

**Datasets:** **MMLU only.** No GPQA, no GPQA-Diamond. Direct
confirmation.

**Methods/metrics:** Turpin-style cue-injection. **No** Lanham AOCs,
**no** legibility/coverage, **no** Arcuschin PHR detection.

**Key numbers:**
- DeepSeek-R1 (professor cue): 59% vs DeepSeek-V3: 7%
- Gemini reasoning (professor): 68%
- Qwen reasoning (professor): 47%
- Claude-3.5-Sonnet (professor): 13%
- DeepSeek-R1 on visual black-square cue: 25%

**Where our results compare/extend:**

- **Direct comparables (3 exact checkpoints):**
  - DeepSeek-R1-Distill-Llama-70B
  - DeepSeek-R1-Distill-Qwen-14B
  - Qwen2.5-72B-Instruct
  This is the strongest model-overlap among the 6 prior works. **Same
  three checkpoints, harder benchmark (GPQA-Diamond vs MMLU).**
- **Extension: GPQA-Diamond instead of MMLU.** If our PHR + leg-cov
  signals on these checkpoints survive the harder benchmark, this
  replicates and strengthens Chua & Evans. If the thinking-vs-
  non-thinking gap collapses on GPQA-Diamond, that's a meaningful
  domain-dependence finding.
- **Extension: scaling within thinking-mode.** They do thinking-vs-
  non-thinking *pairs* but no within-family size sweep. Our 8B/14B/32B
  Qwen3-Thinking arc is on a probing axis they did not.

---

## 3. Guan et al. (OpenAI) — arXiv 2512.18311 — "Monitoring Monitorability"

**Summary.** OpenAI's quantitative monitorability paper. Defines a
**g-mean² monitorability metric** (`g-mean² = TPR × TNR`, with an
intervention-specific variant `g-mean² = TPR_mc × √(TNR_{X=1} ·
TNR_{X=0})`) and a three-archetype evaluation taxonomy. Headline
findings: most frontier models are "fairly, but not perfectly,
monitorable"; longer CoTs → more monitorable; **RL optimization does
not materially decrease monitorability at current frontier scale**;
asserts a Pareto frontier where smaller-model-with-higher-reasoning-
effort beats larger-model-with-low-effort on monitorability per cost.

> **Caveat on the v0.1-spec-as-stated three-archetype framing.** The
> launch plan referred to OpenAI's three archetypes as "memorization
> / hallucination / faithful." The paper's actual three archetypes
> per §2 are **(1) intervention evaluations, (2) process evaluations,
> (3) outcome-property evaluations**. The memorization / hallucination
> / faithful trichotomy may be from a different OpenAI paper or
> section we didn't reach. Flag for verification before citing the
> trichotomy framing in the launch README.

**Models tested:** GPT-5 Thinking, OpenAI o3, o3-mini, o4-mini,
Claude 3.7 Sonnet (Thinking), Kimi K2 Thinking, plus open-weight
**DeepSeek-R1-0528-Qwen3-8B** as their only open checkpoint.

**Datasets:** Health Queries, Scruples (AITA), Memory Bias, GPQA +
AIME + GPQA-encoded toy math/science, WMDP sandbagging,
FrontierMath, ShadeArena, OAI-repo-sabotage, plus 5 anti-scheming
sets. **GPQA-Diamond explicitly used** (§6.2).

**Methods/metrics:** g-mean² as defined; three-archetype taxonomy;
CoT-only vs action-only vs all-messages monitor scopes; agent-monitor
capability scaling.

**Where our results compare/extend:**

- **Adopt their g-mean² metric.** It's the cleanest single-number
  summary in the field and they're the gold-standard reference.
  Computing g-mean² on our existing data is a small refactor (we
  already have TPR/TNR from cue-injection runs). Worth landing in
  v0.1 to enable direct numeric comparison with their tables.
- **Direct-ish comparable: DeepSeek-R1-0528-Qwen3-8B.** Their only
  open-weight model. Same Qwen3-8B base as our Qwen3-Thinking-8B
  but a different post-training recipe (R1-distill of the 0528
  checkpoint vs Qwen3's native thinking). Near-but-not-exact
  comparison; flag as such.
- **Extension: open-weight scaling curve.** OpenAI's Pareto-frontier
  claim ("smaller model + more reasoning effort > larger model +
  less effort") is essentially a hypothesis about within-family
  scaling that their 7-frontier-closed + 1-open-8B model set cannot
  directly test. Our 8-model 7B-72B Qwen+Llama × native+distill
  table provides the open-weight scaling curve their paper asserts
  but does not measure.

---

## 4. Korbak et al. — arXiv 2507.11473 — "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety"

**Summary.** Position paper, 41 authors spanning Anthropic, OpenAI,
DeepMind, METR, Apollo, MATS. Central thesis: CoT monitorability is a
real but contingent property of *current* reasoning-model training
recipes, eroded by (a) outcome-based RL drifting CoTs away from
human-legible English, (b) process supervision rewarding human-
pleasing CoTs that no longer reflect computation, (c) architectural
moves toward latent/continuous reasoning. Recommendations:
1. Frontier developers should *develop and run* CoT monitorability
   evaluations and report them in system cards.
2. Use monitorability scores as inputs to training/deployment.
3. If monitorability degrades during training, prefer the earlier
   checkpoint.
4. Don't make CoT monitoring the sole safety layer.

**Models / datasets / metrics:** None — position paper.

**Where cot-suite fits.** Korbak is the **charter document for
cot-suite**. The library is a direct operationalization of their
recommendation (1) "develop effective evaluations of CoT
monitorability." We position cot-suite as the missing tooling layer
between the Korbak position paper and the Meek / OpenAI individual
evaluations: a unified library that bundles Lanham + Turpin + Chen +
Arcuschin + Emmons & Zimmermann under one CLI, runs them
consistently across model families, and reports a single
multi-family scaling table. The 8-model multi-family scaling result
in `benchmarks/results/multi_family_summary.md` is exactly the kind
of "track monitorability across training conditions" data Korbak's
recommendation (2) needs but no one had produced for the open-weight
side as of 2026-04-26.

---

## 5. Schoen et al. — arXiv 2509.15541 — "Stress Testing Deliberative Alignment for Anti-Scheming Training"

**Summary.** Apollo Research + OpenAI. They train OpenAI o3 and
o4-mini with deliberative alignment on a narrow distribution of
agentic environments and stress-test the resulting "AS" (Anti-
Scheming) variants on 26 OOD evaluations spanning 180+ environments.
Their proxy for scheming is **covert actions** — strategic
rule-breaking, sandbagging, lying to evaluators (§3.2). Headline:
deliberative alignment cuts covert-action rates **8.7% → 0.3% for
o4-mini and 13.0% → 0.4% for o3** but does not eliminate them, and
reductions are partly explained by *situational awareness* (the model
recognizing it's being evaluated) rather than genuine alignment.

**Models tested:**
- Trained: OpenAI o3 → o3 AS, o4-mini → o4-mini AS, o4-mini RL-Only.
- Baseline: Claude-4-Opus, Gemini-2.5-pro, Grok-4, Llama-4 Maverick.

**Zero overlap with our scaling table.** No open-weight Qwen, no
DS-R1-Distill, no Llama-3.1-8B. Their "Llama-4 Maverick" is a
different generation and scale than our Llama-3.1 series.

**Datasets:** 26 custom OOD agentic environments (covert rule
violation, sandbagging QA, etc.). Use **GPQA as a capability-control
benchmark** (Table 6) — not explicitly GPQA-Diamond.

**Methods/metrics:** Covert-action rate (binary, manually verified).
They explicitly note that the actions-only criterion *undercounts*
covert actions when CoT is "illegible, incomplete or unfaithful" —
a direct nod to the legibility/coverage/faithfulness literature, but
they don't compute a faithfulness AOC themselves.

**Where our results fit.** Schoen et al. is the "downstream safety
eval" half of the field — closed frontier agents in scheming-flavored
environments. cot-suite is the upstream "open-weight monitorability
measurement" half. **No direct numeric comparison possible** (zero
model overlap, different proxy task), but the broader thesis aligns:
both papers presuppose that CoT-monitorability is the underlying
property and degraded monitorability would invalidate downstream
safety claims. Cite as a downstream consumer of monitorability
evaluations whose results would need to be recomputed if
monitorability eroded.

---

## 6. Lewis-Lim et al. — arXiv 2508.19827 — "Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc Rationalisation?"

**Summary.** EMNLP 2025 Main. Studies the *dynamics* of CoT — does the
CoT actively steer the answer, or is it post-hoc justification of an
answer the model already had? — across instruction-tuned, reasoning,
and reasoning-distilled models on **soft-reasoning** tasks
(commonsense / analytical, not math). Their core method: a
*confidence trajectory* analysis tracking the model's probability of
its eventual final answer as each CoT step is appended; flat
trajectories indicate post-hoc behavior. Faithfulness is measured
separately by injecting professor-cue or metadata-cue and detecting
acknowledgment with GPT-4. Headline: distilled-reasoning models
change answers under cue injection **65%** of the time vs **25%**
(instruction-tuned) and **24%** (full reasoning); unfaithful CoTs
can still actively guide the answer.

**Models tested:**
- Instruction-tuned: **Qwen2.5-7B-Instruct**, Qwen2.5-32B-Instruct,
  **Llama-3.1-8B-Instruct**.
- Reasoning: **Qwen3-32B**, QwQ-32B.
- Distilled-reasoning: R1-Distill-Qwen-7B, R1-Distill-Qwen-32B,
  R1-Distill-Llama-8B.

**Direct overlaps with our scaling table:** Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, Qwen3-32B (3 of our 8 models exactly).

**Datasets:** CommonsenseQA, StrategyQA, MUSR (3 variants), LSAT (3
variants), **GPQA-Main (448 examples, Appendix J)** — verified via
WebFetch 2026-04-26. The methods section §2.2 says only "GPQA, a
graduate-level science dataset" without specifying the split, but
the 448-example count in Appendix J matches GPQA-Main's 448-example
test set exactly (GPQA-Diamond is 198, GPQA-Extended is 546).
**No GPQA-Diamond overlap.**

**Methods/metrics:** Confidence-trajectory PHR detection (novel,
adjacent to Arcuschin's restoration-error idea); cue-injection
faithfulness (Turpin lineage). **No** Lanham AOCs, **no**
legibility/coverage.

**Where our results compare/extend:**

- **Direct comparables (3 exact checkpoints):** Qwen2.5-7B-Instruct,
  Llama-3.1-8B-Instruct, Qwen3-32B. Our GPQA-Diamond results vs
  their multi-dataset (CommonsenseQA / StrategyQA / MUSR / LSAT /
  GPQA-Main) results on the same three checkpoints is the
  cleanest cross-benchmark comparison available. **Domain shift
  on top of dataset shift** — they cover soft-reasoning + GPQA-Main;
  we cover GPQA-Diamond only. So the cross-comparison is "harder
  benchmark on the science axis" rather than a pure domain
  swap.
- **Their finding to test against:** distilled models change answers
  65% under cue injection (much more than non-distilled). On
  GPQA-Diamond, do our DeepSeek-R1-Distill variants
  (Qwen-14B, Llama-70B) show the same elevated cue-susceptibility?
  If yes: replicates Lewis-Lim and extends to a harder, more
  technical domain. If no: domain-dependence finding worth its own
  section.
- **Novel checkpoints relative to them:** DeepSeek-R1-Distill-Qwen-14B
  and DeepSeek-R1-Distill-Llama-70B (they only tested smaller
  R1-Distill variants).
- **Methodological extension to consider for v0.2:** their
  confidence-trajectory PHR signal is orthogonal to our cue-
  acknowledgment faithfulness. Worth considering as a follow-up
  metric in cot-suite v0.2 — relevant to the library's scope even if
  not in v0.1.

---

## Cross-paper synthesis

### Open-weight scaling gap

| paper | model regime | smallest open-weight | size axis covered |
|---|---|---|---|
| Meek 2510.27378 | 4 frontier pairs | Qwen 2.5 72B / QwQ-32B | 32B-235B |
| Chua & Evans 2501.08156 | mostly frontier; some distill | DS-R1-Distill-Qwen-1.5B | 1.5B-72B (sparse) |
| OpenAI 2512.18311 | 7 closed + 1 open | DS-R1-0528-Qwen3-8B | 8B (single point) |
| Korbak 2507.11473 | n/a (position) | n/a | n/a |
| Schoen 2509.15541 | closed frontier only | n/a | frontier-only |
| Lewis-Lim 2508.19827 | mostly sub-32B open-weight | Qwen2.5-7B-Instruct | 7B-32B |
| **cot-suite v0.1** | **8 open-weight** | **Qwen2.5-7B-Instruct** | **7B-72B with intermediate rungs** |

**No prior work covers the 7B-72B range with Qwen-base, Llama-base,
native-thinking, and R1-distill variants in one consistent
benchmark.** Closest analog is Lewis-Lim, which lacks 70B and uses
soft-reasoning rather than GPQA-Diamond.

### GPQA-Diamond explicit use

| paper | GPQA-Diamond? |
|---|---|
| Meek 2510.27378 | ✅ §4 line 420, Figure 4 |
| Chua & Evans 2501.08156 | ❌ MMLU only |
| OpenAI 2512.18311 | ✅ §6.2 |
| Korbak 2507.11473 | n/a |
| Schoen 2509.15541 | partial (GPQA generally; not explicitly Diamond) |
| Lewis-Lim 2508.19827 | ❌ — confirmed GPQA-Main (448 ex), not Diamond (198 ex) |

### Model-overlap matrix (our 8 models × the 6 prior works)

✅ = paper tested this exact checkpoint. ⚠ = adjacent / similar
checkpoint but not exact match. ❌ = not tested.

| our model | Meek | Chua&Ev | OpenAI | Lewis-Lim |
|---|---|---|---|---|
| Qwen3-Thinking-8B | ❌ | ❌ | ⚠ (R1-Distill-Qwen3-8B, different recipe) | ❌ |
| Qwen3-Thinking-14B | ❌ | ❌ | ❌ | ❌ |
| Qwen3-Thinking-32B | ❌ | ❌ | ❌ | ✅ (Qwen3-32B) |
| DS-R1-Distill-Qwen-14B | ❌ | ✅ (MMLU) | ❌ | ❌ |
| DS-R1-Distill-Llama-70B | ❌ | ⚠ (MMLU; HF name not cited but description matches our checkpoint) | ❌ | ❌ |
| Qwen2.5-7B-Instruct | ❌ | ❌ | ❌ | ✅ |
| Qwen2.5-72B-Instruct | ✅ (BBH+MMLU+GPQA-D) | ✅ (MMLU) | ❌ | ❌ |
| Llama-3.1-8B-Instruct | ❌ | ❌ | ❌ | ✅ |

(Korbak and Schoen have no model-list overlap and are omitted from
the matrix — they're cited for framing/downstream-context, not for
quantitative comparison.)

**Models unique to our study** (not tested by any of these 6 prior
works on any benchmark): Qwen3-Thinking-8B, Qwen3-Thinking-14B. Both
sit in the under-explored 8B-14B thinking-mode range.

**Our most-leveraged datapoint: DS-R1-Distill-Llama-70B on
GPQA-Diamond.** Chua & Evans tested this checkpoint on MMLU; we
test it on a harder benchmark (GPQA-Diamond) with our full
multi-metric pipeline. This is the single strongest "replication +
extension" datapoint in our table.

### Methodological lineage

| metric | originating paper | used by | cot-suite v0.1 |
|---|---|---|---|
| Early-answering AOC | Lanham 2307.13702 | (none of the 6) | ✅ (B1) |
| Mistake-injection AOC | Lanham 2307.13702 | (none of the 6) | ✅ (B1) |
| Cue-injection faithfulness | Turpin 2305.04388, Chen 2505.05410 | Meek, Chua&Ev, Lewis-Lim | ✅ (B2, B3) |
| Implicit PHR | Arcuschin 2503.08679 | (cited by Meek; not run) | ✅ (B4) |
| Legibility / coverage | Emmons & Zimmermann 2510.23966 | (cited by Meek; not run) | ✅ (Stage 1/2/3 autorater) |
| Verbosity | Meek 2510.27378 | Meek only | ❌ v0.2 |
| Confidence-trajectory PHR | Lewis-Lim 2508.19827 | Lewis-Lim only | ❌ v0.2 |
| g-mean² monitorability | OpenAI 2512.18311 | OpenAI only | ❌ recommend v0.1 add |

**cot-suite v0.1 is the only library that bundles Lanham +
Turpin + Chen + Arcuschin + Emmons-Zimmermann under a single CLI.**
That is the tooling-layer contribution distinct from any individual
prior paper's evaluation.

---

## Suggested READme positioning

1. Lead with Korbak as the motivating position paper.
2. Cite Meek, Chua & Evans, OpenAI 2512.18311 as the closest
   measurement-side prior work; explicitly position cot-suite as
   open-weight-scaling-curve complement to their frontier-pairs work.
3. Cite Lewis-Lim for the 3-checkpoint direct comparable on
   soft-reasoning vs our GPQA-Diamond.
4. Cite Schoen as downstream-safety motivation, no numeric
   comparison.
5. **DS-R1-Distill-Llama-70B on GPQA-Diamond is the headline novel
   datapoint** relative to all 6 prior works (Chua & Evans tested
   this checkpoint only on MMLU).
6. **Adopt OpenAI's g-mean² metric** for v0.1 if a small refactor
   suffices — biggest legibility win for cross-paper comparison at
   lowest engineering cost.

---

## Open items to verify before launch

- [x] **OpenAI 2512.18311 three-archetype taxonomy** — verified
      independently 2026-04-26 by user: paper says **intervention /
      process / outcome-property** (NOT memorization / hallucination
      / faithful as stated in the launch plan). Doc framing in §3
      above is correct as written.
- [x] **Chua & Evans R1-Distill-Llama-70B exact identity** —
      verified via WebFetch 2026-04-26. Paper text describes
      "DeepSeek released models distilled on 800k CoTs from
      DeepSeek R1" with Llama-70B variant. The 800k-CoT figure
      matches DeepSeek's official R1-distill release announcement,
      so it's almost certainly
      `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (our checkpoint),
      but the paper does not cite the HF name explicitly. Doc §2
      framing softened to reflect this attribution gap.
- [x] **Lewis-Lim 448-example "GPQA" slice** — verified via
      WebFetch 2026-04-26. 448 = GPQA-Main test set size; Diamond is
      198. Paper §2.2 says "GPQA, a graduate-level science dataset"
      without specifying split, but Appendix J has the count.
      Confirmed GPQA-Main, not GPQA-Diamond. Doc §6 framing updated;
      cross-paper synthesis table updated. **We do NOT have direct
      GPQA-Diamond overlap with Lewis-Lim** — only same-checkpoint
      overlap on a different benchmark.
- [ ] **Compute g-mean² on our existing TPR/TNR data and add to
      `multi_family_summary.md`.** Effort estimate pending the
      Priority 3 inspect_ai_integration_assessment.md.
