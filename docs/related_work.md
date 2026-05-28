# Related work

cot-suite operationalizes CoT-monitorability evaluations from a set of
existing papers and runs them on a multi-family open-weights scaling
table. This document positions our v0.1 results relative to the eight
most-related prior works as of 2026-04-27, plus a recent-literature
sweep covering October 2025–April 2026, and (Updated 2026-05-28) a
March–May 2026 addendum covering the Young classifier-sensitivity
trilogy, OpenAI's monitorability-evals release, and the external-review
precedent for frontier-lab monitorability claims. The framing
throughout: we are doing **replication + tooling + open-weight
scaling**, not novel methodology — the methodologies are Lanham,
Turpin, Chen, Arcuschin, Emmons & Zimmermann; we wire them into one
library and apply them consistently across 8 models.

> **Updated 2026-05-28.** New material is concentrated in the dated
> sub-sections of §3, §4, §7, §8, and in new sections §12 (Young
> trilogy), §13 (external review of frontier-lab monitorability
> claims), and §14 (commercial agent-oversight). The original §1-11
> treatment is preserved; only the dated insertions are new. The
> "Phase-6" label used below refers to the Inspect-native interop work
> tracked as v0.2 / v0.2.1+ in `ROADMAP.md` (Inspect Scorers for the
> remaining four metrics + an Inspect-native dataset loader); it is a
> naming convenience for the cross-tool-interop milestone, not a new
> scope beyond what the roadmap already commits to.

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
  distilled on 800k CoTs from DeepSeek R1" with Llama-70B and
  Qwen-32B variants (Section 4 / Appendix A.1). **The paper does
  not specify a HuggingFace model identifier, snapshot date, or
  commit SHA for the Llama-70B distilled model** — verified via
  WebFetch of the v3 HTML on 2026-04-27. Whether their Llama-70B is
  the canonical `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` or a
  different distillation cannot be determined from the paper. Our
  DS-R1-Distill-Llama-70B GPQA-Diamond run is on the canonical HF
  checkpoint regardless — so for our positioning purposes, our
  exact-named checkpoint is unambiguously novel. Also tested:
  Distilled Qwen-32B, DeepSeek-R1-Distill-Qwen-1.5B / 14B (same
  identity ambiguity).

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

- **Direct comparables (2 exact + 1 ambiguous):**
  - Qwen2.5-72B-Instruct (exact match — explicitly named)
  - DeepSeek-R1-Distill-Qwen-14B (exact match — explicitly named)
  - "Distilled Llama-70B" (ambiguous — paper does not specify HF
    identifier, see above). Our run is on the canonical
    `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`.
  Two exact-checkpoint matches plus one underspecified-attribution
  match. **Harder benchmark (GPQA-Diamond vs MMLU)** for all three.
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

The three archetypes per §2 are **(1) intervention evaluations,
(2) process evaluations, (3) outcome-property evaluations** —
verified directly from the paper (see open-items §c).

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

- **Adopt their g-mean² metric for v0.1.1** — pending the
  binarization-scheme + ground-truth-proxy methodology decision
  flagged in `docs/inspect_ai_integration_assessment.md`. Computing
  g-mean² on our existing v0.1 data is **not** a small refactor: it
  requires picking a threshold to binarize the 0-4 Likert autorater
  scores into TPR/TNR inputs and a ground-truth proxy for monitor
  flagging. The assessment doc estimates 4-8h methodology design +
  2-4h implementation. Defer until methodology is locked.
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

### Updated 2026-05-28: OpenAI monitorability-evals release (April 2026)

OpenAI released the companion evaluation harness for 2512.18311 as a
standalone open-source project in April 2026, alongside a write-up on
their alignment blog:

- **Blog:** `https://alignment.openai.com/monitorability-evals/`
- **Repo:** `github.com/openai/monitorability-evals` — **verified**
  2026-04-22 creation date, Apache-2.0 license, 76 stars (as checked
  for this addendum). This is the reference implementation the
  `docs/inspect_ai_integration_assessment.md` g-mean² section already
  points to (`run_eval_scaffold_lib.py`).

**g-mean² mechanics, expanded.** Reading the released scaffold makes
the metric's structure explicit:

- The reported quantity is **g-mean² = TPR × TNR**, the product of a
  true-positive rate and a true-negative rate for a binary monitor
  evaluated against a binary ground-truth proxy. (This matches the
  formula already extracted into the integration assessment:
  `gmean2 = tpr * tnr` over `(arm × correctness × monitor signal)`
  triples.)
- **TNR is a geometric mean across intervention arms**, not a single
  pooled rate — the intervention-specific variant
  `√(TNR_{X=1} · TNR_{X=0})` in the paper generalizes to a geo-mean
  over the arm set in the scaffold, so a monitor must keep its
  false-positive rate low across *every* arm to score well.
- **TPR is reported as a minimal-criterion upper bound** — it credits
  the monitor whenever it satisfies a minimal flagging criterion, so
  the published TPR is an optimistic ceiling rather than a strict
  point estimate. The practical consequence is that g-mean² is
  primarily disciplined by the TNR (specificity) term.
- Uncertainty is quantified with a **bootstrap**, and arms pass
  through a **cross-fit eligibility filter** before entering the
  metric: a selection fraction `selection_frac=0.6` partitions data
  for the eligibility decision, and a Wald multiplier `wald_k=1.96`
  (the ~95% normal critical value) sets the eligibility band. Arms
  that fail the filter are excluded from the geo-mean rather than
  dragging the score toward an artifact.

**The three archetypes, named.** The scaffold organizes evaluations
into the same three archetypes the paper defines (§2), which is the
taxonomy cot-suite's Phase-6 interop should map onto:

1. **Intervention evaluations** — inject a decision-relevant factor
   (an arm `X`) and check whether the monitor flags trajectories where
   the factor was used. This is the archetype closest to the
   Turpin/Chen cue-injection lineage cot-suite already wraps (B2/B3).
2. **Process evaluations** — score whether the CoT process itself is
   legible/complete enough to monitor, independent of any single
   injected factor. This is the archetype closest to cot-suite's
   Emmons-Zimmermann legibility/coverage autorater.
3. **Outcome-property evaluations** — check a property of the outcome
   (e.g., sandbagging, a withheld capability) against what the CoT
   reveals.

**Phase-6 Inspect-native interop.** cot-suite's Inspect-native
scorers (v0.1 ships `cot_legibility_coverage` and
`cot_post_hoc_rationalization`; the remaining four metrics land in
v0.2 per `ROADMAP.md`) put cot-suite and `monitorability-evals` on the
same substrate at the data-contract level: both can read/produce
Inspect-style `(arm, correctness, monitor-signal)` triples. The
Phase-6 goal is to emit our binarized monitor signals in a shape that
`monitorability-evals`'s `bootstrapped_gmean_metric` can consume
directly, so that a cot-suite run can be scored under OpenAI's exact
g-mean² implementation rather than a reimplementation. This is the
cleanest path to a like-for-like comparison and supersedes
hand-rolling g-mean² internally — but it is still gated on the same
binarization-scheme + ground-truth-proxy decision flagged above and in
the integration assessment. No g-mean² number is claimed for v0.1.

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

**Endorsers / signatories (Updated 2026-05-28).** Beyond the
multi-lab author list, the paper carries a prominent set of endorsers
that signals how broad the cross-lab buy-in is — including **Geoffrey
Hinton, Ilya Sutskever, John Schulman, and Samuel Bowman**. This is
the umbrella conceptual frame cot-suite operates *inside*: the library
is not arguing that monitorability matters (that case is made here by
a who's-who of the field); it is supplying the measurement tooling the
position paper's recommendation (1) calls for.

**Provenance note (Updated 2026-05-28).** Lead author **Tomek Korbak
is now at OpenAI** (previously UK AISI). This is worth flagging only
because several adjacent papers in §9 (Lee–Chen–Korbak 2602.22303,
the CoT-Control authorship in §7, the deployment-context papers in
§10) share Korbak as a co-author across institutional moves; the
affiliation drift does not change the position paper's standing as the
charter document.

**Where cot-suite fits.** Korbak is the **charter document for
cot-suite**. The library is a direct operationalization of their
recommendation (1) "develop effective evaluations of CoT
monitorability." We position cot-suite as the missing tooling layer
between the Korbak position paper and the Meek / OpenAI individual
evaluations: the **first PyPI-installable bundle of Lanham
2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin 2503.08679,
and Emmons & Zimmermann 2510.23966 under a unified CLI**, run
consistently across model families and reported as a single
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

## 7. Yueh-Han Chen et al. — arXiv 2603.05706 — "Reasoning Models Struggle to Control their Chains of Thought"

**Summary.** Anthropic + UK AISI + MATS collaboration (Yueh-Han Chen,
McCarthy, Lee, He, Kivlichan, Baker, Carroll, Korbak — March 2026).
Introduces **CoT-Control**, an evaluation suite that asks not "is the
CoT faithful?" but "*can the model intentionally control what its
CoT verbalizes?*" along three axes: **Information Suppression**
(omit a fact the model uses), **Information Addition** (verbalize
content the model does not actually use), and **Stylistic Control**
(format / tone / language). The threat model is steganographic /
adversarial: a deceptively-aligned model that wants to act on
information without exposing it in CoT. Headline numbers: Claude
Sonnet 4.5 manages 2.7% successful CoT control vs 61.9% on output-
only control (gap of ~59pp); DeepSeek R1 manages 0.1% on CoT, the
weakest of all tested models. **Conclusion: current frontier
reasoning models cannot reliably hide reasoning content in CoT even
when explicitly instructed to** — a *positive* monitorability
finding, complementary to Korbak's "fragile opportunity" thesis.

**Models tested:** Closed frontier (Claude Sonnet 4.5, GPT-5, Gemini
2.5 Pro, etc.) plus open-weight **Qwen3-14B**, **Qwen3-32B**, and
DeepSeek R1.

**Direct overlaps with our scaling table:** Qwen3-14B and Qwen3-32B
(both base models for our Qwen3-Thinking-14B / Qwen3-Thinking-32B
checkpoints — same weights, same family, although CoT-Control runs
them via Anthropic API / OpenRouter rather than local Modal).

**Datasets:** Custom CoT-Control task suite — synthetic prompts
designed to induce specific CoT/output content gaps. Not GPQA,
not MMLU.

**Methods/metrics:** Per-controllability-type success rate
(binary: did the model successfully suppress / add / restyle CoT
content under the instruction). LLM-as-judge for content
verification. Code: https://github.com/YuehHanChen/CoTControl
(single-paper companion repo, no PyPI distribution).

**Where our results compare/extend:**

- **Different threat model, complementary finding.** CoT-Control
  measures *intentional* unfaithfulness ("can the model do it on
  purpose?"); cot-suite measures *structural* unfaithfulness
  ("does the CoT track the computation by default?"). A model
  that scores low on CoT-Control (cannot intentionally hide CoT
  content) *and* low on PHR-strict (does not unconsciously
  rationalize) is doubly monitorable. Our PHR-strict ≤5.74% on
  Qwen3-Thinking-14B/32B + their <5% intentional-control success
  on Qwen3-14B/32B compose into the same conclusion: **at this
  scale, in this family, monitorability is robust under both
  unintentional and intentional pressure.**
- **Direct co-authorship overlap on the open-weight scaling
  axis.** Korbak (Korbak 2507.11473) and Baker (named on the
  March 2026 monitoring-obfuscation work 2503.11926) appear on
  CoT-Control's author list. Their next-paper choice of Qwen3-14B/
  32B as the open-weight test points validates the model selection
  for our scaling table — those are the right open checkpoints to
  probe in 2026.
- **Lineage note.** CoT-Control does not implement Lanham AOCs,
  Turpin / Chen cue-injection, Arcuschin PHR, or Emmons-Zimmermann
  legibility/coverage. It is a sibling evaluation in the
  monitorability portfolio, not a competitor to cot-suite's bundle.

### Updated 2026-05-28: CoT-Control as a Phase-6 stress-test integration target

CoT-Control is the **intentional-concealment stress test** cot-suite's
Phase-6 interop should be able to drive. Its task suite is a clean fit
for an Inspect-native loader: synthetic prompts with a known
suppress/add/restyle instruction and an LLM-judge verification step
map directly onto an Inspect Task (Solver applies the control
instruction) + Scorer (judge checks whether the CoT complied)
structure — the same Solver+Scorer pair pattern the ROADMAP already
specifies for the Turpin and Chen ports. Integrating CoT-Control as a
stress-test dataset would let cot-suite report *structural*
faithfulness (Lanham/Turpin/Chen/Arcuschin/Emmons) and *intentional*
controllability (CoT-Control) on the **same open-weight checkpoints**
(Qwen3-14B/32B already overlap), under one CLI. This is an integration
target, not a v0.1 claim: no CoT-Control numbers are produced by
cot-suite today.

---

## 8. ASTRAL Group — arXiv 2603.28590 — "MonitorBench: A Comprehensive Benchmark for Chain-of-Thought Monitorability in Large Language Models"

**Summary.** A 1,514-instance benchmark across 19 tasks in 7
categories, designed to characterize when CoT can be used to
monitor decision-critical factors driving model behavior. Two
**stress-test settings** are layered on top of the base instances:
*direct concealment* (model instructed to hide the decision-
critical factor) and *monitor-aware evasion* (model instructed to
defeat a downstream CoT monitor). Like CoT-Control, this is an
*intentional* unfaithfulness probe; unlike CoT-Control, it is
positioned as a stand-alone benchmark with curated instances rather
than a methodology evaluated on existing tasks. Code:
https://github.com/ASTRAL-Group/MonitorBench (15 stars, 10 commits,
MIT license).

**Affiliation (Updated 2026-05-28).** ASTRAL-Group is a UIUC / UW /
UCSD collaboration.

**Three dimensions (Updated 2026-05-28).** MonitorBench organizes its
base measurement along three monitorability dimensions, which align
closely with OpenAI's three archetypes (§3) and are therefore the
natural mapping target for a shared Phase-6 data contract:
1. **Input Intervention** — was a decision-critical input factor used
   and is it surfaced in the CoT? (≈ OpenAI's *intervention*
   archetype; ≈ the Turpin/Chen cue-injection axis cot-suite wraps.)
2. **Outcome Justification** — does the CoT justify the actual
   outcome? (≈ OpenAI's *outcome-property* archetype.)
3. **Solution Process** — is the reasoning process itself
   legible/complete enough to monitor? (≈ OpenAI's *process*
   archetype; ≈ cot-suite's legibility/coverage autorater.)

**Two stress tests (Updated 2026-05-28).** *Direct Concealment*
(instruct the model to hide the decision-critical factor) and
*Monitor-aware Evasion* (instruct the model to defeat a downstream CoT
monitor) — the same intentional-concealment family as CoT-Control's
suppression axis.

**Models tested:** Per repo + abstract — frontier closed + a
selection of open-weight reasoning models. Not enumerated in the
abstract; full list in §3 of the paper.

**Datasets:** 1,514 novel instances (synthetic + curated), not
sourced from MMLU / GPQA / BBH / Turpin's BBH subsets.

**Methods/metrics:** Stress-test concealment success rate plus
monitor-detection rate, computed under the two stress settings.
**Does not implement** Lanham AOCs (early answering / mistake
injection / paraphrasing / filler tokens), Turpin counterfactual
bias, Chen six-cue injection, Arcuschin IPHR pair construction, or
Emmons-Zimmermann legibility/coverage autorater. The bibliography
cites Lanham 2307.13702, Chen 2505.05410, Arcuschin 2503.08679,
Emmons 2510.23966, and Korbak 2507.11473 as related work but does
not port their methods. Turpin 2305.04388 is not cited.

**Distribution.** Verified absent from PyPI (`monitorbench` returns
404 from the PyPI JSON API on 2026-04-27). Repo has `requirements.txt`
only — no `pyproject.toml` or `setup.py`. Designed to be cloned and
run locally, not pip-installed.

**License / Inspect relationship (Updated 2026-05-28).** The repo is
**Apache-2.0**. It **pins `inspect_ai==0.3.177`** in its requirements,
but on inspection it **does not run as Inspect tasks** — the
evaluation executes through a **custom vLLM pipeline**, and the
`inspect_ai` pin appears to be a transitive/utility dependency rather
than the execution substrate. Practically this means MonitorBench is
*not* directly loadable as an Inspect eval today; a **cot-suite
Phase-6 Inspect-native loader** for the MonitorBench instances is
planned (ingest the 1,514 instances as an Inspect dataset and run them
through cot-suite's existing Inspect scorers). This is the same
"ingest as a v0.2 dataset" follow-up flagged below, now with the
concrete note that the upstream `inspect_ai` pin does not by itself
make the instances Inspect-runnable.

**Where this leaves cot-suite's positioning.** MonitorBench is the
closest thing in the field to a "comprehensive monitorability
benchmark" published as a stand-alone artifact. It is **not** the
**first PyPI-installable bundle of Lanham 2307.13702, Turpin
2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons &
Zimmermann 2510.23966 under a unified CLI** — different category.
Its
1,514-instance benchmark could in principle be ingested as a new
*dataset* for cot-suite's existing scorers in v0.2 (interesting
follow-up: do MonitorBench's stress-test instances expose additional
PHR / leg-cov signal vs the original Lanham/Turpin/Chen/Arcuschin
test data?).

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
| Yueh-Han Chen 2603.05706 | closed frontier + Qwen3-14B/32B + R1 | Qwen3-14B | 14B-32B (open) |
| MonitorBench 2603.28590 | mixed closed + open reasoning | not enumerated in abstract | unenumerated |
| Young 2603.20172 | 12 open-weight, 9 families | 7B (family-spanning) | 7B-1T |
| **cot-suite v0.1** | **8 open-weight** | **Qwen2.5-7B-Instruct** | **7B-72B with intermediate rungs** |

**No prior work covers the 7B-72B range with Qwen-base, Llama-base,
native-thinking, and R1-distill variants in one consistent
benchmark.** Closest analog is Lewis-Lim, which lacks 70B and uses
soft-reasoning rather than GPQA-Diamond. (Updated 2026-05-28: the
Young trilogy — §12 — spans a far wider 7B-1T / 9-family range but on
MMLU+GPQA-Diamond with cue-injection faithfulness only, not the
five-methodology bundle; it is the broadest open-weight comparison
baseline now available and is treated in §12.)

### GPQA-Diamond explicit use

| paper | GPQA-Diamond? |
|---|---|
| Meek 2510.27378 | ✅ §4 line 420, Figure 4 |
| Chua & Evans 2501.08156 | ❌ MMLU only |
| OpenAI 2512.18311 | ✅ §6.2 |
| Korbak 2507.11473 | n/a |
| Schoen 2509.15541 | partial (GPQA generally; not explicitly Diamond) |
| Lewis-Lim 2508.19827 | ❌ — confirmed GPQA-Main (448 ex), not Diamond (198 ex) |
| Yueh-Han Chen 2603.05706 | ❌ custom CoT-Control task suite |
| MonitorBench 2603.28590 | ❌ 1,514 novel curated instances |
| Young 2603.22582 | ✅ MMLU + GPQA-Diamond (498 questions) |

### Model-overlap matrix (our 8 models × the 6 prior works)

✅ = paper tested this exact checkpoint. ⚠ = adjacent / similar
checkpoint but not exact match. ❌ = not tested.

| our model | Meek | Chua&Ev | OpenAI | Lewis-Lim | CoT-Control |
|---|---|---|---|---|---|
| Qwen3-Thinking-8B | ❌ | ❌ | ⚠ (R1-Distill-Qwen3-8B, different recipe) | ❌ | ❌ |
| Qwen3-Thinking-14B | ❌ | ❌ | ❌ | ❌ | ✅ (Qwen3-14B) |
| Qwen3-Thinking-32B | ❌ | ❌ | ❌ | ✅ (Qwen3-32B) | ✅ (Qwen3-32B) |
| DS-R1-Distill-Qwen-14B | ❌ | ✅ (MMLU) | ❌ | ❌ | ❌ |
| DS-R1-Distill-Llama-70B | ❌ | ⚠ (MMLU; tested *a* Llama-70B distilled on R1 CoTs but did not specify HF identifier — distillation identity ambiguous) | ❌ | ❌ | ❌ |
| Qwen2.5-7B-Instruct | ❌ | ❌ | ❌ | ✅ | ❌ |
| Qwen2.5-72B-Instruct | ✅ (BBH+MMLU+GPQA-D) | ✅ (MMLU) | ❌ | ❌ | ❌ |
| Llama-3.1-8B-Instruct | ❌ | ❌ | ❌ | ✅ | ❌ |

(Korbak and Schoen have no model-list overlap and are omitted from
the matrix — they're cited for framing/downstream-context, not for
quantitative comparison. The Young trilogy — §12 — reports 12
open-weight models across 9 families but does not publish a
per-checkpoint HF-identifier table in the abstract, so an
exact-overlap column is deferred to the §12 treatment rather than
guessed at here.)

**Models unique to our study** (not tested by any of these 6 prior
works on any benchmark): Qwen3-Thinking-8B, Qwen3-Thinking-14B. Both
sit in the under-explored 8B-14B thinking-mode range.

**Our most-leveraged datapoint: DS-R1-Distill-Llama-70B on
GPQA-Diamond.** Chua & Evans tested *a* Llama-70B distilled on
DeepSeek R1 CoTs on MMLU; whether it is the canonical
`deepseek-ai/DeepSeek-R1-Distill-Llama-70B` or a different
distillation is not specified in the paper (see §2). Our run is on
the canonical HF checkpoint, evaluated on GPQA-Diamond, with our
full multi-metric pipeline — so for our positioning purposes our
exact-named-checkpoint result is unambiguously novel regardless of
how their checkpoint identity resolves.

### Methodological lineage

| metric | originating paper | used by | cot-suite v0.1 |
|---|---|---|---|
| Early-answering AOC | Lanham 2307.13702 | (none of the 8) | ✅ (B1) |
| Mistake-injection AOC | Lanham 2307.13702 | (none of the 8) | ✅ (B1) |
| Cue-injection faithfulness | Turpin 2305.04388, Chen 2505.05410 | Meek, Chua&Ev, Lewis-Lim, Young (§12) | ✅ (B2, B3) |
| Implicit PHR | Arcuschin 2503.08679 | (cited by Meek + MonitorBench; not run) | ✅ (B4) |
| Legibility / coverage | Emmons & Zimmermann 2510.23966 | (cited by Meek + MonitorBench; not run) | ✅ (Stage 1/2/3 autorater) |
| Verbosity | Meek 2510.27378 | Meek only | ❌ v0.2 |
| Confidence-trajectory PHR | Lewis-Lim 2508.19827 | Lewis-Lim only | ❌ v0.2 |
| g-mean² monitorability | OpenAI 2512.18311 | OpenAI only | ❌ recommend v0.1.1 add |
| Cross-classifier faithfulness sensitivity | Young 2603.20172 | Young only | ✅ default posture (cross-classifier; see §12) |
| Thinking-vs-answer-text faithfulness split | Young 2603.22582 / 2603.26410 | Young only | ❌ v0.2 candidate |
| Intentional CoT-control success rate | Yueh-Han Chen 2603.05706 | CoT-Control only | ❌ different threat model — out of scope |
| Stress-test concealment under instruction | MonitorBench 2603.28590 | MonitorBench only | ❌ different threat model — out of scope |

**cot-suite v0.1 is the first PyPI-installable bundle of Lanham
2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin
2503.08679, and Emmons & Zimmermann 2510.23966 under a unified
CLI.** That is the tooling-layer contribution distinct from
any individual prior paper's evaluation. See §11 for the
competing-bundle check and the lit-sweep methodology behind this
claim.

---

## Suggested README positioning

1. Lead with Korbak as the motivating position paper.
2. Cite Meek, Chua & Evans, OpenAI 2512.18311 as the closest
   measurement-side prior work; explicitly position cot-suite as
   open-weight-scaling-curve complement to their frontier-pairs work.
3. Cite Lewis-Lim for the 3-checkpoint direct comparable on
   soft-reasoning vs our GPQA-Diamond.
4. Cite Schoen as downstream-safety motivation, no numeric
   comparison.
5. **DS-R1-Distill-Llama-70B on GPQA-Diamond is the headline novel
   datapoint** relative to all 8 prior works. Chua & Evans tested
   *a* Llama-70B distilled on R1 CoTs on MMLU but did not specify
   HF identifier; our run is on the canonical HF checkpoint with
   the full multi-metric pipeline on a harder benchmark.
6. **Adopt OpenAI's g-mean² metric** for v0.1.1 — biggest legibility
   win for cross-paper comparison at lowest engineering cost.
7. Reference CoT-Control 2603.05706 and MonitorBench 2603.28590 as
   *complementary* monitorability evaluations with different threat
   models (intentional / stress-test concealment); position cot-suite
   as the structural-faithfulness bundle, not a competitor to either.
8. (Updated 2026-05-28) Cite the **Young trilogy** (§12) as the
   concurrent work that motivates cot-suite's **cross-classifier
   reporting default**: because Young 2603.20172 shows faithfulness
   numbers swing 74.4% / 82.6% / 69.7% on *identical traces* purely by
   changing the classifier, cot-suite reports under multiple
   classifier configurations rather than a single number. Young is a
   comparison baseline and a methodological motivator, **not**
   upstream art cot-suite derives from.

---

## Open items to verify before launch

- [x] **OpenAI 2512.18311 three-archetype taxonomy** — verified
      independently 2026-04-26 by user: paper says **intervention /
      process / outcome-property** (NOT memorization / hallucination
      / faithful as stated in the launch plan). Doc framing in §3
      above is correct as written.
- [x] **Chua & Evans R1-Distill-Llama-70B exact identity** —
      verified via WebFetch of the v3 HTML on 2026-04-27. Paper
      describes "DeepSeek released models distilled on 800k CoTs
      from DeepSeek R1" with Llama-70B variant (Section 4 / Appendix
      A.1) but provides **no HuggingFace identifier, no snapshot
      date, no commit SHA, no model-card link**. Whether their
      Llama-70B distillation is the canonical
      `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` or a different
      distillation cannot be determined from the paper. Doc §2
      framing committed to: "their checkpoint identity is
      ambiguous; our exact-named checkpoint is unambiguously novel
      regardless." Same identity ambiguity applies to their
      Distilled Qwen-32B / 14B / 1.5B references.
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
- [x] **MonitorBench 2603.28590 positioning** — verified
      2026-04-27. Not on PyPI (`monitorbench` returns 404 from JSON
      API). No `pyproject.toml`/`setup.py` in repo. Bibliography
      cites Lanham/Chen/Arcuschin/Emmons/Korbak but does not port
      their methods — stress-test paradigm is novel
      ("direct concealment" + "monitor-aware evasion"). 1,514 novel
      curated instances, not sourced from prior eval data.
      Conclusion: not a competing PyPI bundle. cot-suite v0.1's
      "first PyPI-installable bundle of Lanham 2307.13702, Turpin
      2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and
      Emmons & Zimmermann 2510.23966 under a unified CLI" claim
      survives.
- [x] **(Updated 2026-05-28; closed 2026-05-28) Young trilogy verification.**
      §12 abstract-level facts re-confirmed this session by direct
      WebFetch: 2603.22582 states "12 open-weight reasoning models …
      9 architectural families (7B-685B)" (12 is canonical; the repo
      description's "11" is stale); the 74.4/82.6/69.7 three-classifier
      split (2603.20172) and the 55.4% thinking-answer divergence
      (2603.26410) verified against their abstracts. The GitHub repo
      `github.com/ricyoung/cot-faithfulness-open-models` is verified to
      exist (created 2026-03-04) and is the companion to 2603.22582 +
      2603.20172 only — it does NOT contain 2603.26410 (separate work).
      Its README headline shows pipeline 80.8 vs the abstract's 82.6
      (likely an earlier draft).
- [ ] **(Updated 2026-05-28) OpenAI monitorability-evals repo
      metadata** — `github.com/openai/monitorability-evals` verified
      this session: created 2026-04-22, Apache-2.0, 76 stars. The
      blog URL `alignment.openai.com/monitorability-evals/` is cited
      per the task brief; re-confirm before launch.
- [x] **(Updated 2026-05-28; closed 2026-05-28) Accidental-CoT-grading +
      Redwood review URLs** — both verified this session by WebFetch.
      `alignment.openai.com/accidental-cot-grading/` = "Investigating the
      consequences of accidentally grading CoT during RL" (Carroll,
      Korbak, Dou, Baker, Kivlichan; May 7 2026);
      `blog.redwoodresearch.org/p/openai-cot` = Buck Shlegeris's review
      of it (May 7 2026). See §13.

---

## 9. Adjacent monitorability research (Oct 2025–Apr 2026 sweep)

The recent-literature sweep surfaced these papers as **adjacent**
to cot-suite's bundle — relevant to the broader monitorability
agenda but not direct prior art for any of the five wrapped
methodologies. Treatments are paragraph-length; full §1-8 treatment
is reserved for papers with direct method overlap or model overlap.

### Intentional / steganographic unfaithfulness

- **Baker, Huizinga, Gao et al. — arXiv 2503.11926 — "Monitoring
  Reasoning Models for Misbehavior and the Risks of Promoting
  Obfuscation"** (March 14, 2025, OpenAI; Bowen Baker first
  author). Shows that monitoring frontier reasoning models
  (e.g. o3-mini) for reward hacking via a CoT monitor (a separate
  LLM observing the model's chain of thought) is more effective
  than action-only monitoring, but training against the monitor
  produces obfuscated CoTs that defeat detection while the
  misbehavior persists. Charter result for the "Goodharting CoT
  monitors" risk Korbak warns about. Threat model is RL-induced
  obfuscation, not the inference-time concealment cot-suite
  measures.

- **Lee, Yueh-Han Chen, Korbak — arXiv 2602.22303 — "Training
  Agents to Self-Report Misbehavior"** (February 25, 2026; Bruce
  W. Lee, Chen Yueh-Han, Tomek Korbak). Proposes self-incrimination
  training: agents emit a visible signal when they covertly
  misbehave, instead of relying on alignment training to prevent
  the misbehavior. Sibling result to CoT-Control: instead of
  probing the model's *ability* to control CoT, probes whether
  targeted training can *reverse* concealment defaults. Out of
  scope for cot-suite (we measure base-model behavior, not
  post-training behavior of a self-report-trained variant).

- **Storf, Barton-Cooper, Peters-Gill, Hobbhahn — arXiv 2603.00829
  — "Constitutional Black-Box Monitoring for Scheming in LLM
  Agents"** (February 28, 2026; Hobbhahn last author, Apollo
  Research). Black-box behavioral monitoring approach for
  scheming detection that does not depend on CoT legibility.
  Defensive posture under the assumption Korbak's "fragile
  opportunity" thesis is correct — if CoT becomes unmonitorable,
  what fallback do you have? Out of scope for cot-suite.

- **Nishimura-Gasparian & Parikh (METR) —
  https://metr.org/blog/2026-04-01-fine-tuning-cot-controllability/
  — "Fine-tuning experiments on CoT controllability"** (April 1,
  2026). METR's stress-test of Yueh-Han Chen et al.'s CoT-Control
  benchmark (§7): can supervised fine-tuning give models more
  control over what their CoT verbalizes? Method: ~950-example LoRA
  SFT (rank 32, lr 1e-4, single epoch) using prompts from the
  Multilingual Thinking dataset paired with ReasonIF instructions,
  evaluated on GPT-OSS-20B / GPT-OSS-120B / Qwen-3-8B / Qwen-3-32B.
  Headline: average CoT-Control compliance rises from 2.9% → 8.8%
  across the four fine-tuned models, with per-model variation
  visible in the bar chart (GPT-OSS-120B approximately 4.5% → ~11%).
  Mixed signal — small SFT pressure increases controllability, but
  more examples does not monotonically improve, and some fine-tuned
  variants performed *worse* than base. **Significance:**
  CoT-Control (§7) is being actively stress-tested by METR within
  weeks of publication, validating it as the right intentional-
  controllability benchmark to cite. Out of scope for cot-suite as
  an evaluation library (we measure base-model behavior, not
  fine-tuned variants), but a fresh datapoint that strengthens the
  §7 framing.

### Activation-level / mechanistic monitorability

- **McGuinness, Serrano, Bailey, Emmons — arXiv 2512.11949 —
  "Neural Chameleons: Language Models Can Learn to Hide Their
  Thoughts from Unseen Activation Monitors"** (December 12, 2025;
  Emmons last author). Investigates whether language models can
  learn to evade *previously unseen* activation monitors — a
  steganography-at-the-activation-level threat model. Conceptual
  follow-up to Emmons & Zimmermann 2510.23966 (the
  surface-CoT autorater paper cot-suite ports). We flag this as
  the highest-priority candidate to inform a future cot-suite v0.2
  activation-level scorer (gated on the optional `activations`
  extra: nnsight + transformer-lens).

### Bias and motivated-reasoning extensions

- **Arcuschin, Chanin, Garriga-Alonso, Camburu — arXiv 2602.10117
  — "Biases in the Blind Spot: Detecting What LLMs Fail to
  Mention"** (February 10, 2026; Arcuschin first author).
  Introduces the concept of *unverbalized biases* — internal
  biases that influence the model's answer but never surface in
  its CoT — and notes that monitoring via stated reasoning is
  unreliable because of these blind-spot biases. Sibling paper to
  the IPHR detector cot-suite ports (Arcuschin 2503.08679); could
  become an additional detector in v0.2.

- **Venhoff, Arcuschin, Torr, Conmy, Nanda — arXiv 2510.07364 —
  "Base Models Know How to Reason, Thinking Models Learn When"**
  (October 8, 2025; Arcuschin second author). Hybrid-model
  experiments showing that thinking models like DeepSeek R1
  primarily learn *when* to deploy reasoning mechanisms that
  already exist in the base model, rather than acquiring new
  capabilities. Methodologically tangential to monitorability,
  but informs the post-training-shapes-faithfulness intuition
  underlying our scaling-table interpretation.

- **Howe & Carroll — arXiv 2510.17057 — "The Ends Justify the
  Thoughts: RL-Induced Motivated Reasoning in LLM CoTs"**
  (October 20, 2025; Nikolaus Howe and Micah Carroll). Probes a
  different unfaithfulness failure mode — the model rationalizes
  toward a desired conclusion rather than constructing an answer
  — under post-hoc instructions that conflict with learned
  behaviors. Adjacent to PHR but distinct construct.

- **Qiu, Carroll, Allen — arXiv 2601.20299 — "Truthfulness Despite
  Weak Supervision: Evaluating and Training LLMs Using Peer
  Prediction"** (January 28, 2026; Tianyi Alex Qiu first author,
  Carroll second). Mechanism-design / peer-prediction approach to
  the evaluator-capability gap — eliciting honest answers when
  the evaluator cannot verify ground truth. Affects the broader
  monitorability agenda but not cot-suite's specific bundle.

### Safe RL / training-time monitorability

- **Kaufmann, Lindner, Zimmermann, Shah — arXiv 2603.30036 —
  "Aligned, Orthogonal or In-conflict: When can we safely optimize
  Chain-of-Thought?"** (March 31, 2026; Max Kaufmann first
  author). Conceptual framework predicting when CoT optimization
  preserves vs degrades monitorability, validated empirically.
  Pairs with Korbak's recommendation (3) ("if monitorability
  degrades during training, prefer the earlier checkpoint"). Out
  of scope for cot-suite as an evaluation library (we measure, we
  don't train).

---

## 10. Deployment context (Dec 2025 AISI work)

Two December 2025 UK-AISI deployment-monitoring papers appeared in
the sweep. Neither is a methodology evaluation, but both are
relevant context for *how* cot-suite-style evaluations would be
consumed in production safety work.

- **Lindner, Griffin, Korbak, Zimmermann, Irving, Farquhar, Cooney —
  arXiv 2512.22154 — "Practical Challenges of Control Monitoring in
  Frontier AI Deployments"** (December 2025). UK AISI's framework
  for deploying CoT monitoring in production: latency budgets,
  monitor-model selection, escalation policies, alert volume
  management. Cites Korbak 2507.11473 throughout. cot-suite is
  positioned as the kind of *measurement* tooling such a deployment
  framework consumes — measure monitorability per checkpoint, feed
  results into deployment-time policy decisions.

- **Stickland, Michelfeit, Mani, Griffin, Matthews, Korbak, Inglis,
  Makins, Cooney — arXiv 2512.13526 — "Async Control"**
  (December 2025). Asynchronous multi-agent control mechanisms
  tested in software-engineering environments. Not CoT-faithfulness
  per se — control-protocol research. Mentioned for completeness:
  the December 2025 AISI burst included this alongside the
  monitoring-deployment paper.

---

## 11. Lit-sweep methodology and competing-bundle check

### Sweep coverage (Oct 2025–Apr 2026)

To validate that the eight prior works in §1-8 are genuinely the
closest related work — and that no competing PyPI bundle exists —
the following sweep was completed on 2026-04-27.

**Author searches** (arXiv author-query API): Korbak, Baker,
Emmons, Zimmermann, Yanda Chen, Owain Evans, Hobbhahn, Benton,
Arcuschin, Carroll, Bowman, Yueh-Han Chen, Lindner, Lee. Lanham,
Turpin, Perez, Kivlichan covered transitively via co-authorship on
the above sweeps.

**Topic searches**: "chain-of-thought monitorability",
"CoT monitoring/faithfulness", "post-hoc rationalization",
"reasoning model alignment evaluation", "monitorability
evaluation", "CoT controllability", "AI control monitoring".

**Org pages**: Anthropic alignment publications, OpenAI alignment
publications, Apollo Research, UK AISI, METR, Constellation,
Redwood Research.

**Not re-fetched but verified covered transitively**: Schoen
(Apollo, picked up via Hobbhahn / org page), Meek
(picked up via topic + GitHub Actions repo metadata), Chua & Evans
(picked up via Owain Evans author search).

### Competing-bundle check

The "first PyPI-installable bundle of Lanham 2307.13702, Turpin
2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons &
Zimmermann 2510.23966 under a unified CLI" claim was verified
against three negative controls:

1. **PyPI direct probe.** 13 candidate package names probed via
   PyPI JSON API on 2026-04-27 — all 404:
   `cot-faithfulness`, `cot-monitor`, `cot-eval`, `cotfaith`,
   `faithfulness-eval`, `cot-tools`, `chainofthought-eval`,
   `cot-bench`, `monitorability-eval`, `cot-faith`,
   `chain-of-thought-eval`, `cotbench`, `cot-suite-bench`. Plus
   `monitorbench` 404, confirming MonitorBench is not pip-installed.

2. **GitHub topic search** for "cot faithfulness evaluation".
   Surfaced 7 single-paper companion repositories — none bundle
   multiple papers' methodologies, all have 0-2 GitHub stars:
   - gmy2013/COT_Faithfulness_Evaluation
   - shakibyzn/persian-faithful-cot
   - fengrui128/visual-cot-eval
   - DenizErenArici/alignment-pipeline-tr-en
   - timholm/cot-audit
   - sunnypark12/PromptScope
   - skhanzad/AriadneXAI

3. **Comprehensive-benchmark check.** OpenAI's
   `monitorability-evals` repo is the companion to 2512.18311 only
   (intervention/process/outcome-property data; not a bundle of
   prior methods). MonitorBench 2603.28590 (§8) is a stand-alone
   1,514-instance benchmark with novel stress-test methodology;
   does not implement the five wrapped methodologies (verified via
   bibliography + methodology section). CoT-Control 2603.05706
   (§7) is a single-paper companion repo, no PyPI distribution.

**Verdict:** as of 2026-04-27, no other artifact constitutes a
**PyPI-installable bundle of Lanham 2307.13702, Turpin 2305.04388,
Chen 2505.05410, Arcuschin 2503.08679, and Emmons & Zimmermann
2510.23966 under a unified CLI**. The closest comparable artifacts
are MonitorBench (different category — single benchmark, no PyPI)
and CoT-Control (different threat model — intentional control
success rate, single-paper companion).

**Updated 2026-05-28 — re-check against the Young trilogy and the
OpenAI monitorability-evals release.** Neither disturbs the verdict.
The Young repo `github.com/ricyoung/cot-faithfulness-open-models`
(verified 2026-05-28: repo exists; companion to 2603.22582 + 2603.20172
only, does NOT contain 2603.26410, which is separate work) is a
cross-classifier *faithfulness-measurement* study, not a bundle of
the five wrapped methodologies — it operationalizes cue-injection
faithfulness under multiple classifiers, not Lanham AOCs +
legibility/coverage + IPHR pair construction. OpenAI's
`monitorability-evals` (Apache-2.0, created 2026-04-22, 76 stars) is
the companion to 2512.18311 only — intervention/process/outcome-
property g-mean² scaffolding, again not the five-method bundle. The
"first PyPI-installable bundle …" claim survives both. (Neither Young
nor `monitorability-evals` was probed on PyPI this session; both are
research repos distributed via GitHub, consistent with the rest of
the competing-bundle landscape.)

### Per-paper arXiv verification (§9)

Every adjacent-research paper in §9 was independently verified by
WebFetch on 2026-04-27. For each, the abstract page was fetched
and the cited title + author list cross-checked. Verification
results:

- 2503.11926 — Baker et al., "Monitoring Reasoning Models for
  Misbehavior and the Risks of Promoting Obfuscation" — verified.
- 2602.22303 — Lee, Yueh-Han Chen, Korbak, "Training Agents to
  Self-Report Misbehavior" — verified.
- 2603.00829 — Storf, Barton-Cooper, Peters-Gill, Hobbhahn,
  "Constitutional Black-Box Monitoring for Scheming in LLM
  Agents" — verified (Hobbhahn is last author, not first;
  attribution corrected in §9).
- 2512.11949 — McGuinness, Serrano, Bailey, Emmons, "Neural
  Chameleons: Language Models Can Learn to Hide Their Thoughts
  from Unseen Activation Monitors" — verified (Emmons is last
  author, not first; attribution corrected in §9).
- 2602.10117 — Arcuschin, Chanin, Garriga-Alonso, Camburu,
  "Biases in the Blind Spot: Detecting What LLMs Fail to
  Mention" — verified (title corrected from "Blind Spot Biases").
- 2510.07364 — Venhoff, Arcuschin, Torr, Conmy, Nanda, "Base
  Models Know How to Reason, Thinking Models Learn When" —
  verified (Arcuschin is second author, not first; attribution
  corrected in §9).
- 2510.17057 — Howe & Carroll, "The Ends Justify the Thoughts:
  RL-Induced Motivated Reasoning in LLM CoTs" — verified.
- 2601.20299 — Qiu, Carroll, Allen, "Truthfulness Despite Weak
  Supervision: Evaluating and Training LLMs Using Peer
  Prediction" — verified (Carroll is second author, not first;
  attribution corrected in §9).
- 2603.30036 — Kaufmann, Lindner, Zimmermann, Shah, "Aligned,
  Orthogonal or In-conflict: When can we safely optimize Chain-
  of-Thought?" — verified.

All 9 arXiv IDs in §9 resolve to existing papers with matching
content. Five attribution / title corrections were applied based
on verification.

### Post-launch fact-checks (2026-04-27 follow-up)

Three additional verifications surfaced from a manual social-media
scan after the initial sweep landed:

- **Neural Chameleons authorship re-verification.** A Dec 17 2025
  tweet from Alex Serrano (the second author) appeared to claim
  lead authorship. Re-fetched arXiv 2512.11949 author ordering on
  2026-04-27: McGuinness is unambiguously first author with no
  equal-contribution footnote; Serrano is second. The §9 attribution
  "McGuinness, Serrano, Bailey, Emmons" is correct as written.
- **Hobbhahn 2504.12170.** Surfaced as "Hobbhahn full report" via
  social media; verified on 2026-04-27 to be Stix, Pistillo,
  Sastry, **Hobbhahn**, Ortega, Balesni, Hallensleben,
  Goldowsky-Dill, Sharkey — "AI Behind Closed Doors: A Primer on
  The Governance of Internal Deployment" (April 16, 2025). This is
  an internal-deployment-governance paper (Hobbhahn 4th of 9
  authors); off-scope for cot-suite (governance, not CoT-
  monitorability methodology). Not added to §9.
- **METR CoT-Control SFT post.** Surfaced via Korbak's social-media
  reference; verified at
  https://metr.org/blog/2026-04-01-fine-tuning-cot-controllability/
  on 2026-04-27. Active stress-test of Yueh-Han Chen et al.'s
  CoT-Control benchmark (§7) within weeks of its publication.
  Added to §9 under "intentional / steganographic unfaithfulness".

### Stale / superseded items

The recent-literature sweep did not surface any prior paper or
implementation that supersedes the v0.1 framing. The eight
prior works in §1-8 remain the right anchor set; the §9-10
adjacent-research and deployment-context sections capture the
sweep's marginal additions.

---

## 12. The Young trilogy (Mar 2026) — classifier-sensitivity and thinking-vs-answer divergence

**Added 2026-05-28.** A three-paper series by **Richard J. Young**
(March 2026) is the most directly relevant **concurrent** work for
cot-suite's reporting design. It is **not** upstream art cot-suite
derives from — the timing is concurrent and the methodologies do not
overlap with the five wrapped papers — but it is (a) the empirical
motivation for cot-suite's **cross-classifier-sensitivity reporting
default** and (b) the broadest open-weight **comparison baseline** now
available (12 models, 9 families, 7B-1T). The abstract-level facts
below are the verified-abstract summary supplied in the task brief and
are flagged for direct re-verification in the open-items list before
any launch artifact cites a specific number.

### 12.1 — arXiv 2603.20172 — "Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity" (Mar 20 2026)

The central result: **the faithfulness number you report is an
artifact of the classifier you use to read the CoT.** Young runs three
classifiers over the **same** influenced traces:
1. a **regex** classifier,
2. a **regex + LLM pipeline**, and
3. a **Claude Sonnet 4 judge**.

On **12 open-weight models across 9 families (7B–1T)** and **10,276
influenced traces**, the headline faithfulness rates are **74.4%
(regex) / 82.6% (regex+LLM) / 69.7% (Sonnet-4 judge)** on **identical
data**. Per-model gaps span **2.6–30.6pp**, and the classifier choice
induces **ranking reversals** — e.g., a model ranked 1st under one
classifier drops to 7th under another. The paper's load-bearing
conclusion: **"published faithfulness numbers cannot be meaningfully
compared across studies using different classifiers."**

**Why this is the motivator for cot-suite's default.** cot-suite
measures faithfulness with an LLM-as-judge dispatcher
(`cotsuite.core.classify`) and a versioned autorater; Young 2603.20172
is direct evidence that a *single* classifier configuration produces a
number that is not portable across studies. The takeaway baked into
cot-suite's posture is to **report under more than one classifier
configuration** (and to pin the exact classifier in `Provenance`)
rather than publish a single headline faithfulness percentage. This is
a *reporting-discipline* motivation, not a claim that cot-suite
reproduces Young's numbers — cot-suite has not run Young's three-
classifier comparison on its own traces, and no such reproduction is
claimed here.

### 12.2 — arXiv 2603.22582 — "Lie to Me: How Faithful Is CoT Reasoning in Reasoning Models?" (Mar 23 2026)

**12 open-weight models, 9 families, 498 MMLU + GPQA-Diamond
questions, six Chen-style hints, 41,832 inference runs.** Faithfulness
ranges **39.7%–89.9%** across models. The most cot-suite-relevant
finding is a **locus split**: acknowledgment of an influencing hint is
**~87.5% in the thinking tokens** but only **~28.6% in the answer
text** — i.e., models frequently *do* surface the hint in their
internal reasoning while omitting it from the final answer-facing
explanation.

**Relevance to cot-suite.** This is the same six-hint Chen lineage
cot-suite wraps in `cotsuite.tests.chen_cue_injection`, run on a
GPQA-Diamond-inclusive question set that overlaps cot-suite's
benchmark. The thinking-token-vs-answer-text acknowledgment gap is a
concrete candidate metric refinement for cot-suite v0.2: cot-suite's
current PHR / cue-acknowledgment signals do not separately score the
thinking-token channel vs the answer-text channel, and Young 2603.22582
is the evidence that the split is large enough to matter. Logged as a
v0.2 candidate in the methodological-lineage table, not a v0.1 claim.

### 12.3 — arXiv 2603.26410 — "Why Models Know But Don't Say: Thinking-Answer Divergence in Open-Weight Reasoning Models" (Mar 27 2026)

**12 open-weight models on MMLU + GPQA**, reporting **55.4%
thinking-answer divergence among hint-followed cases** — i.e., among
cases where the model's answer changed in the direction of the hint,
the majority show a divergence between what the thinking tokens reveal
and what the answer text states. This is the natural sequel to
2603.22582's locus split, quantified as a divergence rate on the
subset where the hint actually moved the answer.

**Relevance to cot-suite.** Same lineage and same v0.2-candidate
status as 2603.22582 — it sharpens the case that a thinking-vs-answer
divergence scorer belongs on the cot-suite roadmap, and it strengthens
the §12.1 reporting-discipline argument (a single answer-text-only
classifier would systematically under-count acknowledgment relative to
a thinking-token-aware one).

### 12.4 — Code and citation discipline

- **Primary citation is the arXiv IDs**: 2603.20172, 2603.22582,
  2603.26410.
- A companion repository is listed as
  **`github.com/ricyoung/cot-faithfulness-open-models`** — **verified
  2026-05-28: repo exists; companion to 2603.22582 + 2603.20172 only
  (does NOT contain 2603.26410, which is separate work).** Repo
  *description* says "11 models" while the 2603.22582 abstract states
  "12 open-weight reasoning models … 9 architectural families (7B-685B)"
  — 12 is canonical (abstract); the repo description appears stale.
- **No reproduction claim is made against the Young trilogy.**
  cot-suite has run none of Young's three-classifier comparisons,
  thinking-token-vs-answer-text splits, or divergence rates on its own
  traces. Per the reproduction-claim-discipline rules, any future
  statement of the form "cot-suite measured X; Young reports Y; we
  agree by Δ" must cite the producing `scripts/` path, the output
  JSONL, the specific Young arXiv cell, the signed delta, and a
  pre-declared tolerance band — none of which exist today. Young is
  cited here strictly as concurrent motivation + comparison baseline.

### 12.5 — Positioning summary

| Young paper | what it establishes | cot-suite consequence |
|---|---|---|
| 2603.20172 | faithfulness % is classifier-dependent: per-model gaps up to 30.6pp across classifiers (74.4/82.6/69.7 on identical traces), with model-ranking reversals (e.g., Qwen3.5-27B 1st↔7th under different classifiers); see paper for canonical figures — the repo README shows pipeline 80.8 vs the abstract's 82.6 and may reflect an earlier draft | motivates **cross-classifier reporting default**; pin classifier in `Provenance` |
| 2603.22582 | thinking-token (~87.5%) vs answer-text (~28.6%) acknowledgment split, 12 models, GPQA-Diamond-inclusive | v0.2 candidate: separate thinking-channel vs answer-channel scoring |
| 2603.26410 | 55.4% thinking-answer divergence among hint-followed cases | v0.2 candidate: thinking-answer divergence scorer; reinforces 12.1 |

The Young trilogy does **not** change cot-suite's "first
PyPI-installable bundle …" claim (it is a faithfulness-measurement
study, not a five-method bundle — see §11 re-check) and does **not**
make cot-suite derivative (concurrent timing, disjoint methodology).
Its function in this document is to (1) justify the cross-classifier
reporting default and (2) serve as the widest open-weight comparison
baseline available.

---

## 13. External review of frontier-lab monitorability claims (May 2026)

**Added 2026-05-28.** A May 2026 episode is worth recording as
**precedent**: frontier-lab monitorability/CoT-grading claims are now
subject to external technical review, which is part of the ecosystem
cot-suite's open, reproducible measurement posture is meant to serve.

- **OpenAI — "accidental CoT grading" disclosure** —
  `alignment.openai.com/accidental-cot-grading/`. *"Investigating the
  consequences of accidentally grading CoT during RL"* by Micah Carroll,
  Tomek Korbak, Zehao Dou, Bowen Baker, Ian Kivlichan (OpenAI), May 7,
  2026. **Verified 2026-05-28.** A disclosure that a CoT-grading setup
  was applied during RL in a way that was not intended.
- **Buck Shlegeris / Redwood Research review** —
  `blog.redwoodresearch.org/p/openai-cot`. *"A review of 'Investigating
  the consequences of accidentally grading CoT during RL'"* by Buck
  Shlegeris, May 7, 2026. **Verified 2026-05-28.** An external review of
  the OpenAI disclosure above.

**Why this is in scope.** cot-suite's entire value proposition is
*reproducible, externally checkable* monitorability measurement — the
Korbak recommendation (1) that frontier developers "develop and run"
such evaluations and report them implies someone outside the lab can
audit the result. The accidental-CoT-grading episode + Redwood's
review is a concrete instance of exactly that external-audit loop
operating on a frontier lab's monitorability-adjacent claims. It
strengthens the case for cot-suite's `Provenance`-everywhere,
script-and-JSONL-cited reproduction discipline (the
reproduction-claim-discipline rules): claims that can be independently
re-derived are the ones that survive external review. **Both URLs are
cited per the task brief and must be verified before appearing in any
launch artifact** (tracked in open-items).

---

## 14. Commercial agent-oversight tooling — Apollo Watcher

**Added 2026-05-28.** **Apollo Watcher** (Apollo Research,
`apolloresearch.ai`) is a **commercial agent-oversight product** — a
deployment-time monitor for agent behavior. It is recorded here for
landscape completeness and to draw a clean **complementary, not
competitive** boundary:

- **Watcher is a deployment monitor.** It observes running agents in
  production and flags concerning behavior at deploy/runtime — the
  consumer side of the monitorability story (cf. the UK-AISI
  deployment-monitoring papers in §10).
- **cot-suite is a measurement library.** It quantifies, offline and
  reproducibly, *how monitorable a given checkpoint's CoT is* under a
  bundle of published methodologies, across a multi-family scaling
  table.

The two sit at different layers of the same stack: cot-suite produces
the per-checkpoint monitorability measurements that inform whether —
and how much to trust — a CoT-reading deployment monitor like Watcher.
There is no feature overlap and no competing-bundle concern (Watcher
is a product, not a PyPI bundle of the five wrapped methodologies), so
it does not affect the §11 competing-bundle verdict. Apollo's research
arm separately authored / co-authored several §5, §9, §10 papers
(Schoen 2509.15541, Storf 2603.00829, the deployment-context work);
Watcher is the productized counterpart to that research line.
