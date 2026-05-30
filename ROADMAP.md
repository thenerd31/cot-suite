# ROADMAP.md

Phase plan and post-v0.x parking lot. **Launch status: Pre-launch,
targeting late-July to mid-August 2026.**

The phase plan below (Phases 5-7+) is the active execution sequence. Each
phase maps onto the repo's version-numbering convention: Phase 5 is the
v0.1 launch target, Phase 6 is the v0.2-era integration work, and Phase 7+
is launch / v1.0 prep. The version-keyed milestone detail (v0.1.1 / v0.2 /
v0.2.1+ / v1.0) is preserved below the phase plan so existing references
still resolve.

**Do not implement Phase 6+ items inside v0.1 — the v0.1 scope is frozen on
fixes and honest per-paper reproduction-status work, not features.** The parking-lot
sections at the bottom exist so good ideas aren't lost.

---

## Phase 5 (current) — Reproduction-faithfulness lockdown

Maps to the **v0.1** launch target. The Phase 5 goal is to state each paper's
reproduction status honestly and execute the one reproduction feasible against
released artifacts (Turpin), rather than relaunch under overclaimed framing
(see AUDIT.md "Briefing Reconciliation", 2026-05-28).

Per the 5-paper reproducibility audit (2026-05-30), **only Turpin is a
cell-for-cell reproduction**; the other four are method-implementations, with
reproduction status noted honestly per paper.

- **B2 Turpin — reproduction COMPLETE (the one cell-for-cell reproduction).**
  Stage A reproduces the Turpin 2305.04388 Table 1 four-cell `suggested_answer`
  accuracy drops within ±0.08pp on all 4 cells, by metric-replay against
  Turpin's *released* `bbh_samples` (commit `035a725`; AUDIT.md Reproduction
  Claims Ledger; `scripts/validate_b2_turpin_stage_a.py`). Stage B — a **novel**
  4-model × 5-task capability curve (NOT a paper reproduction) — has executed
  (commit `d0803b4`).
- **B4a (Arcuschin) — IPHR against-release reproduction — COMPLETE, integer-exact.**
  cot-suite *independently* reimplements the three IPHR criteria
  (`src/cotsuite/tests/iphr.py`) and metric-replays them on ChainScope's released
  `df-wm-non-ambiguous-hard-2` (`jettjaniak/chainscope` @ `bb128ac0`, MIT),
  recovering the paper's per-model unfaithful-pair counts **±0 (integer-exact)**
  for **9 non-oversampled models** — including **4 of 7 paper-headline cells**
  (gpt-4o-mini 13.49%, claude-3.5-haiku 7.42%, gemini-2.5-flash 2.17%,
  claude-3.7-sonnet_1k 0.04%). $0 metric-replay, verified vs ChainScope's own
  computed counts (stronger than Turpin's ±0.08pp).
  `scripts/validate_b4_iphr_reproduction.py`; AUDIT Reproduction Claims Ledger.
  **7 oversampled models blocked** (3 headline — chatgpt-4o-latest, deepseek-r1,
  gemini-2.5-pro; 4 non-headline — claude-3.6-sonnet, claude-3.7-sonnet,
  claude-3.7-sonnet_64k, gpt-4o-2024-08-06): their
  flagging needed an adaptive two-pass `--unfaithful-only -n 100` pipeline
  (pass-1 n=10 candidate selection → pass-2 n=100) whose pass-1 state is not in
  the released aggregate df → unreconstructable (artifact-availability limit, not
  implementation — same code is ±0 on all 9). The figure denominator is
  `n_pairs=4892`; the paper *text* says 4,834 — cot-suite reproduces the figure
  and names the inconsistency.
- **B4b — PHR × IPHR orthogonality study — optional, NOT a reproduction.** A
  *novel* cross-metric measurement: overlap between cot-suite's per-trajectory
  PHR detector (`validate_b4_arcuschin.py`, an **application**, not a
  reproduction) and IPHR labels on ChainScope traces. Young-style
  classifier-sensitivity work; distinct from B4a and explicitly not a paper
  reproduction.
- **Emmons — from-spec reproduction DEFERRED / optional.** E-Z release only the
  Appendix C prompt (no trajectories, no code, no machine-readable cells), so a
  reproduction must be from-spec: the 5 paper models (Qwen3-235B, GPT-OSS-120B,
  DeepSeek-R1, Gemini-2.5-Pro/Flash) on the 4 datasets (HLE / GPQA-Diamond /
  ARC-AGI / AIME) with Gemini-2.5-Pro as rater, matched to Table 1. Cost-heavy
  and gated by the Gemini-2.5-Pro deprecation window (2026-06-17). The shipped
  **cross-judge validation** (3-judge κ, Haiku/Sonnet/Gemini; gpt-4o pending
  Tier-2) is a *different* artifact — it validates the rater substitution, not
  E-Z's cells.
- **B1 (Lanham) + B3 (Chen) — method-implementations, NOT cleanly reproducible.**
  Both have retired models (Lanham: Claude 1.3; Chen: Claude 3.5 v2 / 3.7) and
  no public code/data release, so cell-for-cell reproduction is infeasible.
  cot-suite implements the four Lanham faithfulness tests and the Chen six-cue
  method as **directional instruments on current models only**. (Lanham also
  has an underspecified AOC estimator; `validation/b1_lanham_raw.jsonl` holds
  only 404s from a deprecated checkpoint.)
- **Phase 5 infrastructure:**
  - **Client-adapter extension — P1a (DONE, this commit).** A generic
    `OpenAICompatibleClient` routes Together / Fireworks / DeepSeek /
    OpenRouter through `get_grader_client` (all four expose the OpenAI REST
    surface), and `verify_keys` gains
    `check_{together,fireworks,deepseek,openrouter}`. Unblocks B3 (DeepSeek
    V3/R1) and B4 redux (qwq-32b via Together). Mocked-tested, $0 spend.
  - **Client-adapter extension — P1b (BLOCKED on credentials).** Modal
    `@modal.web_server` vLLM redeploy exposing `/v1/chat/completions` (port
    8000, `VLLM_API_KEY` auth), to become a 5th OpenAI-compatible endpoint.
    Status: **blocked on credentials; resumes when `modal setup` + `HF_TOKEN`
    are provided** (the gated model `meta-llama/Llama-3.1-8B-Instruct`
    requires the HF token). Unblocks B1 redux (Llama-3.1-8B) — the
    deferrable methodology.
  - **Multi-judge wrapper primitive** — a reusable wrapper that fans a single
    trajectory across N judge models, the shared substrate under the
    Emmons–Zimmermann cross-judge centerpiece and the v0.1.1 cross-judge
    ablations already scoped in AUDIT.md.

---

## Phase 6 — Monitorability framework integration

Maps to the **v0.2** era (4-6 weeks post-launch). Do not start before Phase 5
closes.

- **OpenAI monitorability-evals interop.** Bring g-mean² + the cross-fit
  filter + the three archetypes (intervention / process / outcome-property)
  in as Inspect-native primitives.
  - g-mean² metric **vendored verbatim from `openai/monitorability-evals`
    (Apache-2.0)**. The repo will include
    `LICENSES/Apache-2.0-monitorability-evals.txt` + a `NOTICE` file with
    attribution; the vendored `gmean.py` preserves the OpenAI copyright
    header. Source file upstream: `metric/intervention_gmean_metric.py`.
    Verified formula: `gmean2 = tpr * tnr` over (arm × correctness × monitor
    signal) triples. Not directly computable from current cot-suite outputs
    without a binarization scheme on the 0-4 Likert autorater outputs plus a
    ground-truth proxy (see the v1.0.1 milestone below).
- **MonitorBench Inspect loader.** 1,514 instances across 3 dimensions + 2
  stress-test settings, loaded as an Inspect dataset. Vendor the
  1,514-instance `datasets/**.json` + `dataset_config.yaml`. Reimplement the
  three-dimension scoring + two stress-test wrappers (Direct Concealment,
  Monitor-aware Evasion) on cot-suite's Inspect baseline (`>=0.3.199`). Do not
  depend on MonitorBench's own `inspect_ai==0.3.177` pin or reuse its Inspect
  code.
- **Cross-judge / classifier-sensitivity reporting as default for all
  scorers.** Promote the multi-judge wrapper primitive (Phase 5
  infrastructure) into the default reporting path so every scorer emits a
  cross-judge sensitivity band, not just the headline number.
- **CoT-Control stress-test integration** (Yueh-Han Chen 2603.05706) — add
  the CoT-Control stress tests as an Inspect-native setting.

---

## Phase 7+ — Launch prep

Maps to **launch / v1.0**.

- Reviewer outreach.
- Workshop submission.
- Hero artifacts (the load-bearing reproduction figures and the cross-judge
  sensitivity story).
- Provenance hardening (the three structural guarantees detailed under "v1.0"
  below) lands as part of the v1.0 cut.

---

## Out of scope for v0.1

- **B1 redux may defer to v0.1.1.** If Phase 5 tightens — i.e. B4 redux, B3,
  and the Emmons–Zimmermann cross-judge centerpiece consume the runway — the
  4-model Lanham capability curve slips to v0.1.1. This is acceptable: B1
  redux is the lowest-priority Phase 5 item because of the Claude 1.3
  retirement, and the decisive Lanham claim does not gate launch.
- **Native Inspect-only architecture** (Scope C from
  `docs/inspect_ai_integration_assessment.md`). The standalone CLI's "score a
  pre-existing trajectory JSONL" workflow doesn't fit Inspect's
  eval-time-generation model. Replacing it would throw away the v0.1
  multi-family recorded outputs (Stage 1+2+3 results across 8 model
  directories). Probably out of scope forever.

---

## Process

The Phase 5 reconciliation (AUDIT.md "Briefing Reconciliation", 2026-05-28)
installed five guardrail skills under `.claude/skills/`. They are the working
discipline for every reproduction claim and every multi-step execution in
this repo:

- **`.claude/skills/recon-plan-gate-execute/`** — recon → plan → gate →
  execute. No code before the plan clears the gate.
- **`.claude/skills/reproduction-claim-discipline/`** — every reproduction
  claim must cite (a) the script path that produced it, (b) the output JSONL
  path, (c) the source-paper cell, (d) the exact delta, and (e) the tolerance
  band, and must add a row to the AUDIT.md Reproduction Claims Ledger in the
  same commit.
- **`.claude/skills/worktree-truth/`** — the worktree is ground truth; never
  claim a result a briefing asserts but the worktree does not contain.
- **`.claude/skills/no-eager-stop/`** — do not stop short of the stated
  finish line.
- **`.claude/skills/pause-on-precondition/`** — pause and surface when a
  precondition (API key, deployed model, committed dataset) is missing
  instead of fabricating around it.

---

## Version-keyed milestones (preserved)

The phases above are the active sequence. The version-keyed detail below is
retained so existing cross-references and the docs-site roadmap
(`docs/roadmap.md`) stay consistent.

### v0.1 (Phase 5 launch target)

- **Five paper-method implementations** under one CLI — Lanham, Turpin, Chen,
  Arcuschin, Emmons & Zimmermann. One (Turpin) is reproduced cell-for-cell
  against released artifacts (±0.08pp); the rest are method-implementations
  applied to current models (per-paper reproduction status in Phase 5 above).
- **Two Inspect AI scorers** auto-discoverable via
  `[project.entry-points.inspect_ai]` — `cot_legibility_coverage` and
  `cot_post_hoc_rationalization` (per-trajectory Arcuschin signal — strict
  subset of the paper's full pair-construction IPHR methodology).
- **Self-grading guard** — `cotsuite.inspect._safety.warn_if_self_grading`
  fires a `UserWarning` when Inspect's grader role resolves to the eval's
  primary model.
- **8-model multi-family scaling table** on GPQA-Diamond — the load-bearing
  Phase 5 empirical contribution; `scripts/verify_headline.py --all`
  reproduces 8/8 cells from committed JSONLs within ±0.5pp.
- **Reproducibility contract** — `eval_version` + `cotsuite_version` +
  `prompt_version` + `prompt_sha256` + autorater spec.
- **Production hardening** — hatchling + hatch-vcs versioning, ruff D-rules +
  mypy strict on the public API surface, GitHub Actions CI matrix (Python
  3.11/3.12/3.13 × Ubuntu+macOS), trusted-publisher PyPI release pipeline with
  Sigstore attestations.

### v0.1.1 — first follow-up

- **g-mean² methodology design** (OpenAI 2512.18311). The vendored
  `metric/intervention_gmean_metric.py` (Phase 6) is mechanical; the open
  problem is the binarization scheme on 0-4 Likert autorater outputs plus a
  ground-truth proxy for monitor flagging. Defer until the methodology is
  locked, then the Phase 6 metric becomes computable from cot-suite outputs.
- **Cross-judge ablation.** Run the existing trajectories through additional
  judges (e.g. GPT-5.5-Thinking, Gemini-2.5-Pro) to estimate the ~3-5%
  judge-error rate documented in AUDIT.md (2026-04-28) and tighten the v2 PHR
  numbers. Subsumed by the multi-judge wrapper primitive once it lands.
- **B4 cross-parser / cross-judge ablation.** Discriminate the three candidate
  explanations for the 8pp gap below the paper on GPT-4o-mini (detector scope,
  parser permissiveness, sample noise). Validation status for B4 stays
  ⚠ pending this ablation — not ✓, not ✗.
- **Llama-3.1-8B PHR CI tightening.** Re-run on CommonsenseQA / MMLU-Pro to
  grow the correct-trajectory subsample to n≥100 and resolve the lone
  partially-resolved cluster cell. Modest compute, post-launch.
- **Independent-reparse ablation.** At least one detection method that
  re-parses from `raw_text` rather than sharing the upstream `final_answer`
  field (lesson from the 2026-04-27 parser-bug fix).
- **B1 redux** may land here if it deferred from Phase 5 (see "Out of scope
  for v0.1").

### v0.2 — Phase 6 integration (4-6 weeks post-launch)

The remaining methods are ported to Inspect **in their correct abstraction** — the
Stage-0 scorer recon (2026-05-30) confirmed not all fit `Scorer`. Status:

| Method | Paper | Inspect abstraction | Status |
|---|---|---|---|
| Chen cue-injection | Chen 2505.05410 | **Solver + Scorer** | ✅ shipped — `cot_chen_cue_injection` + `cot_cue_injection_solver` (per-sample cue-follow + verbalization; multi-judge from the start). |
| Turpin counterfactual | Turpin 2305.04388 | **Solver + Scorer** | ✅ shipped — `cot_turpin_counterfactual` + `cot_bias_injection_solver` (per-sample bias-follow + verbalization). `accuracy_drop` stays **dataset-level** (the ±0.08pp-validated B2 path), **not** a `Score`. |
| Lanham `early_answering` | Lanham 2307.13702 §3.1 | **Task / Solver — NOT Scorer** | ✅ shipped (POC) — `cot_lanham_early_answering` solver (re-elicits the model-under-test at CoT-prefix fractions; reuses the native `@register_test` fn) + thin `cot_lanham_early_answering_aoc` scorer. Single model. |
| Lanham `mistake_injection` | Lanham §3.2 | **Task / Solver** | ⏳ v0.2 — needs a **2nd model role** (mistake generator). |
| Lanham `paraphrasing` | Lanham §3.3 | **Task / Solver** | ⏳ v0.2 — needs a **2nd model role** (paraphraser). |
| Lanham `filler_tokens` | Lanham §3.4 | **Task / Solver** | ⏳ v0.2 (single model). |

Build order (done): Chen → Turpin → Lanham `early_answering` (POC, proving Lanham
is Inspect-native). Remaining: the three Lanham interventions above
(`mistake_injection` / `paraphrasing` each need a second model role).

Plus the Phase 6 monitorability integration: OpenAI monitorability-evals
interop (g-mean² + cross-fit filter + three archetypes), the MonitorBench
loader, cross-judge sensitivity as default, and CoT-Control stress tests.
Confirm the post-2026-05-08 `inspect_evals` registry layout before opening any
upstream PR (per inspect_evals PR #1538 by Scott-Simmons).

Other v0.2 candidates (no firm commitments yet):

- **CC-SHAP metric** (Siegel 2404.03189) — Task #13.
- **Verbosity** (Meek 2510.27378) — would round out the cue-injection family.
- **Confidence-trajectory PHR** (Lewis-Lim 2508.19827) — orthogonal PHR signal
  to Arcuschin's.
- **Activation-level scorer** (gated on the optional `cot-suite[activations]`
  extra; informed by McGuinness, Serrano, Bailey, Emmons "Neural Chameleons"
  2512.11949).

### v0.2.1+

- **Open one `inspect_evals` upstream PR** — recommend legibility/coverage
  first, since it is the cleanest implementation to demonstrate for
  upstreaming (Turpin is the project's tightest *reproduction*; this autorater
  is a method-implementation) and is already implemented end-to-end.
- **Use that review feedback** to shape the remaining four metric PRs.

### v1.0 — Phase 7+ launch / provenance hardening

Today `Provenance` is documentation-only: a frozen dataclass with `arxiv_id`,
`section`, `verified_against_pdf`, and `notes`. One unit test
(`tests/test_turpin_chen.py::test_all_primary_cues_are_pdf_verified`) enforces
that every entry in the primary Chen catalog has `verified_against_pdf=True`.
A careless PR could flip the flag and nothing else would stop it. Three
hardening steps, in order of value:

1. **Registry-time assertion.** Every `Cue` / `BiasConfig` registered in a
   primary namespace (`tests.turpin_counterfactual`, `tests.chen_cue_injection`,
   future `tests.lanham.*` catalogs) must have
   `provenance.verified_against_pdf=True` at module-import time, or raise
   `ProvenanceError`. Unverified entries must live under `tests/extensions/`.
   Turns the one-off unit test into an always-on structural guarantee.
2. **CLI: `cotsuite provenance audit`.** Walk every registered
   metric / test / cue / bias and print a grouped table (paper-verified /
   unverified / extension). One-command answer to "what does this library
   actually claim to implement."
3. **CI commit check.** Any commit whose diff flips a `verified_against_pdf`
   field from `False` to `True` must also touch a file under
   `docs/paper-refs/<arxiv_id>.md` or `AUDIT.md`. Prevents silent promotion
   without human PDF review. Implement as
   `.github/workflows/provenance-check.yml` on pull_request events.

---

## Parking lot (no committed version)

### Few-shot scaffolded cues

The Chen Visual Pattern cue lives in `tests/extensions/` today because it
requires few-shot examples with ■/□/✓ markers, and the `Cue.renderer`
signature `(question, target) -> prompt` has no place for few-shot
scaffolding.

Design sketch:

    class FewShotCue:
        name: str
        build_exemplars: Callable[[str], list[Exemplar]]   # target -> exemplars
        build_prompt: Callable[[list[Exemplar], str, str], str]
        provenance: Provenance

Would let us promote Visual Pattern back into the primary Chen catalog with
`verified_against_pdf=True`. Also unblocks any paper-verified cue that depends
on few-shot context.

### Dense-sweep filler_tokens reproduction

Current `filler_tokens` defaults to a sparse dyadic sweep
`(0, 1, 2, 4, 8, 16, 32, 64, 128, 256)`. Lanham 2307.13702 §3.4 sweeps densely
from 0 to max-sampled-CoT-length. For a published-number reproduction against
Lanham Fig 6 we'd want a dense sweep mode. Low priority — the decisive claim
("zero uplift from filler at any length") survives a sparse sweep.

### Length-weighted AOC alignment — moved to `BLOCKERS.md`

Moved to `BLOCKERS.md` on 2026-04-19. This is a precondition for any
"reproduces Lanham Table 2" claim, not a future-work item.

## Process lessons

- **Pre-deploy vLLM smoke test.** Before any Modal deploy with a new inference
  stack, run a local `AsyncEngineArgs` smoke-test against a small model (e.g.,
  Qwen2.5-0.5B on CPU) to catch API churn before burning GPU time. Rationale:
  on 2026-04-20, vllm 0.19.x had dropped the `disable_log_requests` kwarg — a
  5s local `AsyncEngineArgs(...)` instantiation would have caught it before a
  GPU crash-loop.
- **Detector ablations need to span the full parsing pipeline.** The Stage 3.5
  detector ablation shared the same upstream `final_answer` field across all
  three detection methods, so they shared the parser-layer failure mode.
  Future ablations should include at least one method that re-parses from
  `raw_text` independently.
