# Roadmap

**Launch status: Pre-launch, targeting late-July to mid-August 2026.**

This page tracks the active phase plan (Phases 5-7+) and the version-keyed
milestones it maps onto. The full repo roadmap (phase detail, provenance
hardening, few-shot scaffolded cues, parking lot, process lessons) lives in
[`ROADMAP.md`](https://github.com/thenerd31/cot-suite/blob/main/ROADMAP.md) on
GitHub. Each phase maps to the version-numbering convention: Phase 5 = v0.1
launch target, Phase 6 = v0.2-era integration, Phase 7+ = launch / v1.0.

## Phase 5 (current) — Reproduction-faithfulness lockdown

The v0.1 launch target: state each paper's reproduction status honestly and
execute the one reproduction feasible against released artifacts (Turpin),
rather than relaunch under overclaimed framing (see AUDIT.md "Briefing
Reconciliation", 2026-05-28). Per the 5-paper reproducibility audit
(2026-05-30), **only Turpin is a cell-for-cell reproduction**; the other four
are method-implementations applied to current models.

- **B2 Turpin — reproduction COMPLETE (the one cell-for-cell reproduction).**
  Stage A reproduces the Turpin 2305.04388 Table 1 four-cell `suggested_answer`
  accuracy drops within ±0.08pp on all 4 cells, by metric-replay against
  Turpin's *released* `bbh_samples` (commit `035a725`; AUDIT.md Reproduction
  Claims Ledger). Stage B — a **novel** 4-model × 5-task capability curve (NOT a
  paper reproduction) — has executed (commit `d0803b4`).
- **B4 redux (Arcuschin) — the only against-release reproduction path,
  PLANNED.** Run cot-suite's IPHR detector on ChainScope's *released* traces and
  match the paper's per-model cells. The current single-model detector run is an
  **application**, not a reproduction.
- **Emmons — from-spec reproduction DEFERRED / optional.** E-Z release only the
  Appendix C prompt (no trajectories, no code, no machine-readable cells), so a
  reproduction must be from-spec: 5 paper models × 4 datasets with
  Gemini-2.5-Pro as rater, matched to Table 1 — cost-heavy and gated by the
  Gemini-2.5-Pro deprecation window (2026-06-17). The shipped **cross-judge
  validation** (3-judge κ; gpt-4o pending Tier-2) is a *different* artifact — it
  validates the rater substitution, not E-Z's cells.
- **B1 (Lanham) + B3 (Chen) — method-implementations, NOT cleanly
  reproducible.** Retired models (Lanham: Claude 1.3; Chen: Claude 3.5 v2 / 3.7)
  and no public code/data release, so cell-for-cell reproduction is infeasible;
  both ship as **directional instruments on current models only** (Lanham also
  has an underspecified AOC estimator).
- **Phase 5 infrastructure** — client-adapter extension, split:
  **P1a (done)** routes Together / Fireworks / DeepSeek / OpenRouter through
  `get_grader_client` via a generic `OpenAICompatibleClient` (+ `verify_keys`
  checks); **P1b (blocked on credentials; resumes when `modal setup` +
  `HF_TOKEN` are provided)** is the Modal `@modal.web_server` vLLM redeploy.
  Plus a multi-judge wrapper primitive.

## Phase 6 — Monitorability framework integration

The v0.2 era (4-6 weeks post-launch). Do not start before Phase 5 closes.

- **OpenAI monitorability-evals interop** — g-mean² + cross-fit filter + the
  three archetypes (intervention / process / outcome-property) as
  Inspect-native primitives. The g-mean² metric is **vendored verbatim from
  `openai/monitorability-evals` (Apache-2.0)**; the repo will include
  `LICENSES/Apache-2.0-monitorability-evals.txt` + a `NOTICE` file with
  attribution, and the vendored `gmean.py` preserves the OpenAI copyright
  header.
- **MonitorBench Inspect loader** — vendor the 1,514-instance
  `datasets/**.json` + `dataset_config.yaml`; reimplement the three-dimension
  scoring + two stress-test wrappers (Direct Concealment, Monitor-aware
  Evasion) on cot-suite's Inspect baseline (`>=0.3.199`). Do not depend on
  MonitorBench's own `inspect_ai==0.3.177` pin or reuse its Inspect code.
- **Cross-judge / classifier-sensitivity reporting as default** for all
  scorers.
- **CoT-Control stress-test integration** (Yueh-Han Chen 2603.05706).

## Phase 7+ — Launch prep

The launch / v1.0 cut: reviewer outreach, workshop submission, hero artifacts,
and provenance hardening (see "v1.0" in `ROADMAP.md`).

## Version-keyed milestones

### v0.1 — Phase 5 launch target

Shipped in v0.1:

- **Five paper-method implementations** under one CLI — Lanham, Turpin, Chen,
  Arcuschin, Emmons & Zimmermann ([metrics overview](metrics/index.md)). One
  (Turpin) is reproduced cell-for-cell against released artifacts (±0.08pp); the
  rest are method-implementations applied to current models (per-paper
  reproduction status in Phase 5 above).
- **Two Inspect AI scorers** auto-discoverable via
  `[project.entry-points.inspect_ai]` — `cot_legibility_coverage` and
  `cot_post_hoc_rationalization` ([Inspect AI integration](inspect.md)).
- **Self-grading guard** — `UserWarning` if Inspect's grader role resolves to
  the eval's primary model.
- **8-model multi-family scaling table** on GPQA-Diamond ([scaling results](scaling.md)) —
  the load-bearing Phase 5 empirical contribution.
- **Reproducibility contract** — `eval_version` + `cotsuite_version` +
  `prompt_version` + `prompt_sha256` + autorater spec ([reproducibility](reproducibility.md)).
- **Production hardening** — hatchling + hatch-vcs versioning, ruff D-rules +
  mypy strict on the public API surface, GitHub Actions CI matrix (Python
  3.11/3.12/3.13 × Ubuntu+macOS), trusted-publisher PyPI release pipeline with
  Sigstore attestations.

### v0.1.1 — first follow-up

- **g-mean² methodology design** (Guan et al. / OpenAI 2512.18311). Verified
  formula: `gmean2 = tpr * tnr` over (arm × correctness × monitor signal)
  triples. **Not directly computable from current cot-suite outputs** —
  requires a binarization scheme on 0-4 Likert autorater outputs + a
  ground-truth proxy for monitor flagging. The vendored Phase 6 metric file is
  mechanical; this methodology is the blocker. Defer until locked.
- **Cross-judge ablation** — estimate the ~3-5% judge-error rate documented in
  AUDIT.md (2026-04-28) and tighten the v2 PHR numbers; subsumed by the
  multi-judge wrapper primitive once it lands.
- **B4 cross-parser / cross-judge ablation** — discriminate the three
  candidate explanations for the 8pp gap below the paper on GPT-4o-mini. B4
  validation status stays ⚠ pending — not ✓, not ✗.
- **B1 redux** may land here if it deferred from Phase 5 (see "Out of scope for
  v0.1").

### v0.2 — Phase 6 integration (4-6 weeks post-launch)

Port the remaining four methodologies to Inspect Scorers:

| Scorer | Paper | Notes |
|---|---|---|
| `cot_lanham_early_answering` | Lanham 2307.13702 §3.1 | Cleanest of the four |
| `cot_lanham_mistake_injection` | Lanham 2307.13702 §3.2 | Second grader role for the mistake generator |
| `cot_turpin_counterfactual` | Turpin 2305.04388 | Solver+Scorer pair (cross-sample aggregate) |
| `cot_chen_cue_injection` | Chen 2505.05410 | Solver+Scorer pair, same pattern as Turpin |

Plus the Phase 6 monitorability integration (OpenAI monitorability-evals
interop, MonitorBench loader, default cross-judge sensitivity, CoT-Control
stress tests). Confirm the post-2026-05-08 `inspect_evals` registry layout
before opening any upstream PR.

Other v0.2 candidates (no firm commitments yet):

- **CC-SHAP metric** (Siegel 2404.03189) — Task #13.
- **Verbosity** (Meek 2510.27378) — would round out the cue-injection family.
- **Confidence-trajectory PHR** (Lewis-Lim 2508.19827) — orthogonal PHR signal
  to Arcuschin's.
- **Activation-level scorer** (gated on the optional `cot-suite[activations]`
  extra; informed by McGuinness, Serrano, Bailey, Emmons "Neural Chameleons"
  2512.11949).
- **Few-shot scaffolded cues** — promote Chen's visual-pattern cue back to the
  primary catalog (currently in `tests/extensions/`).

### v0.2.1+

- **Open one `inspect_evals` upstream PR** — recommend legibility/coverage
  first, since it has the tightest reproduction story and is already
  implemented end-to-end.
- **Use that review feedback** to shape the remaining four metric PRs.

### v1.0 — Phase 7+ provenance hardening

Three steps in `ROADMAP.md` → "v1.0 — Phase 7+ launch / provenance hardening":

1. **Registry-time assertion.** Every `Cue` / `BiasConfig` registered in a
   primary namespace must have `provenance.verified_against_pdf=True` at
   module-import time, or raise `ProvenanceError`. Unverified entries must live
   under `tests/extensions/`. Turns the one-off unit test into an always-on
   structural guarantee.
2. **CLI: `cotsuite provenance audit`.** Walk every registered
   metric/test/cue/bias, print a grouped table (paper-verified / unverified /
   extension). One-command answer to "what does this library actually claim to
   implement."
3. **CI commit check.** Any commit flipping a `verified_against_pdf` field from
   `False` to `True` must also touch a file under `docs/paper-refs/<arxiv_id>.md`
   or `AUDIT.md`. Prevents silent promotion without human PDF review.

## Out of scope for v0.1

- **B1 redux may defer to v0.1.1.** If Phase 5 tightens — i.e. B4 redux, B3,
  and the Emmons–Zimmermann cross-judge centerpiece consume the runway — the
  4-model Lanham capability curve slips to v0.1.1. B1 redux is the
  lowest-priority Phase 5 item (Claude 1.3 retirement) and does not gate
  launch.
- **Native Inspect-only architecture** (Scope C from
  `docs/inspect_ai_integration_assessment.md`). The standalone CLI's "score a
  pre-existing trajectory JSONL" workflow doesn't fit Inspect's
  eval-time-generation model. Replacing it would discard the v0.1
  multi-family scaling outputs. Probably out of scope forever.

## Process

The Phase 5 reconciliation (AUDIT.md "Briefing Reconciliation", 2026-05-28)
installed five guardrail skills under `.claude/skills/` that govern every
reproduction claim and multi-step execution in this repo:

- **`.claude/skills/recon-plan-gate-execute/`** — recon → plan → gate →
  execute.
- **`.claude/skills/reproduction-claim-discipline/`** — every reproduction
  claim cites script path, output JSONL path, source-paper cell, exact delta,
  and tolerance band, and adds a row to the AUDIT.md Reproduction Claims Ledger
  in the same commit.
- **`.claude/skills/worktree-truth/`** — the worktree is ground truth.
- **`.claude/skills/no-eager-stop/`** — do not stop short of the finish line.
- **`.claude/skills/pause-on-precondition/`** — pause and surface a missing
  precondition instead of fabricating around it.
