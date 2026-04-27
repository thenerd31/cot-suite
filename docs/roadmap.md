# Roadmap

This page tracks the v0.1 launch state and the v0.1.1 / v0.2 milestones. The full repo roadmap (provenance hardening, few-shot scaffolded cues, Inspect AI integration milestones, process lessons) lives in [`ROADMAP.md`](https://github.com/thenerd31/cot-suite/blob/main/ROADMAP.md) on GitHub.

## v0.1 — current launch target (mid-May 2026)

Shipped in v0.1:

- **Five paper-verified evaluation methodologies** under one CLI — Lanham, Turpin, Chen, Arcuschin, Emmons & Zimmermann ([metrics overview](metrics/index.md)).
- **Two Inspect AI scorers** auto-discoverable via `[project.entry-points.inspect_ai]` — `cot_legibility_coverage` and `cot_post_hoc_rationalization` ([Inspect AI integration](inspect.md)).
- **Self-grading guard** — `UserWarning` if Inspect's grader role resolves to the eval's primary model.
- **8-model multi-family scaling table** on GPQA-Diamond ([scaling results](scaling.md)).
- **Reproducibility contract** — `eval_version` + `cotsuite_version` + `prompt_version` + `prompt_sha256` + autorater spec ([reproducibility](reproducibility.md)).
- **Production hardening** — hatchling + hatch-vcs versioning, ruff D-rules + mypy strict on the public API surface, GitHub Actions CI matrix (Python 3.11/3.12/3.13 × Ubuntu+macOS), trusted-publisher PyPI release pipeline with Sigstore attestations.

## v0.1.1 — first follow-up

- **g-mean² adoption** (Guan et al. / OpenAI 2512.18311). Verified formula: `gmean2 = tpr * tnr` over (arm × correctness × monitor signal) triples. **Not directly computable from current cot-suite outputs** — requires a binarization scheme on 0-4 Likert autorater outputs + a ground-truth proxy for monitor flagging. Pre-work: 4-8 h methodology design; implementation: 2-4 h. Defer until methodology is locked.

## v0.2 — 4-6 weeks post-launch

Port the remaining four methodologies to Inspect Scorers:

| Scorer | Paper | Effort | Notes |
|---|---|---|---|
| `cot_lanham_early_answering` | Lanham 2307.13702 §3.1 | ~6-10h | Cleanest of the four |
| `cot_lanham_mistake_injection` | Lanham 2307.13702 §3.2 | ~8-12h | Second grader role for the mistake generator |
| `cot_turpin_counterfactual` | Turpin 2305.04388 | ~12-18h | Solver+Scorer pair (cross-sample aggregate) |
| `cot_chen_cue_injection` | Chen 2505.05410 | ~10-14h | Solver+Scorer pair, same pattern as Turpin |

Confirm the post-2026-05-08 `inspect_evals` registry layout before opening any upstream PR.

Other v0.2 candidates (no firm commitments yet):

- **CC-SHAP metric** (Siegel 2404.03189) — Task #13.
- **Verbosity** (Meek 2510.27378) — would round out the cue-injection family.
- **Confidence-trajectory PHR** (Lewis-Lim 2508.19827) — orthogonal PHR signal to Arcuschin's.
- **Activation-level scorer** (gated on the optional `cot-suite[activations]` extra; informed by McGuinness, Serrano, Bailey, Emmons "Neural Chameleons" 2512.11949).
- **Few-shot scaffolded cues** — promote Chen's visual-pattern cue back to the primary catalog (currently in `tests/extensions/`).

## v0.2.1+

- **Open one `inspect_evals` upstream PR** — recommend legibility/coverage first, since it has the tightest reproduction story and is already implemented end-to-end.
- **Use that review feedback** to shape the remaining four metric PRs.

## Out of scope (probably forever)

- **Native Inspect-only architecture** (Scope C from `docs/inspect_ai_integration_assessment.md`). The standalone CLI's "score a pre-existing trajectory JSONL" workflow doesn't fit Inspect's eval-time-generation model. Replacing it would discard the v0.1 multi-family scaling outputs.

## v1.0 — provenance hardening

Three steps in `ROADMAP.md` → "Provenance hardening":

1. **Registry-time assertion.** Every `Cue` / `BiasConfig` registered in a primary namespace must have `provenance.verified_against_pdf=True` at module-import time, or raise `ProvenanceError`. Unverified entries must live under `tests/extensions/`. Turns the one-off unit test into an always-on structural guarantee.
2. **CLI: `cotsuite provenance audit`.** Walk every registered metric/test/cue/bias, print a grouped table (paper-verified / unverified / extension). One-command answer to "what does this library actually claim to implement."
3. **CI commit check.** Any commit flipping a `verified_against_pdf` field from `False` to `True` must also touch a file under `docs/paper-refs/<arxiv_id>.md` or `AUDIT.md`. Prevents silent promotion without human PDF review.
