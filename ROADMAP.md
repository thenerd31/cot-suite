# ROADMAP.md

Post-v0.x future work. **Do not implement any of these inside v0.x — the
v0.x scope is frozen on fixes and paper-verified replications, not features.**
This file is a parking lot so good ideas aren't lost.

## Provenance hardening

Today `Provenance` is documentation-only: a frozen dataclass with
`arxiv_id`, `section`, `verified_against_pdf`, and `notes`. One unit test
(`tests/test_turpin_chen.py::test_all_primary_cues_are_pdf_verified`)
enforces that every entry in the primary Chen catalog has
`verified_against_pdf=True`. That's it — a malicious or careless PR could
flip the flag on an unverified entry and nothing else would stop it.

Three hardening steps, in order of value:

1. **Registry-time assertion.** Every `Cue` / `BiasConfig` registered in
   a primary namespace (`tests.turpin_counterfactual`, `tests.chen_cue_injection`,
   future `tests.lanham.*` catalogs) must have `provenance.verified_against_pdf=True`
   at module-import time, or raise `ProvenanceError`. Unverified entries
   must live under `tests.extensions/`. This turns the one-off unit test
   into an always-on structural guarantee.

2. **CLI: `cotdiv provenance audit`.** Walk every registered metric /
   test / cue / bias, print a grouped table:
   - Paper-verified replications (arxiv_id set, verified=True)
   - Unverified replications (arxiv_id set, verified=False) — must be in
     extensions/
   - Extensions (arxiv_id None)
   One-command answer to "what does this library actually claim to implement."

3. **CI commit check.** Any commit whose diff flips a
   `verified_against_pdf` field from `False` to `True` must also touch a
   file under `docs/paper-refs/<arxiv_id>.md` (a per-paper cross-check
   log) or `AUDIT.md`. Prevents silent promotion without human PDF
   review. Implement as a `.github/workflows/provenance-check.yml`
   workflow running on pull_request events.

## Few-shot scaffolded cues

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

Would let us promote Visual Pattern back into the primary Chen catalog
with `verified_against_pdf=True`. Also unblocks any paper-verified cue
that depends on few-shot context.

## Dense-sweep filler_tokens reproduction

Current `filler_tokens` defaults to a sparse dyadic sweep `(0, 1, 2, 4,
8, 16, 32, 64, 128, 256)`. Lanham 2307.13702 §3.4 sweeps densely from 0
to max-sampled-CoT-length. For a published-number reproduction against
Lanham Fig 6, we'd want a dense sweep mode. Low priority — the
decisive claim ("zero uplift from filler at any length") survives a
sparse sweep.

## Inspect AI integration milestones

Driven by `docs/inspect_ai_integration_assessment.md`. v0.1 ships
**Scope A** (standalone CLI + 2 Inspect scorers + entry-point);
later versions extend.

### v0.1 (shipped 2026-04-26)

- `cotsuite.inspect.scorers.cot_legibility_coverage` — Emmons &
  Zimmermann legibility/coverage autorater as Inspect Scorer.
- `cotsuite.inspect.scorers.cot_post_hoc_rationalization` — per-
  trajectory PHR detector (NOT Arcuschin's full pair-construction
  IPHR methodology — strict subset; documented).
- `[project.entry-points.inspect_ai] cotsuite = "cotsuite.inspect"`
  — `inspect eval` discovers both scorers without further setup.
- `cotsuite.inspect._safety.warn_if_self_grading` — UserWarning
  when grader role resolves to the eval's primary model.
- `tests/test_inspect_integration.py` — 6 smoke tests for entry-
  point loading + registry name correctness.

### v0.1.1

- **g-mean² adoption** (OpenAI 2512.18311). Verified formula:
  `gmean2 = tpr * tnr` over (arm × correctness × monitor signal)
  triples. NOT directly computable from current cot-suite outputs;
  requires a binarization scheme on 0-4 Likert autorater outputs
  + a ground-truth proxy. Pre-work: 4-8 h methodology design;
  implementation: 2-4 h. Defer until methodology is locked.

### v0.2 (4-6 weeks post-launch)

- Port the remaining 4 metrics to Inspect Scorers:
  - Lanham early-answering (~6-10 h, cleanest of the four)
  - Lanham mistake-injection (~8-12 h, second grader role for
    the mistake generator)
  - Turpin counterfactual-bias (~12-18 h, needs Solver+Scorer
    pair pattern because the accuracy-drop is a cross-sample
    aggregate)
  - Chen cue-injection (~10-14 h, same Solver+Scorer pair
    pattern as Turpin)
- Confirm the post-2026-05-08 `inspect_evals` registry layout
  before opening any upstream PR (per inspect_evals PR #1538 by
  Scott-Simmons).

### v0.2.1+

- Open one `inspect_evals` PR — recommend legibility/coverage
  first, since it has the tightest reproduction story and is
  already implemented end-to-end.
- Use that review feedback to shape the remaining four metric
  PRs.

### Out of scope (probably forever)

- **Native Inspect-only architecture** (Scope C from the
  assessment). The standalone CLI's "score a pre-existing
  trajectory JSONL" workflow doesn't fit Inspect's eval-time-
  generation model. Replacing it would throw away expensive
  recorded outputs (Stage 1+2+3 multi-family results,
  ~$200 of compute).

## Length-weighted AOC alignment — promoted to `BLOCKERS.md`

Moved to `BLOCKERS.md` on 2026-04-19. This is a precondition for any
"reproduces Lanham Table 2" claim, not a future-work item.

## CI hygiene — pending

- **Bump GitHub Actions to Node.js 24 runtime** before June 2026.
  The `actions/checkout@v4`, `actions/setup-python@v5`,
  `actions/upload-artifact@v4`, `actions/download-artifact@v4`, and
  `codecov/codecov-action@v4` pins all run on the deprecating
  Node.js 20 runtime. CI logs already show deprecation warnings on
  every run. Current pin → next-major mapping (verify before bumping):
  `actions/checkout@v5`, `actions/setup-python@v6`,
  `actions/upload-artifact@v5`, `actions/download-artifact@v5`,
  `codecov/codecov-action@v5`. Single small PR.

## Process lessons

- **Pre-deploy vLLM smoke test.** Before any Modal deploy with a new
  inference stack, run a local `AsyncEngineArgs` smoke-test against a
  small model (e.g., Qwen2.5-0.5B on CPU) to catch API churn before
  burning GPU time. Rationale: on 2026-04-20, vllm 0.19.x had dropped
  the `disable_log_requests` kwarg — a 5s local `AsyncEngineArgs(...)`
  instantiation would have caught it before ~$0.18 of H100 crash-loop.
