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

## Length-weighted AOC alignment — BLOCKING for headline reproduction claim

`early_answering` and `mistake_injection` length-weight by `round(f*n)`
(sentence count at prefix fraction f). Lanham's paper weights by token
count through sentence k — noticeably different for variable-length
sentence distributions. For exact Table 2 reproduction we'd need to
swap in the token-count weighting.

**Required before any "reproduces Table 2" claim in outreach, blog
posts, or workshop paper.** If we ever write "CoT-Divergence reproduces
Lanham Table 2 within stated error bars," the weighting function has to
match the paper. Not urgent this week, but blocking before any headline
reproduction claim ships.
