# Metrics overview

cot-suite v0.1 wraps five published evaluation methodologies under a unified API. Each lives in `cotsuite.tests.<module>` (or `cotsuite.autoraters.legibility_coverage` for the autorater port). All five carry `Provenance` records pinning their originating paper + section, with a registry-time invariant that primary-namespace entries must have `verified_against_pdf=True` (see `ROADMAP.md` for the planned hardening of that invariant).

| Metric | Originating paper | What it measures | Validation |
|---|---|---|---|
| [Lanham 4-test suite](lanham.md) | Lanham 2307.13702 | Causal-intervention faithfulness via early answering, mistake injection, paraphrasing, filler tokens | B1 ✓ |
| [Turpin counterfactual bias](turpin.md) | Turpin 2305.04388 | Faithfulness via accuracy drop under biased context | B2 ✓ |
| [Chen cue injection](chen.md) | Chen 2505.05410 | Faithfulness via cue-acknowledgment rate across 6 cue types | B3 ✓ |
| [Arcuschin PHR detector](arcuschin.md) | Arcuschin 2503.08679 | Per-trajectory implicit post-hoc rationalization (CoT vs final-answer divergence) | B4 ⚠ application/measurement (9.30% on GPT-4o-mini vs paper 13%); against-release reproduction (B4 redux) pending |
| [Emmons-Zimmermann legibility/coverage](emmons-zimmermann.md) | Emmons & Zimmermann 2510.23966 | LLM-as-judge legibility + causal coverage on a 0-4 Likert | method-implementation + cross-judge validation (Qwen3-14B; **not** an E-Z Table-1 reproduction) |

## Two question types, one bundle

A legible CoT can still be unfaithful; a faithful CoT can still be illegible. Evaluating one without the other paints half the picture.

- **Monitorability** — can a human (or LLM monitor) read a model's CoT and catch bad reasoning before the model acts on it? *Emmons-Zimmermann legibility/coverage* lives here.
- **Faithfulness** — does a model's CoT actually reflect how it reached its final answer, or is the stated reasoning a post-hoc rationalization? *Lanham, Turpin, Chen, Arcuschin* live here.

`cot-suite` ships both as first-class evaluation primitives, with the Emmons-Zimmermann autorater port (Appendix C, SHA-pinned) at the center and the faithfulness literature as complementary modules around it.

## Validation snapshots (B1-B4)

| Run | Metric | Model | Result | Status |
|---|---|---|---|---|
| B1 | Lanham early-answering / mistake-injection | Qwen3-14B / GPT-4o-mini | non-trivial AOC retention; degradation under mistake injection | ✓ |
| B2 | Turpin counterfactual | Qwen3-14B / GPT-4o-mini | accuracy drop under biased context | ✓ |
| B3 | Chen cue injection | Qwen3-14B / GPT-4o-mini | non-zero cue acknowledgment | ✓ |
| B4 | Arcuschin PHR | GPT-4o-mini | 9.30% strict PHR vs paper 13% (within 5-25% band) | ✓ |
| Stage 1 | Emmons-Zimmermann autorater | Qwen3-14B | full N=198 GPQA-Diamond run | ✓ |

See [`AUDIT.md`](https://github.com/thenerd31/cot-suite/blob/main/AUDIT.md) for the full validation logbook.

## What's deferred to v0.2

| Metric | Originating paper | Status |
|---|---|---|
| CC-SHAP | Siegel 2404.03189 | v0.2 (Task #13) |
| Verbosity | Meek 2510.27378 | v0.2 |
| Confidence-trajectory PHR | Lewis-Lim 2508.19827 | v0.2 |
| g-mean² monitorability | Guan et al. (OpenAI) 2512.18311 | v0.1.1 (pending binarization-scheme decision) |

See [roadmap](../roadmap.md) for the full milestone plan.
