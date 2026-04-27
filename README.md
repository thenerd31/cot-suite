# cot-suite

A unified Python library for **chain-of-thought monitorability and faithfulness** on reasoning-model agents.

**Status:** pre-alpha. Targeting a public launch mid-May 2026. Stage 1 validation run (Qwen3-14B × GPQA-Diamond × Haiku 4.5 autorater) complete; results and methodology in [`benchmarks/results/qwen3_14b_gpqa_full/`](benchmarks/results/qwen3_14b_gpqa_full/).

> Renamed from `cot-divergence` on 2026-04-21. The old `cotdiv` import path still works via a shim until 2026-07-21.

## What it does

`cot-suite` treats two closely-related but distinct questions as one evaluation surface:

- **Monitorability** — can a human (or an LLM monitor) read a model's chain-of-thought and catch bad reasoning before the model acts on it? This is the near-term AI-safety bet laid out in Korbak et al. [2507.11473](https://arxiv.org/abs/2507.11473) and operationalized by Emmons & Zimmermann et al. [2510.23966](https://arxiv.org/abs/2510.23966).
- **Faithfulness** — does a model's CoT *actually* reflect how it reached its final answer, or is the stated reasoning a post-hoc rationalization? Lanham [2307.13702](https://arxiv.org/abs/2307.13702), Turpin [2305.04388](https://arxiv.org/abs/2305.04388), Chen/Benton/Perez [2505.05410](https://arxiv.org/abs/2505.05410), and Arcuschin [2503.08679](https://arxiv.org/abs/2503.08679) each probe this from different angles.

A legible CoT can still be unfaithful; a faithful CoT can still be illegible. Evaluating one without the other paints half the picture. `cot-suite` ships both as first-class, Inspect-AI-native primitives, with a versioned autorater port of the 2510.23966 Appendix C prompt at the center and the faithfulness literature as complementary modules around it.

## Modules

- **`cotsuite.autoraters`** — verbatim port of the Emmons & Zimmermann 2510.23966 Appendix C autorater (legibility + coverage), SHA-256-hashed for reproducibility.
- **`cotsuite.tests.lanham`** — Lanham 2307.13702 four-test suite (early answering, mistake injection, paraphrasing, filler tokens).
- **`cotsuite.tests.turpin_counterfactual`** — Turpin 2305.04388 counterfactual bias battery.
- **`cotsuite.tests.chen_cue_injection`** — Chen 2505.05410 six-hint cue-injection catalog (five verified from the paper's Table 1, Visual Pattern in extensions pending a few-shot scaffold).
- **`cotsuite.tests.post_hoc_rationalization`** — Arcuschin 2503.08679 implicit-rationalization detector (CoT conclusion vs final-answer divergence via LLM-as-judge).
- **`cotsuite.core.classify`** — faithfulness classification dispatcher with strict near-zero thresholds (`computational`, `rationalization`, `mixed`, `unknown`).
- **`cotsuite.core.provenance`** — every test / cue / metric carries a `Provenance` record; unverified or extension work lives in `tests/extensions/` until PDF cross-check.
- **`cotsuite.inspect.scorers`** — Inspect AI scorers. Two ship in v0.1: `cot_legibility_coverage` (Emmons & Zimmermann) and `cot_post_hoc_rationalization` (per-trajectory Arcuschin signal — strict subset of the paper's full pair-construction IPHR methodology). Three more (Lanham, Turpin, Chen) coming in v0.2 — see [`ROADMAP.md`](ROADMAP.md). Auto-discoverable via `[project.entry-points.inspect_ai]`: after `pip install cot-suite`, `inspect eval some_task --scorer cotsuite/cot_legibility_coverage` works without further setup. Self-grading guard fires a `UserWarning` if Inspect's grader role resolves to the eval's primary model.

Full methodology and shortcut disclosures in [`AUDIT.md`](AUDIT.md). Known pre-release blockers in [`BLOCKERS.md`](BLOCKERS.md). Roadmap in [`ROADMAP.md`](ROADMAP.md).

## Install

```bash
pip install cot-suite                 # core
pip install "cot-suite[nlp]"          # + NLTK punkt (Lanham paper-faithful sentence splitting)
pip install "cot-suite[langgraph]"    # + LangGraph middleware
pip install "cot-suite[activations]"  # + nnsight / TransformerLens (open-weights only)
```

The legacy name still resolves:

```bash
pip install cot-divergence  # aliased to cot-suite until 2026-07-21
```

## Quickstart

```python
from cotsuite import score_trajectory
from cotsuite.adapters import from_anthropic

traj = from_anthropic(messages, model="claude-opus-4-5")
result = score_trajectory(
    traj,
    metrics=["legibility", "coverage"],
    autorater="anthropic/claude-haiku-4-5",  # or "google/gemini-2.5-pro"
)
print(f"legibility = {result.metrics['legibility'].value:.2f} ± {result.metrics['legibility'].stderr:.2f}")
```

## Citation

See [`CITATION.cff`](CITATION.cff) (DOI pending v1.0). In the interim, cite the repo URL.

## License

MIT.
