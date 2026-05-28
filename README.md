# cot-suite

cot-suite is an Inspect AI-native library for chain-of-thought monitorability evaluation on reasoning-model agents.

**Status:** Pre-launch, targeting late-July to mid-August 2026.

> Renamed from `cot-divergence` on 2026-04-21. The old `cotdiv` import path still works via a shim until 2026-07-21.

## What it does

The library bundles four contributions:

1. **Paper-faithful reproductions** of foundational CoT-monitorability methodologies as Inspect scorers — Lanham [2307.13702](https://arxiv.org/abs/2307.13702), Turpin [2305.04388](https://arxiv.org/abs/2305.04388), Yanda Chen [2505.05410](https://arxiv.org/abs/2505.05410), Arcuschin & Janiak [2503.08679](https://arxiv.org/abs/2503.08679), Emmons & Zimmermann [2510.23966](https://arxiv.org/abs/2510.23966). Each scorer cites its source paper, vendors released artifacts where available, and ships with a documented delta against the paper's reported cells.
2. **Cross-classifier sensitivity reporting by default.** Every faithfulness scorer runs against at least two classifiers (regex pipeline + LLM judge) and reports per-judge scores, Cohen's κ, and ranking-reversal warnings. Motivated by Young [2603.20172](https://arxiv.org/abs/2603.20172), which showed per-model gaps up to 30.6pp across classifiers, with model-ranking reversals, on the same trajectories — making single-number faithfulness measurements methodologically unreliable.
3. **Inspect-native interop** with OpenAI's monitorability-evals (Apr 2026) — supports g-mean² with cross-fit filter, the three eval archetypes (intervention / process / outcome-property), and the released datasets via Inspect adapters.
4. **An Inspect-native MonitorBench task loader** (ASTRAL-Group [2603.28590](https://arxiv.org/abs/2603.28590), 1,514 instances) and a roadmap for additional task surfaces (ChainScope, CoT-Control).

Built to support the kind of external review of frontier-lab monitorability claims that OpenAI established as a precedent in their May 2026 accidental-CoT-grading disclosure.

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

## Related work

cot-suite tracks Young's trilogy ([2603.20172](https://arxiv.org/abs/2603.20172), [2603.22582](https://arxiv.org/abs/2603.22582), [2603.26410](https://arxiv.org/abs/2603.26410)) as *concurrent* work; that line motivates the cross-classifier sensitivity reporting at the core of this library. MonitorBench (ASTRAL-Group [2603.28590](https://arxiv.org/abs/2603.28590)) supplies the 1,514-instance task surface we load natively. OpenAI's monitorability-evals (the companion release to "Monitoring Monitorability," [2512.18311](https://arxiv.org/abs/2512.18311)) supplies the g-mean² metric and three-archetype eval taxonomy we interoperate with. The faithfulness scorers reproduce Lanham 2307.13702, Turpin 2305.04388, Yanda Chen 2505.05410, and Arcuschin & Janiak 2503.08679, while the autorater ports Emmons & Zimmermann 2510.23966. See [`docs/related_work.md`](docs/related_work.md) for the full landscape.

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
