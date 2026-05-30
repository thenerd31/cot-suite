# cot-suite

cot-suite is an Inspect AI-native library for chain-of-thought monitorability evaluation on reasoning-model agents.

**Status:** Pre-launch, targeting late-July to mid-August 2026.

> Renamed from `cot-divergence` on 2026-04-21. The old `cotdiv` import path still works via a shim until 2026-07-21.

## What it does

The library bundles four contributions:

1. **Paper-method implementations** of foundational CoT-monitorability methodologies, each exposed through its correct Inspect-AI abstraction — **scorers** for the per-trajectory / judge methods, **solvers/tasks** for the intervention methods (see [Modules](#modules) and [`ROADMAP.md`](ROADMAP.md); not every method fits Inspect's `Scorer`). Each cites its source paper; **two are against-release reproductions** (Turpin cell-for-cell ±0.08pp; Arcuschin IPHR integer-exact ±0 for 9 models), and the rest are method-implementations applied to current models, with reproduction status stated honestly per paper:
   - **Turpin [2305.04388](https://arxiv.org/abs/2305.04388)** — **reproduction** (cell-for-cell, ±0.08pp against Turpin's released `bbh_samples`).
   - **Arcuschin & Janiak [2503.08679](https://arxiv.org/abs/2503.08679)** — **integer-exact (±0) against-release reproduction** of the IPHR per-model unfaithfulness rates for 9 models (4 of 7 paper-headline cells: gpt-4o-mini 13.49%, claude-3.5-haiku 7.42%, gemini-2.5-flash 2.17%, claude-3.7-sonnet_1k 0.04%), by independently reimplementing the three IPHR criteria (`cotsuite.tests.iphr`) and metric-replaying ChainScope's released df — verified against ChainScope's own computed counts. 7 oversampled cells are blocked — 3 of them headline (chatgpt-4o-latest, deepseek-r1, gemini-2.5-pro) — by unreconstructable adaptive oversampling, not an implementation limit. Also ships a per-trajectory PHR detector — a different, narrower signal.
   - **Emmons & Zimmermann [2510.23966](https://arxiv.org/abs/2510.23966)** — implements the legibility/coverage autorater (E-Z Appendix C prompt), applied to a capability-diverse model set with cross-judge validation; from-spec reproduction of E-Z's Table-1 cells not yet performed.
   - **Yanda Chen [2505.05410](https://arxiv.org/abs/2505.05410)** — implements the six-cue verbalization method (directional; original models retired, no public code/data release).
   - **Lanham [2307.13702](https://arxiv.org/abs/2307.13702)** — implements the four faithfulness tests (directional; Claude 1.3 retired, no public release, AOC estimator underspecified).
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
- **`cotsuite.tests.iphr`** — Arcuschin 2503.08679 pair-level IPHR metric (group-bias → accuracy-diff → direction criteria); cot-suite's independent reimplementation used for the B4a integer-exact against-release reproduction. Distinct construct from the per-trajectory detector above.
- **`cotsuite.core.classify`** — faithfulness classification dispatcher with strict near-zero thresholds (`computational`, `rationalization`, `mixed`, `unknown`).
- **`cotsuite.core.provenance`** — every test / cue / metric carries a `Provenance` record; unverified or extension work lives in `tests/extensions/` until PDF cross-check.
- **`cotsuite.inspect.scorers`** — Inspect AI scorers. Two ship in v0.1: `cot_legibility_coverage` (Emmons & Zimmermann) and `cot_post_hoc_rationalization` (per-trajectory Arcuschin signal — strict subset of the paper's full pair-construction IPHR methodology). v0.2 adds the remaining methods **in their correct abstractions, not all as scorers**: Chen cue-verbalization and Turpin bias-verbalization as **scorers** (each paired with a cue/bias-injection **solver**); Lanham's four tests as Inspect **tasks/solvers**, *not* scorers (mid-trajectory interventions + per-item AOC don't fit `score(state, target)`); Turpin's `accuracy_drop` stays a **dataset-level** metric (the ±0.08pp-validated B2 path), not a `Score`. See [`ROADMAP.md`](ROADMAP.md). Auto-discoverable via `[project.entry-points.inspect_ai]`: after `pip install cot-suite`, `inspect eval some_task --scorer cotsuite/cot_legibility_coverage` works without further setup. Self-grading guard fires a `UserWarning` if Inspect's grader role resolves to the eval's primary model.

Full methodology and shortcut disclosures in [`AUDIT.md`](AUDIT.md). Known pre-release blockers in [`BLOCKERS.md`](BLOCKERS.md). Roadmap in [`ROADMAP.md`](ROADMAP.md).

## Related work

cot-suite tracks Young's trilogy ([2603.20172](https://arxiv.org/abs/2603.20172), [2603.22582](https://arxiv.org/abs/2603.22582), [2603.26410](https://arxiv.org/abs/2603.26410)) as *concurrent* work; that line motivates the cross-classifier sensitivity reporting at the core of this library. MonitorBench (ASTRAL-Group [2603.28590](https://arxiv.org/abs/2603.28590)) supplies the 1,514-instance task surface we load natively. OpenAI's monitorability-evals (the companion release to "Monitoring Monitorability," [2512.18311](https://arxiv.org/abs/2512.18311)) supplies the g-mean² metric and three-archetype eval taxonomy we interoperate with. The faithfulness scorers **implement the methods of** Lanham 2307.13702, Turpin 2305.04388, Yanda Chen 2505.05410, and Arcuschin & Janiak 2503.08679, and the autorater ports Emmons & Zimmermann 2510.23966. **Two are against-release reproductions:** Turpin is reproduced cell-for-cell (±0.08pp), and Arcuschin's IPHR rates are reproduced integer-exact (±0) for 9 models against ChainScope's released df (`jettjaniak/chainscope`, MIT). The rest are method-implementations applied to current models, with per-paper reproduction status in the contributions list above. See [`docs/related_work.md`](docs/related_work.md) for the full landscape.

## Install

```bash
pip install cot-suite                 # core
pip install "cot-suite[nlp]"          # + NLTK punkt (Lanham-style sentence splitting)
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
