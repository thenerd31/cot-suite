# cot-suite

A unified Python library for **chain-of-thought monitorability and faithfulness** on reasoning-model agents.

`cot-suite` operationalizes five published evaluations as scorers under a single CLI: Lanham 2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons & Zimmermann 2510.23966. The v0.1 release ships an 8-model open-weight scaling demonstration on GPQA-Diamond.

## 30-second value prop

> **Two paradigm signals.** *PHR-strict cluster:* 5 thinking-mode models at ≤4.72%, 2 non-thinking models (Qwen2.5-7B, Qwen2.5-72B) at ≥14.29% — **bootstrap-robust 7/8 cluster membership** with a 9.57pp absolute non-overlap band. Llama-3.1-8B is **partially-resolved at v0.1 sample size** (n=46 correct trajectories from its 23.2% GPQA-Diamond accuracy floor; 95% bootstrap CI [4.35%, 23.91%] crosses the boundary by 0.37pp on the lower bound; v0.1.1 grows n via cross-benchmark replication). *Legibility-coverage gap:* all thinking-mode ≤0.29, all non-thinking ≥1.01 — **8/8 cluster membership**, 3.5× separation, partially scale-sensitive (Qwen2.5-72B closes 40% of the gap at ~10× parameters, doesn't cross the boundary). [→ Multi-family scaling](scaling.md) · [pre-launch parser-bug discovery + bootstrap analysis in AUDIT.md](https://github.com/thenerd31/cot-suite/blob/main/AUDIT.md)

## Install

```bash
pip install cot-suite
```

Optional extras:

```bash
pip install "cot-suite[nlp]"          # + NLTK punkt (Lanham paper-faithful sentence splitting)
pip install "cot-suite[langgraph]"    # + LangGraph middleware
pip install "cot-suite[activations]"  # + nnsight / TransformerLens (open-weights only)
```

## Five-line minimum-viable example

```python
from cotsuite import score_trajectory
from cotsuite.adapters import from_anthropic

traj = from_anthropic(messages, model="claude-opus-4-5")
result = score_trajectory(traj, metrics=["legibility", "coverage"], autorater="anthropic/claude-haiku-4-5")
print(f"legibility = {result.metrics['legibility'].value:.2f} ± {result.metrics['legibility'].stderr:.2f}")
```

## Built on Inspect AI

`pip install cot-suite` auto-registers `cot_legibility_coverage` and `cot_post_hoc_rationalization` as Inspect scorers via the `inspect_ai` entry-point group — usable directly from `inspect eval` without any additional setup:

```bash
inspect eval some_task.py --scorer cotsuite/cot_legibility_coverage
```

[→ Quickstart](quickstart.md) · [→ Inspect AI integration](inspect.md)

## What's in v0.1

- **Five paper-verified evaluation methodologies** under one CLI ([metrics overview](metrics/index.md))
- **Two Inspect AI scorers** auto-discoverable via entry-point ([Inspect AI integration](inspect.md))
- **8-model multi-family scaling table** on GPQA-Diamond ([scaling results](scaling.md))
- **Reproducibility contract**: every Score row carries `eval_version`, `cotsuite_version`, `prompt_version`, `prompt_sha256`, model spec ([reproducibility](reproducibility.md))

## Status

Pre-alpha. v0.1 launch targets mid-May 2026. v0.2 (Lanham/Turpin/Chen Inspect scorers) follows 4-6 weeks after launch — see [roadmap](roadmap.md).

[Source on GitHub](https://github.com/thenerd31/cot-suite){.md-button} [PyPI](https://pypi.org/project/cot-suite/){.md-button}
