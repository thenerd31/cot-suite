# cot-suite

A unified Python library for **chain-of-thought monitorability and faithfulness** on reasoning-model agents.

`cot-suite` operationalizes five published evaluations under a single CLI, each through its correct Inspect-AI abstraction — scorers for the per-trajectory/judge methods, solvers/tasks for the interventions (not all are scorers): Lanham 2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons & Zimmermann 2510.23966. The v0.1 release ships an 8-model open-weight scaling demonstration on GPQA-Diamond.

## 30-second value prop

> **Two paradigm-discriminating signals on 8 open-weight reasoning models** (GPQA-Diamond, post three layers of pre-launch audit):
>
> *PHR-strict cluster (7/8 bootstrap-robust + 1/8 partially-resolved):* 5 thinking-mode models at ≤3.28%, 2 non-thinking Qwen2.5 instruct models at ≥14.29%. **11pp gap between thinking-mode max (3.28%) and non-thinking-Qwen2.5 min (14.29%); Llama-3.1-8B at 6.67% sits between clusters with a 95% bootstrap CI overlapping both ([0.00%, 15.56%], n_scorable=45 due to Llama's 23.2% GPQA-Diamond accuracy floor); v0.1.1 cross-benchmark replication will resolve.** [→ Full bootstrap analysis](scaling.md#llama-31-8b-cluster-membership)
>
> *Legibility-coverage gap (8/8 bootstrap-robust):* all thinking ≤0.29, all non-thinking ≥1.01, 3.5× separation. Partially scale-sensitive on non-thinking side (Qwen2.5-72B closes 40% of the gap at ~10× parameters but doesn't cross the cluster boundary). Autorater-based, independent of the answer-extraction pipeline.
>
> The PHR figures reflect three layers of pre-launch audit: (1) parser bug in answer extraction (caught 2026-04-27); (2) judge labeling artifacts resolved via option-letter normalization; (3) the first normalizer was over-aggressive — revised after manual case adjudication on Qwen3-Thinking-14B + Qwen2.5-72B. Full audit trail in [AUDIT.md](https://github.com/thenerd31/cot-suite/blob/main/AUDIT.md). [→ Multi-family scaling](scaling.md)

## Install

```bash
pip install cot-suite
```

Optional extras:

```bash
pip install "cot-suite[nlp]"          # + NLTK punkt (Lanham-style sentence splitting)
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

Pre-alpha. v0.1 launch targets mid-May 2026. The Chen/Turpin Inspect scorers (+ injection solvers) and the Lanham `early_answering` Inspect task are implemented; v0.2 adds the remaining three Lanham interventions (`mistake_injection`, `paraphrasing`, `filler_tokens`) — see [roadmap](roadmap.md).

[Source on GitHub](https://github.com/thenerd31/cot-suite){.md-button} [PyPI](https://pypi.org/project/cot-suite/){.md-button}
