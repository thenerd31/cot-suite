# cot-suite

A unified Python library for **chain-of-thought monitorability and faithfulness** on reasoning-model agents.

`cot-suite` operationalizes five published evaluations as scorers under a single CLI: Lanham 2307.13702, Turpin 2305.04388, Chen 2505.05410, Arcuschin 2503.08679, and Emmons & Zimmermann 2510.23966. The v0.1 release ships an 8-model open-weight scaling demonstration on GPQA-Diamond.

## 30-second value prop

> **Two findings.** *PHR rate is paradigm-locked:* all 5 thinking-mode models stay at ≤5.74%, all 3 non-thinking instruct models exceed 22%, with no scale rescue (Qwen2.5-72B at 22.62% vs Qwen2.5-7B at 20%). *The legibility-coverage gap is paradigm-dominated but partially scale-sensitive:* Qwen2.5-72B closes the gap to 1.012 (39% smaller than Qwen2.5-7B's 1.65 at ~10× the parameters), but does not cross into thinking-mode territory (≤0.29 across all 5 thinking-mode models). **Cluster separation: 5.3× on the PHR axis, 3.5× on the gap axis, 8/8 models in their predicted quadrant.** [→ Multi-family scaling](scaling.md)

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
