# Quickstart

Score a stored trajectory in three steps. The example uses the GPQA-Diamond Qwen3-Thinking-14B traces from the v0.1 multi-family scaling run.

## 1. Install

```bash
pip install cot-suite
```

## 2. Set an autorater key

`cot-suite` autoraters use Anthropic, OpenAI, or Google models via Inspect AI's provider syntax. For the default Haiku 4.5 autorater:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

`cotsuite verify-keys` validates the keys before you spend any tokens:

```bash
cotsuite verify-keys
```

## 3. Score a stored trajectory

`cotsuite` reads trajectories from the JSONL format produced by its own `bench` runs (one JSON object per line, each with a `messages` array and a `final_answer` field):

```python
import json
from cotsuite import score_trajectory
from cotsuite.adapters import from_anthropic

with open("benchmarks/results/qwen3_14b_gpqa_full/trajectories.jsonl") as f:
    rows = [json.loads(line) for line in f]

# Score the first 5 rows on legibility + coverage + post-hoc rationalization.
for row in rows[:5]:
    traj = from_anthropic(row["messages"], model=row["model_id"])
    result = score_trajectory(
        traj,
        metrics=["legibility", "coverage", "post_hoc_rationalization"],
        autorater="anthropic/claude-haiku-4-5",
        runs=5,
    )
    print(f"{row['question_id']}: leg={result.metrics['legibility'].value:.2f} "
          f"cov={result.metrics['coverage'].value:.2f} "
          f"phr_strict={result.metrics['post_hoc_rationalization'].value:.2f}")
```

Each `MetricValue` carries `value`, `stderr`, `stddev`, `n_runs`, plus `metadata` with the prompt SHA-256, autorater spec, and per-run justifications. See [reproducibility](reproducibility.md) for the full metadata contract.

## Inspect AI variant

If you're already running Inspect evals, the scorer factories are auto-discovered after `pip install cot-suite`:

```python
from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.dataset import hf_dataset
from cotsuite.inspect.scorers import cot_legibility_coverage, cot_post_hoc_rationalization

@task
def gpqa_diamond_monitorability():
    return Task(
        dataset=hf_dataset("Idavidrein/gpqa", split="diamond"),
        solver=generate(),
        scorer=[
            cot_legibility_coverage(autorater="anthropic/claude-haiku-4-5"),
            cot_post_hoc_rationalization(judge="anthropic/claude-haiku-4-5"),
        ],
    )
```

Run it with:

```bash
inspect eval gpqa_diamond_monitorability.py \
  --model openai/gpt-5 \
  -M grader=anthropic/claude-haiku-4-5
```

The `-M grader=...` flag is the recommended way to specify the autorater — it routes through Inspect's model-role mechanism. Without it, the eval's primary model is used as its own grader, which collapses LLM-as-judge measurements into self-evaluation. cot-suite emits a `UserWarning` in that case via `cotsuite.inspect._safety.warn_if_self_grading`.

## Next steps

- [Metrics overview](metrics/index.md) — what each of the five wrapped methodologies measures
- [Inspect AI integration](inspect.md) — how the entry-point hook + scorer factories work
- [Multi-family scaling](scaling.md) — the v0.1 8-model GPQA-Diamond results
