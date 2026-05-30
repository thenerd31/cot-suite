# Inspect AI integration

cot-suite ships native Inspect AI scorers for two of the five wrapped methodologies in v0.1, with the remaining three planned for v0.2.

## What's in v0.1

| Scorer | Methodology | Score shape |
|---|---|---|
| `cotsuite/cot_legibility_coverage` | Emmons & Zimmermann 2510.23966 | Dict-valued: `{legibility, coverage, passed}` |
| `cotsuite/cot_post_hoc_rationalization` | Arcuschin 2503.08679 (per-trajectory subset) | Binary: 1.0 = strict PHR, 0.0 = otherwise, NaN = unscorable |

Both are auto-discovered after `pip install cot-suite` via the entry-point group:

```toml
[project.entry-points.inspect_ai]
cotsuite = "cotsuite.inspect"
```

Importing `cotsuite.inspect` fires every `@scorer` decorator and populates Inspect's process-global registry, so this works without further setup:

```bash
inspect eval some_task.py --scorer cotsuite/cot_legibility_coverage
```

## Self-grading guard

Inspect's `get_model(role="grader")` returns the model the user specified via `-M grader=<spec>` on the CLI. If they forget, it silently falls back to the eval's primary model — which means the model under test would be grading its own CoT. That collapses LLM-as-judge measurements into self-evaluation.

cot-suite's `cotsuite.inspect._safety.warn_if_self_grading` fires a once-per-process `UserWarning` when this happens, with explicit instructions to pass `-M grader=<provider>/<model>` or the scorer-specific `autorater=`/`judge=` kwarg.

## Score metadata

Both v0.1 scorers stamp the following keys into `Score.metadata` for downstream reproducibility (see [reproducibility](reproducibility.md) for the full contract):

- `eval_version` — bumped on methodology changes that break numeric comparability across runs (currently `"1.0.0"` for both)
- `cotsuite_version` — the cotsuite package version that produced the score
- `prompt_version` — the Appendix C / PHR prompt version (e.g. `emmons_zimmermann_v1`)
- `prompt_sha256` — SHA-pinned prompt content for drift detection
- `autorater` / `judge_model` — provider-prefixed model spec
- `runs`, `justifications` (legibility/coverage only) — per-run rationales for cross-checking

## Examples

### Legibility/coverage Inspect task

```python
from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.dataset import hf_dataset
from cotsuite.inspect.scorers import cot_legibility_coverage

@task
def gpqa_diamond_faithfulness():
    return Task(
        dataset=hf_dataset("Idavidrein/gpqa", split="diamond"),
        solver=generate(),
        scorer=cot_legibility_coverage(autorater="google/gemini-2.5-pro"),
    )
```

### Per-trajectory PHR Inspect task

```python
from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.dataset import hf_dataset
from cotsuite.inspect.scorers import cot_post_hoc_rationalization

@task
def gpqa_diamond_phr():
    return Task(
        dataset=hf_dataset("Idavidrein/gpqa", split="diamond"),
        solver=generate(),
        scorer=cot_post_hoc_rationalization(judge="anthropic/claude-haiku-4-5"),
    )
```

### Both at once

```python
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

## v0.2 roadmap

The remaining methods, **in their correct Inspect abstraction** (Stage-0 recon, 2026-05-30 — not all fit `Scorer`). Status:

| Method | Methodology | Abstraction | Status |
|---|---|---|---|
| Chen cue-injection | Chen 2505.05410 | **Solver + Scorer** | ✅ `cot_chen_cue_injection` + `cot_cue_injection_solver` |
| Turpin counterfactual | Turpin 2305.04388 | **Solver + Scorer** | ✅ `cot_turpin_counterfactual` + `cot_bias_injection_solver`; `accuracy_drop` stays dataset-level (B2 path) |
| Lanham `early_answering` | Lanham 2307.13702 §3.1 | **Task / Solver — NOT Scorer** | ✅ POC — `cot_lanham_early_answering` solver + `cot_lanham_early_answering_aoc` scorer (single model) |
| Lanham `mistake_injection` | Lanham §3.2 | **Task / Solver** | ⏳ v0.2 — needs a 2nd model role (mistake generator) |
| Lanham `paraphrasing` | Lanham §3.3 | **Task / Solver** | ⏳ v0.2 — needs a 2nd model role (paraphraser) |
| Lanham `filler_tokens` | Lanham §3.4 | **Task / Solver** | ⏳ v0.2 (single model) |

Confirm the post-2026-05-08 `inspect_evals` registry layout before opening any upstream PR (per `inspect_evals` PR #1538 by Scott-Simmons). The first upstream PR will be legibility/coverage — the cleanest implementation to demonstrate for upstreaming, already implemented end-to-end. (It is not the project's tightest *reproduction* — that is Turpin, cell-for-cell ±0.08pp; the legibility/coverage autorater is a method-implementation, with from-spec E-Z reproduction deferred.)

See [`ROADMAP.md`](https://github.com/thenerd31/cot-suite/blob/main/ROADMAP.md) for v0.2 / v0.2.1+ milestones.

## Out of scope

**Native Inspect-only architecture** (Scope C from `docs/inspect_ai_integration_assessment.md`). The standalone CLI's "score a pre-existing trajectory JSONL" workflow doesn't fit Inspect's eval-time-generation model. Replacing it would discard the v0.1 multi-family scaling outputs (~$200 of compute). The v0.1 architecture is **Scope A**: standalone CLI + entry-point hook + a small set of Inspect scorers as a thin wrapper over the underlying methodologies.
