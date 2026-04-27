# Emmons-Zimmermann legibility/coverage

**Originating paper:** Emmons & Zimmermann et al., "A Pragmatic Way to Measure Chain-of-Thought Monitorability" — [arXiv 2510.23966](https://arxiv.org/abs/2510.23966).

**What it measures.** LLM-as-judge **legibility** and **causal coverage** on a 0-4 Likert scale, computed by a separate autorater model that reads the CoT and answers a structured rubric.

- **Legibility** — is the CoT readable, well-structured, and free of incoherent jumps?
- **Coverage** — does the CoT enumerate the causal factors required to answer the question?

A monitorable CoT scores high on both. A legible-but-not-faithful CoT scores high on legibility but low on coverage (the CoT reads cleanly but doesn't actually do the work). An illegible CoT scores low on legibility regardless of what's underneath.

**Module:** `cotsuite.autoraters.legibility_coverage`. Validation status: **Stage 1 reproduction** — full N=198 GPQA-Diamond run on Qwen3-14B with Claude Haiku 4.5 as autorater.

## The Appendix C prompt

The autorater uses the verbatim Appendix C prompt from the paper. The prompt file ships at `cotsuite/autoraters/prompts/emmons_zimmermann_v1.txt`, with its canonical SHA-256 (`ac1e0ac4044b0a64…`) committed alongside in `emmons_zimmermann_v1.sha256` and cross-referenced in `docs/paper-refs/2510.23966-appendix-c.md`.

The prompt is **versioned, not mutated.** To update — ship a new version (`emmons_zimmermann_v1_1`, `emmons_zimmermann_v2`), never modify the shipped file. This invariant is enforced by the SHA-256 drift check in `tests/test_appendix_c_prompt_integrity.py`.

## API signature

The autorater can be invoked directly:

```python
from cotsuite.autoraters.legibility_coverage import legibility_coverage

leg, cov = await legibility_coverage(
    trajectory=traj,
    autorater="google/gemini-2.5-pro",       # or "anthropic/claude-haiku-4-5"
    runs=5,
    prompt_version="emmons_zimmermann_v1",
)
# leg.value — mean legibility 0-4
# leg.stderr — standard error over runs
# leg.metadata.justifications — per-run justification strings
# leg.metadata.prompt_sha256 — pinned to "ac1e0ac4044b0a64..."
```

Or via the unified `score_trajectory` wrapper:

```python
from cotsuite import score_trajectory

result = score_trajectory(
    traj,
    metrics=["legibility", "coverage"],
    autorater="anthropic/claude-haiku-4-5",
    runs=5,
)
print(f"legibility = {result.metrics['legibility'].value:.2f}")
print(f"coverage   = {result.metrics['coverage'].value:.2f}")
```

## Example

```python
from cotsuite import score_trajectory
from cotsuite.adapters import from_openai

traj = from_openai(messages, model="gpt-5", final_answer="42")
result = score_trajectory(
    traj,
    metrics=["legibility", "coverage"],
    autorater="anthropic/claude-haiku-4-5",
    runs=5,
)
# Stage 1 reference numbers on Qwen3-14B / GPQA-Diamond:
#   legibility ≈ 3.61, coverage ≈ 3.42, leg-cov gap ≈ 0.19
# (See scaling.md for the full 8-model table.)
```

## Known limitations

- **Self-grading is a research footgun.** If the autorater happens to be the same model as the one being scored, the judgment collapses into self-evaluation. cot-suite emits a `UserWarning` via `cotsuite.inspect._safety.warn_if_self_grading` when Inspect's grader role resolves to the eval's primary model. If you're invoking the autorater directly, you're responsible for picking an independent model.
- **Autorater drift.** The Appendix C prompt is SHA-pinned, but the *autorater model* is not — same prompt against `claude-haiku-4-5` and `gemini-2.5-pro` will give different absolute numbers (within roughly 0.2-0.4 on a 0-4 scale, per the v0.1 detector ablation). For cross-paper comparison, fix the autorater model + pin the prompt SHA in your reporting.
- **0-4 Likert is not a continuous metric.** The mean over 5 runs gives meaningful precision only if the underlying ratings vary; pegged ratings (all 4s, all 0s) are less informative than the absolute values suggest.
- **Single-question framing.** The autorater scores one CoT against one question. Multi-turn agent trajectories with tool calls are reduced to "everything before the final answer is the CoT, everything in the final user-facing message is the answer" — that's a simplification of agent state.

## Inspect AI integration (v0.1, shipped)

`cotsuite.inspect.scorers.cot_legibility_coverage` is one of the two Inspect AI scorers shipping in v0.1. Returns a dict-valued Score with `legibility`, `coverage`, and `passed` (legibility ≥ 3.0) sub-metrics:

```bash
inspect eval some_task.py \
  --scorer cotsuite/cot_legibility_coverage \
  -M grader=anthropic/claude-haiku-4-5
```

See [Inspect AI integration](../inspect.md) for the full integration story.
