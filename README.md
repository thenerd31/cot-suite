# cot-divergence

Chain-of-thought faithfulness evaluation for reasoning-model agents.

**Status:** pre-alpha. Targeting v0.1 (2510.23966 autorater port) late October 2026; v1.0 mid-February 2027.

## What it does

`cot-divergence` packages the published faithfulness literature into a single pip-installable library:

- **Legibility + Coverage** — Emmons & Zimmermann 2510.23966 autorater, versioned and testable.
- **Lanham four-test suite** — early answering, mistake injection, paraphrasing, filler tokens (2307.13702).
- **Counterfactual / cue injection** — Turpin 2305.04388 and Chen/Benton/Perez 2505.05410.
- **Hybrid action+CoT monitor** — Arnav et al. 2505.23575 with FPR calibration.
- **Divergence metrics** — CoT ↔ action and CoT ↔ activation mismatch.
- **Inspect AI scorers** and **LangGraph / OpenAI Agents / Claude Agent SDK** middleware.
- **Multi-model leaderboard** across Claude, GPT, Gemini, DeepSeek-R1, Qwen3, GPT-OSS.

## Install

```bash
pip install cot-divergence                 # core
pip install "cot-divergence[activations]"  # + nnsight / TransformerLens
pip install "cot-divergence[langgraph]"    # + LangGraph middleware
```

## Quickstart

```python
from cotdiv import score_trajectory
from cotdiv.adapters import from_anthropic

traj = from_anthropic(messages)
result = score_trajectory(
    traj,
    metrics=["legibility", "coverage", "action_consistency"],
    autorater="google/gemini-2.5-pro",
)
print(result.legibility.value, "±", result.legibility.stderr)
```

## Citation

See `CITATION.cff` (pending v0.1 DOI).

## License

MIT.
