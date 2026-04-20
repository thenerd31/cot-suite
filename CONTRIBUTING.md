# Contributing

## Dev setup

```bash
git clone https://github.com/cot-divergence/cot-divergence
cd cot-divergence
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Adding a new metric

1. Implement under `src/cotdiv/metrics/<name>.py`; subclass the interface in `core/registry.py`.
2. Add a synthetic-degradation unit test in `tests/test_<name>_degradation.py`. The test MUST assert the metric drops monotonically under at least three standard perturbations: (a) sentence redaction, (b) language translation, (c) filler-token replacement. Metrics without this invariant are rejected.
3. Document the reference paper, formula, and invariants in `docs/metrics/<name>.md`.
4. If it produces a scalar compatible with Inspect AI, add a scorer under `src/cotdiv/inspect/scorers/`.

## Adding a new test (Lanham-style causal-intervention)

1. Implement under `src/cotdiv/tests/<name>.py` with the uniform signature described in `docs/tests/README.md`.
2. Include the reference paper arXiv ID in the docstring.
3. Add a minimal-reproduction integration test marked `@pytest.mark.slow`.

## Prompt versioning

Autorater prompts live under `src/cotdiv/autoraters/prompts/` and are versioned via filename (e.g., `emmons_zimmermann_v1.txt`). Prompt hashes are computed at load time and stored in `Score.metadata` for reproducibility. Never mutate a shipped prompt — bump the version instead.

## Agent artefacts for Inspect Evals PRs

When contributing an upstream scorer to `inspect_evals`, include:
- `agent_artefacts/<eval>/review/SUMMARY.md`
- `agent_artefacts/<eval>_<version>/validity/VALIDITY_REPORT.md`
- `agent_artefacts/trajectory_analysis/<eval>/<eval>_<model>_ANALYSIS.md`
