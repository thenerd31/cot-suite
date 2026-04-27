# Reproducibility

cot-suite v0.1 commits to a small, explicit reproducibility contract: every Score row carries enough metadata to back-trace the result to the cotsuite release, the evaluation methodology version, the prompt content, and the autorater model that produced it.

## v0.1 contract

Every `Score.metadata` produced by an Inspect AI scorer in cot-suite carries:

| Field | Type | What it pins |
|---|---|---|
| `eval_version` | str (semver) | The methodology version. Bumped on changes that break numeric comparability of results across runs (binarization threshold, aggregation rule, default judge contract). Currently `"1.0.0"` for both shipped scorers. Independent of the package version. |
| `cotsuite_version` | str | The cotsuite package version (`hatch-vcs` derives this from the latest git tag). Lets any downstream consumer back-trace results to the cotsuite release that produced them. |
| `prompt_version` | str | The Appendix C / PHR prompt version (e.g. `emmons_zimmermann_v1`, `post_hoc_rationalization_v1`). Versioned, never mutated — to update, ship a new version and bump. |
| `prompt_sha256` | str | SHA-256 of the prompt template content. Drift detection: a SHA mismatch between the loaded prompt and the SHA cross-referenced in `docs/paper-refs/<arxiv_id>-<section>.md` is a hard failure. |
| `autorater` (legibility/coverage) <br> `judge_model` (PHR) | str | Provider-prefixed model spec — e.g. `anthropic/claude-haiku-4-5`, `google/gemini-2.5-pro`. Different autoraters give different absolute numbers; pin this for cross-paper comparison. |

Together these five fields let any consumer reproduce the eval bit-for-bit given the same input trajectories and the same grader-model seeds.

## Why these five (not more)

The launch contract intentionally avoids piling on every conceivable provenance field. Things we could pin but don't:

- **Wall-clock timestamp.** Adds noise to snapshot tests; reproducible runs should produce identical metadata.
- **Random seed state.** Inspect AI handles the grader-model RNG; forwarding it through the scorer would just shadow Inspect's own seed plumbing.
- **Inspect AI version.** Already pinned via the dependency floor in `pyproject.toml` — recording it in every Score row is redundant.
- **Model snapshot date.** Where a provider exposes one (Anthropic's date-suffixed model names like `claude-haiku-4-5-20251001`) it's already inside the `autorater` / `judge_model` string. Where it's not (OpenAI's rolling `gpt-5`), recording a snapshot date in metadata is a polite fiction — the API will silently route to whatever's current.

If a v0.1 user needs more metadata, the scorer's `**kwargs` plumbing accepts arbitrary extras that flow through to `Score.metadata`. The five-field contract is the *minimum*, not the maximum.

## Prompt-SHA drift detection

Both shipped Inspect scorers load their prompts via a versioned loader (`LegibilityCoveragePrompt.load(version)` and `PostHocRationalizationPrompt.load(version)`). The loader:

1. Reads the prompt template from `cotsuite/autoraters/prompts/<version>.txt` (or equivalent for PHR).
2. Computes its SHA-256 at load time.
3. Cross-checks it against `cotsuite/autoraters/prompts/<version>.sha256`.
4. Raises if they disagree.

`tests/test_appendix_c_prompt_integrity.py` runs this drift check in CI on every push, so a silent edit to a SHA-pinned prompt fails the build immediately.

## EVAL_VERSION pinning

`tests/test_inspect_integration.py::test_scorer_eval_version_pinned_snapshot` enforces a hardcoded snapshot of the v0.1 launch values:

```python
expected = {
    "legibility_coverage": "1.0.0",
    "post_hoc_rationalization": "1.0.0",
}
```

To bump either scorer's `EVAL_VERSION`, update this snapshot in the same commit. The intent is to force a deliberate decision — bumping `EVAL_VERSION` breaks numeric comparability with prior runs and should be conscious, not accidental. (We considered syrupy snapshots but rejected them: a hardcoded dict is simpler to audit in code review than a binary `__snapshots__/` file, and avoids the dependency.)

## Reproducing the v0.1 multi-family scaling table

Every row in [the 8-model scaling table](scaling.md) was produced by the same code on `main`, with deterministic inputs:

- **Dataset:** GPQA-Diamond, 198 questions (HuggingFace `Idavidrein/gpqa`, split=`diamond`)
- **Sampling:** Qwen3 thinking-mode default — temperature 0.6, top_p 0.95, top_k 20, min_p 0, max_new_tokens 32768
- **Per-question cap:** 5-minute Modal RPC timeout
- **Autorater:** Claude Haiku 4.5 (`anthropic/claude-haiku-4-5`) with Appendix C prompt SHA `ac1e0ac4044b0a64…`
- **PHR detector:** Claude Haiku 4.5 with PHR prompt SHA `4d7cc712e9456b80…`
- **Detector ablation:** Stage 3.5 ran the same 122 correct Qwen3-14B trajectories through Claude judge / Arcuschin regex / exact-match — max spread 1.64pp

Per-row trajectory JSONL and a `RUN.md` per model live in `benchmarks/results/<model_id>_full/` in the repo. The aggregated summary at [`benchmarks/results/multi_family_summary.md`](https://github.com/thenerd31/cot-suite/blob/main/benchmarks/results/multi_family_summary.md) reports the same numbers as the [scaling page](scaling.md).
