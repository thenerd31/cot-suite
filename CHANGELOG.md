# Changelog

All notable changes to cot-suite are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `cotsuite.tests.turpin_counterfactual.Sample.bias_target_letter` —
  optional per-question variable bias target (matches Turpin's
  `random_ans_idx`). Enables faithful reproduction of Turpin's
  `suggested_answer` bias mode where the target letter varies per
  question. When unset, falls back to `BiasConfig.default_target`.
- `cotsuite.tests.turpin_counterfactual.Sample.task` — optional task
  name; when set, `counterfactual_bias` computes accuracy_drop per-task
  then averages across tasks (matching Turpin's pandas-pivot
  `aggfunc='mean'`). When unset, samples are flat-pooled into one
  implicit `_default` group.
- `counterfactual_bias(... inconsistent_only: bool = True)` kwarg —
  restricts accuracy_drop denominator to the bias-points-to-wrong-answer
  subset (matches Turpin's
  `bias_consistent_labeled_examples == 'Inconsistent'`).
- `BiasConfig.target_injector` (variable-target injector) and
  `BiasConfig.default_target` (fallback bias-target letter).
- New `BIAS_CATALOG["suggested_answer"]` entry with variable-target
  support via `target_injector=_inject_suggested_answer_variable`.

### Deprecated

- `BIAS_CATALOG["sycophancy"]` is now a deprecated alias for
  `BIAS_CATALOG["suggested_answer"]`. Accessing the deprecated key emits
  a `DeprecationWarning`. Both keys return the same `BiasConfig` object;
  the alias is kept in the dict so `set(BIAS_CATALOG)` still includes
  `"sycophancy"` during the deprecation window.
  - **Sunset:** 2026-08-28 (90-day window, matching the existing
    `cotmon` / `cotdiv` import-shim pattern).
  - **Migration:** replace `bias="sycophancy"` with
    `bias="suggested_answer"` in calls to `counterfactual_bias()`.

### Changed

- `counterfactual_bias` `raw` output now includes `per_task_drops`,
  `inconsistent_only`, `n_eval_pool`, and per-sample `bias_target`.
  Default `inconsistent_only=True` is a behavior change relative to
  prior versions, which pooled over all samples. Numeric outputs of
  `counterfactual_bias` are not directly comparable across this
  boundary.

### Verified

- Turpin 2305.04388 4-cell suggested_answer accuracy drops reproduced
  exactly (delta 0.0pp at 1dp rounding) by running upstream
  `bbh_analysis.py` against vendored bbh_samples
  (`validation/turpin_artifacts/`). Upstream commit:
  `df099452736946533f59498a90c23be3f09631c4`.
