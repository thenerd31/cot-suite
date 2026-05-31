# cot-suite

cot-suite is an Inspect AI-native library for chain-of-thought monitorability evaluation on reasoning-model agents.

![Cohen's κ collapses to ~0.2 on saturated legibility while raw agreement and Gwet's AC2 stay ~0.97; on coverage κ tracks AC2.](results/kappa_degeneracy.png)

On frontier-model legibility, the usual agreement statistic (Cohen's κ) collapses to about 0.2 and reads as "the judges barely agree," yet raw agreement (`p_o` ≈ 0.97) and the prevalence-robust Gwet AC2 (≈ 0.96) show they agree on ~98% of items, so that low κ is a base-rate artifact rather than disagreement. cot-suite also carries two against-release reproductions of prior monitorability results, Turpin and Arcuschin, documented below.

> Pre-launch, targeting late-July to mid-August 2026. Renamed from `cot-divergence` on 2026-04-21; the `cotdiv` import path keeps working through a shim until 2026-07-21.

## What this reproduces

Two against-release reproductions, each replaying a paper's published metric on its own released data. Original vs. ours vs. Δ.

**Turpin et al.** ([2305.04388](https://arxiv.org/abs/2305.04388)). Biased-context accuracy drop (percentage points), the four Table-1 cells; reproduction within 0.08pp (tolerance 0.5pp).

| model | shot | paper | ours | Δ |
|---|---|--:|--:|--:|
| text-davinci-003 | zero-shot | -36.3 | -36.38 | -0.08 |
| claude-v1 | zero-shot | -30.6 | -30.65 | -0.05 |
| text-davinci-003 | few-shot | -24.1 | -24.11 | -0.01 |
| claude-v1 | few-shot | -21.5 | -21.57 | -0.07 |

**Arcuschin et al.** ([2503.08679](https://arxiv.org/abs/2503.08679), ChainScope). Implicit Post-Hoc Rationalization (IPHR) pair rate, recomputed from the released ChainScope dataframe (n = 4,892 pairs). Integer-exact metric replay: ours matches the release count for every model below (Δ = 0). Partial, covering 4 of the 7 headline cells.

| model | paper | ours | count | Δ (count) |
|---|--:|--:|--:|--:|
| gpt-4o-mini | 13% | 13.49% | 660 | 0 |
| claude-3.5-haiku | 7% | 7.42% | 363 | 0 |
| gemini-2.5-flash | 2.17% | 2.17% | 106 | 0 |
| claude-3.7-sonnet (1k) | 0.04% | 0.04% | 2 | 0 |

The paper reports rounded integers for the first two rows; ours are the unrounded rates that round to them. The other 3 headline cells (chatgpt-4o-latest, deepseek-r1, gemini-2.5-pro) and 4 non-headline cells are blocked: their per-model pair sets were adaptively oversampled in a way the release does not let us reconstruct, so no honest cell-for-cell Δ exists.

## Why κ misleads

The κ collapse in the figure is the Cohen's-κ prevalence paradox (Feinstein & Cicchetti, 1990): when one rating category dominates, κ's chance-correction term inflates and the coefficient falls toward zero even at near-perfect observed agreement. Gwet's AC2 (Gwet, 2008) is a standard prevalence-robust alternative. The statistic is textbook; what cot-suite adds is showing that this artifact silently breaks cross-judge monitorability metrics on exactly the saturated, near-ceiling regime those metrics target. So every faithfulness scorer reports κ, Gwet AC2, raw agreement, and a saturation flag together, across at least two classifiers, instead of one κ that can read as disagreement where there is none. Full write-up: [`docs/cross_judge_degeneracy.md`](docs/cross_judge_degeneracy.md).

## Install

```bash
pip install cot-suite                 # core
pip install "cot-suite[nlp]"          # + NLTK punkt (Lanham-style sentence splitting)
pip install "cot-suite[langgraph]"    # + LangGraph middleware
pip install "cot-suite[activations]"  # + nnsight / TransformerLens (open-weights only)
pip install cot-divergence            # legacy alias, resolves to cot-suite until 2026-07-21
```

## Reproduce

Every command runs offline on committed or vendored data, with no model calls ($0).

```bash
uv run --with matplotlib python results/figures.py                      # Figure 1
PYTHONPATH=. .venv/bin/python scripts/degeneracy_reanalysis_ez.py       # κ-degeneracy re-analysis
PYTHONPATH=. .venv/bin/python scripts/validate_b2_turpin_stage_a.py     # Turpin reproduction
PYTHONPATH=. .venv/bin/python scripts/validate_b4_iphr_reproduction.py  # Arcuschin IPHR reproduction
```

## Modules, citation, license

Faithfulness methods are exposed through their correct Inspect-AI abstractions (scorers for the judge methods, solvers and tasks for the interventions), each citing its source paper: the Emmons & Zimmermann legibility/coverage autorater, the Lanham four-test suite, the Turpin counterfactual battery, Chen cue-injection, and the Arcuschin pair-level IPHR metric alongside a narrower per-trajectory PHR detector. Scorers are auto-discoverable through Inspect's entry points after `pip install cot-suite`. cot-suite also interoperates with OpenAI's monitorability-evals (g-mean² and the three eval archetypes) and loads the MonitorBench task surface (ASTRAL-Group [2603.28590](https://arxiv.org/abs/2603.28590), 1,514 instances) natively.

Methodology and shortcut disclosures: [`AUDIT.md`](AUDIT.md). Roadmap and Inspect-wrapper status: [`ROADMAP.md`](ROADMAP.md). Pre-release blockers: [`BLOCKERS.md`](BLOCKERS.md). Related-work landscape (Young's trilogy, MonitorBench, OpenAI's monitorability-evals): [`docs/related_work.md`](docs/related_work.md). Usage: [`docs/quickstart.md`](docs/quickstart.md).

Cite via [`CITATION.cff`](CITATION.cff) (DOI pending v1.0); until then, cite the repo URL. License: MIT.
