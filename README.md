# cot-suite

cot-suite is an Inspect AI-native library for chain-of-thought monitorability evaluation on reasoning-model agents.

Cross-judge agreement (Cohen's κ) is increasingly used to decide whether an LLM-judged monitorability metric is trustworthy. On saturated metrics — the high-monitorability regime those metrics are built to measure — κ collapses to ~0.2 and looks like the judges disagree, even though they agree on ~98% of items (raw agreement 0.97, Gwet's AC2 0.96). The low κ is a known statistical artifact, the Feinstein-Cicchetti prevalence paradox, not real disagreement.

![Cohen's κ collapses to ~0.2 on saturated legibility while raw agreement and Gwet's AC2 stay ~0.97; on coverage κ tracks AC2.](results/kappa_degeneracy.png)

The statistic is textbook; the contribution is the applied demonstration that it silently breaks cross-judge monitorability metrics on exactly the saturated regime they target. For example, a recent classifier-sensitivity result (Young, 2603.20172) reads low cross-classifier κ as construct divergence between classifiers, without separating the saturation artifact from real disagreement.

> Pre-launch (v0.1 targeted late-July to mid-August 2026). Renamed from `cot-divergence` on 2026-04-21; `from cotdiv ...` still imports, with a deprecation warning, until 2026-07-21.

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

## What cot-suite reports

Every faithfulness scorer reports κ, Gwet AC2, raw agreement, and a per-judge saturation flag together, across at least two classifiers, rather than a single κ that can read as disagreement where there is none. The flag marks the saturated, near-ceiling case where κ is least trustworthy. Full write-up: [`docs/cross_judge_degeneracy.md`](docs/cross_judge_degeneracy.md).

## Install

Pre-launch: `pip install cot-suite` currently installs a 0.0.1 placeholder, not this library; the described feature set ships in the v0.1 PyPI release (see Roadmap). To run it today, install from source into a local `.venv` (the Reproduce commands below use it):

```bash
git clone https://github.com/thenerd31/cot-suite
cd cot-suite
python3 -m venv .venv
.venv/bin/pip install -e .         # core
.venv/bin/pip install matplotlib   # for the Figure 1 command (figure-only, not a runtime dep)
```

Optional extras, not needed to reproduce below: `.venv/bin/pip install -e ".[nlp]"` (NLTK punkt for Lanham-style sentence splitting), `".[langgraph]"` (LangGraph middleware), `".[activations]"` (nnsight / TransformerLens, open-weights only).

## Reproduce

Each command runs on committed or vendored data and makes no model or API calls ($0). Installing the packages above needs PyPI network; the analysis itself does not.

```bash
.venv/bin/python results/figures.py                        # Figure 1
.venv/bin/python scripts/degeneracy_reanalysis_ez.py       # κ-degeneracy re-analysis
.venv/bin/python scripts/validate_b2_turpin_stage_a.py     # Turpin reproduction
.venv/bin/python scripts/validate_b4_iphr_reproduction.py  # Arcuschin IPHR reproduction
```

## Modules

Five methods ship as Inspect-native scorers, solvers, or tasks, each citing its source paper and auto-registered through Inspect's entry points (`[project.entry-points.inspect_ai]`), so a command like `inspect eval ... --scorer cotsuite/cot_legibility_coverage` resolves once the package is installed: the Emmons & Zimmermann legibility/coverage autorater, the Turpin counterfactual battery, Chen cue-injection (5 of the paper's 6 cues; Visual Pattern ships as a simplified extension), Lanham's `early_answering` test, and a per-trajectory post-hoc-rationalization detector.

Other components are library code, not Inspect scorers: Lanham's three remaining tests (`mistake_injection`, `paraphrasing`, `filler_tokens`), and the pair-level IPHR metric (`cotsuite.tests.iphr`) that backs the Arcuschin reproduction above and runs from its validation script. Separately, cot-suite vendors OpenAI's g-mean² metric with cross-fit filtering (from `openai/monitorability-evals`) and wires it in as an Inspect `@metric`.

Methodology and shortcut disclosures: [`AUDIT.md`](AUDIT.md). Inspect-wrapper status: [`ROADMAP.md`](ROADMAP.md). Pre-release blockers: [`BLOCKERS.md`](BLOCKERS.md). Related-work landscape: [`docs/related_work.md`](docs/related_work.md). Usage: [`docs/quickstart.md`](docs/quickstart.md). Cite via [`CITATION.cff`](CITATION.cff) (DOI pending v1.0). License: MIT.

## Roadmap (v0.2)

- Inspect wrappers for Lanham's `mistake_injection`, `paraphrasing`, and `filler_tokens` (each needs a second model role).
- A MonitorBench loader (ASTRAL-Group [2603.28590](https://arxiv.org/abs/2603.28590), 1,514 instances) as an Inspect dataset.
- The three monitorability-evals archetypes (intervention / process / outcome-property) alongside the already-vendored g-mean² metric.
- The v0.1 PyPI release of the library described above.
