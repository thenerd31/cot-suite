# Inspect AI integration assessment for cot-suite v0.1

**Status:** advisory, for the v0.1 architecture decision before
README writing.
**Date:** 2026-04-26.
**Confidence:** medium-high. Inspect AI internals verified by reading
the installed `inspect_ai==0.3.210` source under `.venv/`. The
inspect_evals contribution bar and the g-mean² formula are verified
against the live upstream repos via WebFetch on 2026-04-26.

This document does **not** advocate; it lays out what we'd be
signing up for under each architectural option.

---

## 1. Per-metric refactoring estimate

Inspect AI's `Scorer` is a Python `Protocol` with one async method:

```python
@runtime_checkable
class Scorer(Protocol):
    async def __call__(self, state: TaskState, target: Target) -> Score | None: ...
```
(`.venv/lib/python3.12/site-packages/inspect_ai/scorer/_scorer.py:34`)

`Score.value` may be a scalar **or** a `Mapping[str, scalar]` — so
one scorer can emit multiple sub-metrics from one autorater call.
Our existing `cot_legibility_coverage` already exploits this
(`src/cotsuite/inspect/scorers/legibility_coverage.py:71-86`).

Estimates assume "convert the existing async function into the
Scorer protocol shape, plus add `@scorer(metrics=...)` registration,
plus wire model calls through `inspect_ai.model.get_model(role="grader")`
instead of our `GraderClient`". Tests + a basic Inspect smoke task
included.

> **Partially superseded by the Stage-0 scorer recon (2026-05-30).** That recon
> reclassified **Lanham as an Inspect task/solver, NOT a `Scorer`**: its tests are
> mid-trajectory interventions + per-item AOC computed via N re-elicitations of the
> *model-under-test*, which fit a task/solver better than `score(state, target)`
> (a scorer *can* call models N times, but that re-runs the model under test, not a
> grader — an idiom mismatch). It also set the v0.2 build order to
> **Chen → Turpin → Lanham** (easiest → hardest) and confirmed Turpin's
> `accuracy_drop` stays a **dataset-level** metric, not a `Score`. The fit/hours
> below are the earlier (more optimistic) estimate; treat the README / ROADMAP
> framing as authoritative.

| Metric | Current shape | Inspect fit | Hours |
|---|---|---|---|
| **Lanham early-answering** (`src/cotsuite/tests/lanham/early_answering.py`) | per-trajectory async fn; takes `cot` string + `question` + `answer_extractor`; emits `TestResult` with `aoc` + `per_fraction` | Clean. Per-sample. The N internal `client.complete` calls become N `grader.generate` calls. AOC reduces to a single `Score.value` with `aoc` and `per_fraction` in metadata. | **6-10 h** |
| **Lanham mistake-injection** (`tests/lanham/mistake_injection.py`) | same shape, requires a *separate* `mistake_generator` model | Clean, with one wrinkle: Inspect's standard pattern is one grader role; mistake-generation is a *second* model role. Inspect supports custom roles via `get_model(role="mistake_generator")`; needs documentation in the scorer's docstring. | **8-12 h** |
| **Turpin counterfactual-bias** (`tests/turpin_counterfactual.py`) | takes `samples: list[Sample]` and runs baseline + biased prompts internally → emits one aggregate `TestResult` | **Awkward fit.** Inspect's mental model is "one sample in, one Score out". Turpin's accuracy-drop is a *cross-sample* aggregate. Cleanest refactor: a **Solver-pair** that runs baseline + biased generations on the same `TaskState`, stashing both completions in `state.metadata`; the scorer computes `(bias_followed, verbalized)` per-sample, and Inspect's metric layer aggregates `accuracy_drop = mean(...)` across the eval. | **12-18 h** |
| **Chen cue-injection** (`tests/chen_cue_injection.py`) | same shape as Turpin: aggregates over `samples` | Same refactor as Turpin. The 5 cues become 5 scorer instances or one parametrized scorer. Verbalization-judge call already matches Inspect's autorater pattern. | **10-14 h** |
| **Arcuschin PHR detector** (`tests/post_hoc_rationalization.py`) | per-trajectory async fn; single judge call; SHA-pinned prompt | Cleanest fit of all five — single sample in, single Score out, scalar boolean (`diverged AND NOT acknowledged`). Almost a 1:1 port. | **4-6 h** |
| **Emmons & Zimmermann legibility/coverage** (`autoraters/legibility_coverage.py`) | already wrapped at `src/cotsuite/inspect/scorers/legibility_coverage.py` | Already done. ~1 h to port the per-run mean+stderr aggregation onto Inspect's `mean()/stderr()` metric helpers (currently we hand-aggregate; Inspect's reducer infrastructure can do this if we emit one `Score` per run). | **1-2 h cleanup** |

**Subtotal for the five metrics + the existing autorater port:
41-62 h** (≈1.0–1.5 calendar weeks at full focus, assuming Aswin
already knows Inspect's idioms; double if learning the framework
cold).

### Arcuschin's cross-question pair construction — does it fit?

The PHR **detector we ship** (single-trajectory: does the CoT's
conclusion match the emitted answer?) fits Inspect's per-sample
model fine.

Arcuschin et al.'s **full IPHR methodology** in 2503.08679 §3 is
different: it constructs *pairs of opposite questions* (e.g., "is X
greater than Y?" / "is Y greater than X?") and flags contradictions
across the pair. **That pair construction does not map onto
Inspect's per-sample scorer.** The clean Inspect pattern would be:

- **Dataset-side**: emit each pair as a *single sample* whose
  `input` carries both questions and whose `target` is the
  consistency invariant. The solver runs the model twice (once per
  sub-question) and stores both outputs in `state.messages` /
  `state.metadata`. The scorer reads both and emits a contradiction
  score.
- Or: dataset-side, emit two samples linked by a shared
  `metadata["pair_id"]` and post-process the eval log to compute
  pairwise contradictions. This works but is awkward — you lose
  Inspect's built-in per-sample scoring and have to write a
  `@metric` that walks the log.

Our v0.1 ships only the **per-trajectory** PHR detector — a strict
subset of Arcuschin's methodology — so we sidestep the pair-
construction problem entirely. The README and docstring should be
**explicit** that what we ship is "the per-trajectory diverged+
unacknowledged signal", not "Arcuschin IPHR replication". This is
already noted in `tests/post_hoc_rationalization.py:52-63`.

If we ever want to ship the full IPHR pairing under Inspect, plan
**+15-25 h** for the dataset-side pair plumbing and a custom
`@metric`.

---

## 2. Dual-shipping feasibility

**Yes, dual-shipping (standalone CLI + Inspect-registered scorers)
is the default supported pattern.** Inspect AI loads scorers,
solvers, and tasks from external packages via two mechanisms:

1. **Direct import** — any installed package can
   `from cotsuite.inspect.scorers import cot_legibility_coverage`
   and pass the result to a `Task(scorer=...)`. The `@scorer(...)`
   decorator registers into Inspect's process-global registry on
   first import (`.venv/.../inspect_ai/_util/registry.py:209`). Our
   existing wrapper at
   `src/cotsuite/inspect/scorers/legibility_coverage.py` already
   does this lazily — `from inspect_ai.scorer import scorer` is
   inside the function body, so importing cot-suite without
   inspect-ai installed doesn't blow up.

2. **Entry-point auto-discovery** — Inspect calls
   `entry_points(group="inspect_ai")` in `_util/entrypoints.py` and
   imports any module advertising itself there. That module's
   `@task` / `@scorer` / `@solver` decorators run at import time,
   populating the registry. Once registered, the user can do
   `inspect eval cotsuite/lanham_early_answering` (or whatever name
   we assign) **with no special CLI flags**.

   Concretely, we add to `pyproject.toml`:

   ```toml
   [project.entry-points.inspect_ai]
   cotsuite = "cotsuite.inspect"
   ```

   and `src/cotsuite/inspect/__init__.py` imports each scorer/task
   module so the decorators fire.

**What this means for the architecture:**

- The standalone `cotsuite` CLI (`src/cotsuite/cli.py`) keeps
  working unchanged. It uses `cotsuite.core.registry` (our own
  metric registry) and Typer.
- The Inspect-facing API lives entirely under
  `src/cotsuite/inspect/{scorers,solvers,tasks}/` and depends only
  on our `cotsuite.tests.*` and `cotsuite.autoraters.*` modules. No
  code duplication: each Inspect scorer is a thin wrapper over the
  underlying async function.
- The Inspect dependency is already required (`pyproject.toml:31` —
  `inspect-ai>=0.3.199`). If we want `inspect-ai` to be optional,
  move it to `[project.optional-dependencies] inspect = [...]` and
  gate imports inside `cotsuite.inspect.*` modules.
  **Recommendation: keep it required for v0.1** to avoid the
  import-error rabbit hole.

**You do not have to upstream into `inspect_evals` to be Inspect-
compatible.** `inspect_evals` is a *curated benchmark suite*, not
the framework. Our scorers work via `inspect eval` regardless of
whether they ever land in that repo. Upstreaming buys
discoverability and a maintainer-blessed quality stamp; it isn't a
technical prerequisite.

**Architectural recommendation:** keep the standalone CLI as the
primary research surface, add the Inspect entry-point shim, and
treat `inspect_evals` upstreaming as a **post-launch**
discoverability play.

---

## 3. Autorater compatibility & SHA-pinning

Inspect's scorer abstraction handles the Emmons/Zimmermann prompt
cleanly. Our existing wrapper
(`src/cotsuite/inspect/scorers/legibility_coverage.py`) shows the
pattern: load the prompt via `LegibilityCoveragePrompt.load()`,
render it inside the `score(state, target)` body, emit
`Score(metadata={"prompt_sha256": prompt.sha256, ...})`. The SHA
flows into the eval log unchanged because Inspect serializes
`Score.metadata` verbatim into the `.eval` log file.

**Where the SHA-integrity test lives:** unchanged —
`tests/test_appendix_c_prompt_integrity.py` sits in the cot-suite
repo, not in Inspect's world. The test computes
`hashlib.sha256(prompt_file.read_bytes())` and compares to the
canonical `.sha256` sidecar. It runs under cot-suite's pytest,
independent of any Inspect-side test fixture. This is the right
place — the prompt file is shipped by cot-suite
(`src/cotsuite/autoraters/prompts/emmons_zimmermann_v1.txt`), so
cot-suite owns its integrity guard. If we ever upstream into
`inspect_evals`, we'd duplicate the test there too (cheap; the SHA
check is ~10 lines).

**No custom adapter is needed.** The only friction is that our
prompt uses literal `.replace()` substitution (because the Appendix
C JSON schema contains `{...}` braces that conflict with
`str.format()`), and our existing wrapper already handles that — it
never calls `format` on the template.

**One real risk to flag:** Inspect's `get_model(role="grader")`
returns whatever model the user specified via `-M grader=...` on
the CLI. If the user forgets the role flag, Inspect falls back to
the eval's primary model — which means the *model under test*
would also be its own grader. We should detect this in the scorer
and emit a warning (or refuse to run). Add **+1 h** for this
guard.

---

## 4. inspect_evals contribution path

`inspect_evals` is the UK AI Security Institute's curated catalogue
of evals (github.com/UKGovernmentBEIS/inspect_evals). The
contribution bar is **explicit and strict**, verified via WebFetch
on the live `CONTRIBUTING.md` 2026-04-26.

### Required tests (verbatim from upstream CONTRIBUTING.md)

> **All new evaluations must include**:
> 1. Unit tests covering all non-trivial custom functions (solvers,
>    scorers, dataset functions, custom tools, utilities) with edge
>    cases and error conditions.
> 2. End-to-end tests demonstrating the full evaluation pipeline
>    using `mockllm/model`. If your evaluation has multiple
>    meaningfully different tasks/variants, include one success and
>    one error-handling test per variant.
> 3. Dataset validation via `assert_huggingface_dataset_structure`
>    if using HuggingFace datasets.
>
> Tests must be marked appropriately: `@pytest.mark.dataset_download`,
> `@pytest.mark.huggingface`, `@pytest.mark.docker`, and
> `@pytest.mark.slow(<seconds>)` for tests exceeding ~10 seconds.
>
> *"If your evaluation is not adequately tested, it will not be
> accepted."*

### Required documentation (verbatim)

> - **README.md** with an "Evaluation Report" section including:
>   - Implementation details and deviations from reference
>     implementations
>   - Results compared against original papers or reputable
>     leaderboards
>   - Total samples run and timestamp
>   - Specific `inspect eval` commands used
>   - Model and evaluation versions
> - **eval.yaml** with required metadata fields: `title`,
>   `description`, `arxiv`, `group`, `contributors`, `tasks`,
>   `external_assets`
> - **Changelog fragment** via `uv run scriv create` for
>   user-facing changes

### Code quality (verbatim)

> - Type hints on all functions
> - Absolute imports (never relative)
> - Google-style docstrings
> - Comments explaining complex logic
> - No substring matching for scoring
> - Use `inspect_ai` built-in components before custom
>   implementations
> - Suppress individual Ruff rules sparingly with comments

### Review timeline

> *"Expect PR review within a couple of days."*
>
> Reviewers use [Conventional Comments](https://conventionalcomments.org/).
> Ensure "Allow edits from maintainers" is enabled.

**This is faster than I'd assumed.** A ready-to-land PR gets review
in ~2 days; merge usually within a week if no major rework needed.

### Strategic criteria for new evals

> New evaluations must also meet strategic criteria: **be
> well-established in research, challenging and non-saturated,
> clearly scoped, verifiable, and from credible sources.**

Lanham, Turpin, Chen, Arcuschin, Emmons & Zimmermann all clearly
meet these — multiple papers, frontier-lab authorship, explicit
methodology in published work. We're not asking maintainers to
bless novel methodology, just bless our implementations of
established ones.

### Recent merge activity (verified 2026-04-26)

Several recently-merged PRs from `claude-code-aisi` (an automated
maintenance bot) landing fixes; one substantive PR from
human contributor `Scott-Simmons`:

- **PR #1538** (Scott-Simmons, merged 2026-04-24): "docs: announce
  /src deprecation (8 May 2026) in favour of registry"

That PR is **load-bearing** for our timeline: the `/src/`
directory layout is being deprecated 2026-05-08 in favor of a
registry-based eval loading approach. Any inspect_evals PR opened
after that date should target the new registry layout, not the old
`/src/` paths. We should confirm the new layout before opening any
PR.

### Realistic timeline estimate (for one PR landing one metric)

- Implementation + tests + README + reproduction run: **20-30 h** of
  our work.
- Maintainer review cycle: ~2-day initial review, **typically 1-2
  weeks** wall-clock to merge if reviews come back clean. (Faster
  than the agent's pre-research estimate of "2-8 weeks".)

**Recommendation: ship cot-suite v0.1 first, then upstream *one*
metric (suggest legibility/coverage — already implemented, the
cleanest implementation to demonstrate for upstreaming) as a proof of
process, then batch the remaining four into 2-3 PRs.** Don't block v0.1 launch on any of
these PRs.

---

## 5. g-mean² adoption

OpenAI's `monitorability-evals` repo
(github.com/openai/monitorability-evals) ships reference code for
the g-mean² metric from arXiv 2512.18311. Verified via WebFetch
2026-04-26.

**The actual formula** — extracted from `run_eval_scaffold_lib.py`:

```python
gmean2 = tpr * tnr if not (math.isnan(tpr) or math.isnan(tnr)) else float("nan")
```

Where `tpr = tp/(tp+fn)` and `tnr = tn/(tn+fp)` derived from a
DataFrame with columns:
- `x` — the **arm** (which intervention condition the sample is
  under, including a baseline arm)
- `y` — **correctness** of the sample's outcome
- `z` — **monitor signal** (does the monitor flag the sample?)

So g-mean² is the product of TPR and TNR for a binary monitor
classifier evaluated against a binary ground truth, optionally
across intervention arms. The repo wraps this in
`bootstrapped_gmean_metric` for uncertainty quantification.

### What does this mean for cot-suite v0.1?

- Our autorater outputs (legibility, coverage) are 0-4 Likert,
  **not** binary monitor signals. Computing g-mean² requires
  thresholding them into a binary monitor decision (e.g., "monitor
  flags trajectory iff legibility ≥ 3 AND coverage ≥ 3"). The
  threshold choice is itself a methodological decision and should
  be documented.
- Our PHR detector outputs are binary (`diverged AND NOT
  acknowledged`) — these can serve as the monitor signal directly.
- Computing g-mean² **requires labeled ground truth** for what the
  monitor *should* flag. For PHR-as-monitor: ground truth would be
  "is this trajectory actually post-hoc rationalizing?", which is
  what we're trying to detect — circular without human labels.

**Bottom line: g-mean² is NOT computable from our existing v0.1
outputs alone** without either (a) human-labeled ground truth, or
(b) reframing as "monitor agreement with a different signal" (e.g.,
"do legibility-coverage and PHR detector agree?"). Both are real
methodological commitments, not refactors.

**Recommendation: defer g-mean² to v0.1.1 or v0.2.** The honest
path: read OpenAI's reference more carefully, decide on a
binarization scheme + ground-truth proxy, document it explicitly,
add it as a v0.1.1 metric. Aim **2-4 h once the choice is locked**;
the pre-work to lock the choice is **4-8 h of careful reading +
methodology design**, not engineering.

---

## 6. Three scopes — total estimates

These assume Aswin is the sole implementer, post-rename codebase as
of 2026-04-26, with the v0.1 metric set already implemented (only
the Inspect adaptation is new work).

### Scope A — Minimum: standalone CLI now, inspect_evals PR after launch

- v0.1 ships the Typer CLI (`cotsuite score`, `cotsuite eval`) plus
  a single Inspect scorer for legibility/coverage (already done).
- Add `[project.entry-points.inspect_ai]` so
  `inspect eval cotsuite/legibility_coverage` works.
- No additional Inspect-callable Solvers; no inspect_evals PR.
- Documentation says "Inspect AI compatibility skeleton shipped —
  one scorer; more in v0.2".

**Effort:** ~**4-8 h** of polish on top of what's already in tree
(mostly README + a single end-to-end Inspect smoke test + the
entry-point line).
**Calendar:** under **2 days**.
**Risk:** lowest.

### Scope B — Medium: standalone CLI + Inspect compatibility layer at v0.1

- All 5 metrics ported into Inspect Scorers under
  `cotsuite.inspect.scorers`.
- Turpin and Chen ship as Solver+Scorer pairs.
- Entry-point registration so `inspect eval cotsuite/<metric>`
  works after `pip install cot-suite`.
- 5 minimal Inspect Tasks (one per metric, on a small reference
  dataset) shipped under `cotsuite.inspect.tasks` for examples.
- inspect_evals PRs deferred to v0.2.

**Effort:**
- Per-metric ports (§1): **41-62 h**.
- Solver-pair scaffolding for Turpin/Chen: ~6-10 h (within the §1
  ranges).
- Entry-point + tasks + smoke tests: ~10-14 h.
- Documentation pass: ~6-8 h.
- Buffer (15%): ~9-13 h.

**Total: ~70-100 h, or 2-3 calendar weeks at ~30 h/week focus.**
**Risk:** medium. The Turpin/Chen Solver-pair pattern is the
riskiest piece — if Inspect's `state.metadata` round-tripping
through the eval log has sharp edges we haven't hit yet, that adds
5-15 h.

### Scope C — Full: native Inspect-only, no standalone CLI

- Drop Typer; `cotsuite.cli` becomes a thin shim that delegates to
  `inspect eval`.
- Drop `cotsuite.core.registry` (or keep it internal-only).
- Everything user-facing is an Inspect Task or Scorer.

**Effort:**
- Everything in Scope B: ~70-100 h.
- Removing Typer and migrating CLI semantics into Inspect-native
  invocation patterns: **~15-25 h** (mostly: figuring out how to
  express "score a pre-existing trajectory JSONL" as an Inspect
  eval — Inspect assumes generations happen during the eval, not
  offline).
- Re-doing all examples in `examples/` against Inspect: ~10-15 h.
- Documentation rewrite: ~10-15 h (the entire README's mental
  model changes).

**Total: ~105-155 h, or 4-6 calendar weeks.**
**Risk:** highest. The "score a pre-existing trajectory" use case
(Stage 3 multi-family results live in `benchmarks/results/*_full/`
as JSONL) doesn't fit Inspect's Solver→Scorer flow naturally —
Inspect wants to *generate during the eval*. We'd have to either
(a) write a `precomputed_solver` that replays stored generations
into a `TaskState`, or (b) tell users "re-run your generation
through Inspect to score it", which loses the existing benchmark
data. (a) is doable but is its own ~10-15 h side quest.

---

## Recommendation (advisory)

**Ship Scope A for v0.1; plan Scope B for v0.2.** Concretely:

1. **v0.1 (next ~2 weeks):** standalone CLI + the one already-shipped
   Inspect scorer. README documents the Inspect path as "one scorer
   today, more coming". Add the `[project.entry-points.inspect_ai]`
   hook so `inspect eval cotsuite/legibility_coverage` works.
2. **v0.1.1 (1-2 weeks after v0.1):** g-mean² as a v0.1.1 metric,
   after reading monitorability-evals' reference code more carefully
   and locking in a binarization scheme + ground-truth proxy.
3. **v0.2 (4-6 weeks after v0.1):** port the remaining 4 metrics to
   Inspect Scorers (Scope B work). Confirm the post-2026-05-08
   inspect_evals registry layout before doing any upstream work.
4. **v0.2.1 / v0.3:** open one inspect_evals PR (legibility/coverage,
   the tightest story), use the review feedback to shape the
   remaining four.

**Why not Scope B at v0.1.** The Turpin/Chen aggregate-vs-per-sample
mismatch is real architectural work, not a wrapper. Doing it under
launch pressure risks shipping a half-correct Solver-pair pattern
that's painful to revise. Doing it after v0.1 launches gives us
actual user feedback on which scorers matter. (**Superseded by the
2026-05-30 Stage-0 recon:** the build order is **Chen → Turpin → Lanham**
— the cue-injection scorers map cleanly as Solver+Scorer and ship first;
Lanham is the *hardest*, ports as a task/solver not a scorer, and is gated
on an abstraction decision. An earlier draft of this note suggested
Lanham-first; the recon reversed that.)

**Why not Scope C ever (probably).** The standalone CLI is doing
real work that Inspect's framework doesn't naturally express:
scoring pre-existing trajectory JSONL files (Aswin's Stage 3
multi-family dataset). Replacing that with "re-run the generation
under Inspect" throws away expensive already-recorded model
outputs. The two interfaces serve different research loops.
Dual-shipping (Scope B) keeps both alive at proportionate cost.

**Highest-leverage single step:** add the
`[project.entry-points.inspect_ai]` hook to `pyproject.toml` *now*,
even at Scope A. It's <1 h of work and unblocks the discoverability
story without committing to porting metrics. If someone runs
`inspect eval cotsuite/legibility_coverage` post-launch and it
works, that's a strong "this library is Inspect-native" signal at
near-zero cost.

---

## Open questions for the v0.1 launch decision

1. **Does Aswin want `inspect eval cotsuite/<metric>` to work at
   v0.1?** If yes → Scope A + entry-point. If no → just Scope A
   minimum.
2. **Is the inspect_evals PR a public-credibility milestone for the
   MATS/Anthropic Fellows app?** If yes, plan one PR for v0.2 and
   ensure the post-2026-05-08 registry layout is targeted.
3. **g-mean² binarization scheme**: legibility ≥ 3 AND coverage ≥
   3? Some other threshold? This is a methodology decision that
   should not be made under launch pressure — defer to v0.1.1.

---

## Appendix: file pointers

- Inspect AI Scorer protocol — `.venv/lib/python3.12/site-packages/inspect_ai/scorer/_scorer.py:34-61`
- Inspect AI Score model (with `Mapping[str, scalar]` value support) — `.venv/lib/python3.12/site-packages/inspect_ai/scorer/_metric.py:81-97`
- Inspect AI Solver protocol — `.venv/lib/python3.12/site-packages/inspect_ai/solver/_solver.py:78-109`
- Inspect AI entry-point loader (group=`"inspect_ai"`) — `.venv/lib/python3.12/site-packages/inspect_ai/_util/entrypoints.py:1-44`
- Inspect AI Task registry — `.venv/lib/python3.12/site-packages/inspect_ai/_eval/registry.py:97-170`
- Existing cot-suite Inspect scorer — `src/cotsuite/inspect/scorers/legibility_coverage.py`
- inspect_evals contribution requirements — https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/CONTRIBUTING.md (verified 2026-04-26)
- inspect_evals 2026-05-08 registry deprecation — PR #1538 by Scott-Simmons (merged 2026-04-24)
- monitorability-evals g-mean² formula — https://github.com/openai/monitorability-evals/blob/main/run_eval_scaffold_lib.py (verified 2026-04-26)
- cot-suite inspect_evals contribution-artefact convention — `CONTRIBUTING.md:33-37` (in cot-suite repo)
