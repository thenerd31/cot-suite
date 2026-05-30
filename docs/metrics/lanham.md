# Lanham 4-test suite

**Originating paper:** Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning" — [arXiv 2307.13702](https://arxiv.org/abs/2307.13702).

**What it measures.** Causal-intervention faithfulness — does intervening on the CoT (truncating, mistake-injecting, paraphrasing, padding with filler) change the final answer in a way consistent with the CoT actually doing the work? If not, the model's reasoning is decorative.

**Module:** `cotsuite.tests.lanham`. Four sibling tests, each computing a length-weighted Area-Over-Curve (AOC) plus a per-fraction retention curve. Validation status: **B1 ✓** on Qwen3-14B / GPT-4o-mini.

## The four tests

| Test | Intervention | Paper section |
|---|---|---|
| `early_answering` | Truncate CoT at varying prefix lengths; resample final answer | §3.1 |
| `mistake_injection` | Inject a generated mistake at varying CoT positions; resample | §3.2 |
| `paraphrasing` | Paraphrase CoT prefix while preserving content; resample | §3.3 |
| `filler_tokens` | Pad CoT with filler ("..." × N) at varying lengths; resample | §3.4 |

A *faithful* CoT shows steep retention drop under early answering and mistake injection (early prefix carries information the final answer depends on), while paraphrasing preserves retention (content matters, not surface form), and filler tokens have zero uplift at any length (length itself doesn't help).

## API signature

```python
from cotsuite.tests.lanham import early_answering, mistake_injection, paraphrasing, filler_tokens

result = await early_answering(
    question="...",
    cot="...",
    full_answer="A",
    answer_extractor=...,        # callable: completion → answer letter
    sentence_splitter=...,       # callable: cot → list[str]; install [nlp] for Lanham-style NLTK punkt
    sampler=...,                 # async callable: prompt → completion
)
# result.aoc — length-weighted AOC scalar
# result.per_fraction — {0.0: retention, 0.25: ..., 0.5: ..., 0.75: ..., 1.0: 1.0}
# result.raw_curve — paper-equivalent points for cross-citation
```

`mistake_injection` additionally requires a `mistake_generator` (callable producing the injected mistake string) — provenance enforcement requires this be passed explicitly rather than defaulted, per fix `P1.2`.

## Example

```python
from cotsuite.tests.lanham import early_answering
from cotsuite.tests.lanham._extractors import letter_answer_extractor
from cotsuite.tests.lanham._sentences import nltk_sentence_splitter

aoc_result = await early_answering(
    question="What is the boiling point of water in Celsius?",
    cot="Water boils at sea level at the temperature where its vapor pressure equals atmospheric pressure...",
    full_answer="100",
    answer_extractor=letter_answer_extractor,
    sentence_splitter=nltk_sentence_splitter,
    sampler=my_anthropic_sampler,
)
print(f"AOC = {aoc_result.aoc:.3f}")
print(f"Retention curve: {aoc_result.per_fraction}")
```

## Known limitations

- **`filler_tokens` uses a sparse dyadic sweep** `(0, 1, 2, 4, 8, 16, 32, 64, 128, 256)` by default, not the dense paper sweep. The decisive claim ("zero uplift from filler at any length") survives sparse sampling — see [`ROADMAP.md`](https://github.com/thenerd31/cot-suite/blob/main/ROADMAP.md) → "Dense-sweep filler_tokens reproduction" for the v0.2+ plan.
- **`mistake_injection` requires an explicit `mistake_generator`** — invalidating mistakes silently break the test, so we made the dependency mandatory rather than default to a heuristic. Pass any async function that takes the question + CoT prefix and returns a plausible-looking mistake.
- **Length-weighted AOC alignment** vs the paper's exact integration scheme is in `BLOCKERS.md` — close enough for cluster-level claims, not yet bit-for-bit identical to the paper's Table 2.
- **Sentence splitting:** the default Python regex splitter handles common cases; install `cot-suite[nlp]` for the NLTK punkt splitter the paper uses (Lanham §3.4 footnote 7).

## Roadmap to Inspect AI (task/solver — NOT scorer)

The 4-test suite does **not** map to Inspect's `Scorer` abstraction (Stage-0 recon, 2026-05-30): each test is a mid-trajectory intervention + per-item AOC computed via N re-elicitations of the model-under-test, which does not fit `score(state, target)`. They are ported as Inspect **tasks/solvers** instead:

- **`early_answering`** — ✅ **shipped** as an Inspect task (the Lanham POC): `cot_lanham_early_answering` solver (truncate the CoT at each prefix fraction + re-elicit the model-under-test, reusing the native `@register_test` fn) + the thin `cot_lanham_early_answering_aoc` scorer. Single model.
- `mistake_injection` — ⏳ v0.2 (corrupt each sentence + re-elicit; **needs a 2nd model role** — the mistake generator).
- `paraphrasing` — ⏳ v0.2 (**needs a 2nd model role** — the paraphraser).
- `filler_tokens` — ⏳ v0.2 (filler-length sweep; single model).

See [Inspect AI integration](../inspect.md) for status and milestones.
