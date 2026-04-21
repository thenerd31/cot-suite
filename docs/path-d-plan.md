# Path D — Full Reproduction of 2510.23966

**Status: scoping doc, pre-execution. Not yet approved.**

**Goal:** Reproduce the headline pooled numbers from Emmons & Zimmermann et al.
(arXiv 2510.23966) Table 1 for Qwen3-235B-A22B-Thinking — pooled legibility
**97.33 ± 0.18%** and coverage **95.27 ± 0.39%** across HLE + GPQA-Diamond +
ARC-AGI + AIME — using Gemini 2.5 Pro as the autorater, exactly matching the
paper's setup.

**Why this supersedes earlier "Stage 2 = Qwen3-235B on GPQA-Diamond alone":**
the paper's Table 1 is pooled-only across the four datasets; a single-dataset
run produces a number that can only be qualitatively compared to the pooled
mean. Path D is the smallest scope that produces a reproduction claim
defensible in a blog post or paper.

---

## 1. Compute architecture

### Model

- **Exact HF repo:** `Qwen/Qwen3-235B-A22B-Thinking-2507`
  (verified via [HF model card](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507), 2026-04-21).
  - This is the **thinking-mode** variant. The non-thinking variant
    `Qwen/Qwen3-235B-A22B-Instruct-2507` produces no reasoning trace and is
    the wrong choice. Differs by one token (`Thinking` vs `Instruct`) — easy to mis-pin, double-check on PR review.
- **Architecture:** MoE — 235B total / **22B activated**, 128 experts /
  8 activated per token.
- **Native context:** 262,144 tokens (256k). 1M with YaRN + Dual Chunk
  Attention if ever needed.
- **HF-card recommended inference (thinking mode, verbatim):**
  `temperature=0.6, top_p=0.95, top_k=20, min_p=0,
  max_new_tokens=32768 (standard) or 81920 (complex math/coding)`.
  Same as Qwen3-14B Stage 1 settings — direct port.

### vLLM version pin

- Stage 1 used **`vllm==0.19.1`** (PyPI latest stable per WebFetch
  2026-04-20). HF model card says **minimum `vllm>=0.8.5`**.
- **[ASSUMPTION]** Stage 1's `0.19.1` does NOT cleanly serve
  Qwen3-235B-A22B-Thinking-FP8. GitHub issue search found three closed
  bugs against this exact model:
  - [#30934](https://github.com/vllm-project/vllm/issues/30934) (Dec 18 2025): FP8 quantization block_n divisibility — closed.
  - [#31045](https://github.com/vllm-project/vllm/issues/31045) (Dec 19 2025): missing `<think>` tags on FP8 — closed.
  - [#31262](https://github.com/vllm-project/vllm/issues/31262) (Dec 24 2025): non-quantized works, FP8 fails on CUDA 13 — closed.
  Fixed-in versions are NOT documented in the issue search results.
  **Action before deploy:** check vLLM PyPI for the latest stable as of
  Path D execution date and verify the three issues are tagged "fixed in
  ≤ that version." If fixes haven't shipped to a stable, fall back to bf16
  serving (much higher GPU cost — see below).

### GPU config

| option | hardware | hourly | notes |
|---|---|---|---|
| **A** | 4×H100 80GB FP8 | ~$15.80/hr (4×$3.95) | tight but feasible if FP8 stable; 4×80=320GB > 235B FP8 (~250GB) |
| **B** | 2×H200 141GB FP8 | ~$9.08/hr (2×$4.54) | **preferred** — fewer GPUs, better mem-headroom (282GB > 250GB), simpler tensor-parallel |
| **C** | 4×H100 bf16 | ~$15.80/hr | 235B bf16 ≈ 470GB > 320GB, **does not fit**; need 8×H100 (~$32/hr) |
| **D** | 4×H200 bf16 | ~$18.16/hr | 235B bf16 fits in 4×141=564GB; fallback if FP8 unusable |

Modal pricing per [modal.com/pricing](https://modal.com/pricing) WebFetch
2026-04-21 — H100 $3.95/hr, H200 $4.54/hr, B200 $6.25/hr.

**Recommendation: option B (2×H200 FP8)** if vLLM FP8 support is stable
at execution time. **Fallback: option D (4×H200 bf16)** if not.

### Per-question latency estimate

- Stage 1 (Qwen3-14B, dense, 1×H100, no quant): **median 73.9 s, p95 171.7 s**.
- Naive scaling: 235B / 14B = 16.8× → 1240 s/q, completely unacceptable.
- **MoE-adjusted:** active params 22B / 14B = 1.57×. Plus routing
  overhead, tensor-parallel comm. **[ASSUMPTION]** ~2× per-question
  latency vs Qwen3-14B is plausible for dense Qwen3-22B-equivalent compute,
  with 1.5–3× as a reasonable range.
- **Estimated p50 per-question:**
  - Optimistic: 90 s (1.2×)
  - Expected: 150 s (2.0×)
  - Pessimistic: 250 s (3.4×) — close to current 5-min cap
- **Implication:** the 5-min per-question cap from Stage 1 will **likely
  fire more often** on Path D. Plan to raise to 8 min (see §5).

---

## 2. Dataset loaders

### 2a. GPQA-Diamond (already done, port directly)

| field | value |
|---|---|
| HF path | `Idavidrein/gpqa`, config `gpqa_diamond` |
| Gating | gated, license auto-approved on accept |
| Sample count | 198 (canonical, already loaded in Stage 1) |
| Format | 4-letter MCQ |
| Extractor | existing `extract_answer_letter()` in `run_qwen3_gpqa.py` |
| Expected Qwen3-235B accuracy | **[ASSUMPTION]** ~75–85% (vs Qwen3-14B's 62%) per scaling trends; not in 2510.23966 |

### 2b. AIME

| field | value |
|---|---|
| HF path candidates | `Maxwell-Jia/AIME_2024` (community), `AI-MO/aimo-validation-aime` (more official). **[ASSUMPTION]** Use AI-MO version for canonicality. |
| Gating | not gated |
| Sample count per year | 30 (15 AIME I + 15 AIME II) |
| Years used by paper | **NOT SPECIFIED** in paper (verified via WebFetch). **[ASSUMPTION]** AIME 2024 + AIME 2025 = 60 problems total, since both are in the model's training-cutoff window. |
| Format | numeric integer answer 0–999 |
| Extractor | new — `re.search(r'(?:answer|=|is)\s*\(?(\d{1,3})\)?', tail)` plus `\boxed{NNN}`; needs care for negative numbers and modular-arithmetic phrasings |
| Expected Qwen3-235B accuracy | ~85–95% (AIME 2024) per Qwen team's own published numbers |

### 2c. HLE (Humanity's Last Exam)

| field | value |
|---|---|
| HF path | `cais/hle` |
| Gating | **gated**, requires license acceptance + use-case justification (delay: hours to days, **[ASSUMPTION]**) |
| Sample count (full) | 2,500 questions |
| Sample count used by paper | **NOT SPECIFIED** in 2510.23966. **[ASSUMPTION]** all 2,500 |
| Format | mixed: MCQ + short-answer, both auto-gradable |
| Schema | not visible on dataset page without login; **action item:** inspect after gating clears |
| Extractor | needs both MCQ (existing) and short-answer (new) variants; HLE provides answer keys per row |
| Expected Qwen3-235B accuracy | ~25–40% per HLE leaderboard (Qwen3-235B is a frontier model on this benchmark; score is intentionally low) |
| Multimodal subset | HLE has image+text rows; Qwen3-235B-A22B-Thinking is text-only. **Filter to text-only rows or accept incorrect answers on multimodal questions.** Paper does not address this. |

**HLE is the critical-path dataset for total spend** — 2500 q × ~150 s/q ≈ 104 GPU-hours ≈ $470 on 2×H200.

### 2d. ARC-AGI

| field | value |
|---|---|
| HF path candidates | `dataset-chollet/arc-agi` doesn't exist; canonical is GitHub `fchollet/ARC-AGI`. HF mirrors include `lgxpsylvain/arc_agi_1` and similar — **[ASSUMPTION]** none have been verified for fidelity to upstream. **Action:** load JSON from the official GitHub repo. |
| Gating | none |
| Versions | ARC-AGI-1 (400 train + 400 eval) and ARC-AGI-2 (varies). Paper does NOT specify which (verified via WebFetch). **[ASSUMPTION]** ARC-AGI-1 eval split (400 tasks). |
| Format | **grid-based**: each task is a list of (input grid, output grid) examples + a test input grid; LLM must produce the test output grid |
| Extractor | **bespoke** — parse model's text output as a 2D integer grid, exact-match against ground truth |
| Expected Qwen3-235B accuracy | ~5–15% — frontier LLMs do poorly on ARC-AGI; this is part of why the paper included it |

**ARC-AGI is the critical-path dataset for format risk.** No canonical
HF version, paper doesn't say which version or how grids were encoded as
text, very low expected accuracy means very few correct trajectories
make it to the autorater stage.

### Per-dataset reproducibility gaps the paper does not address

| dataset | gap |
|---|---|
| HLE | sample count, multimodal handling, gating workflow used |
| GPQA-Diamond | sample count (canonical 198 assumed) |
| ARC-AGI | version (1 vs 2), text encoding scheme, exact-match vs partial-credit grading |
| AIME | year(s), exact answer parsing for non-standard formats |

The paper's own Section 5.1 is silent on all of these. Our reproduction
will document our choices explicitly; deviations from the paper are
unavoidable, and the honest framing is "best-effort reproduction with
documented assumptions."

---

## 3. Gemini autorater integration

### Model identifier

- **`gemini-2.5-pro`** confirmed current (per [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing) WebFetch 2026-04-21, not deprecated).
- API surface: Google AI Studio (free tier exists, paid tier matches Vertex pricing) OR Vertex AI. **Recommendation: Google AI Studio paid tier** — simpler than Vertex GCP project setup.

### Pricing (paid tier, ≤200k token prompts)

- Input: **$1.25 / 1M tokens**
- Output: **$10.00 / 1M tokens**
- Context caching available at $0.125 / 1M input (10× cheaper) — useful since the Appendix C autorater prompt template is constant ~2.6k tokens that we'd send identically every call

### Per-call cost estimate

Using Stage 1's distribution as proxy:
- Input: ~2,600 (template) + ~300 (question) + ~6,000 (CoT, median) + ~150 (final answer) = **~9,050 tokens/call**
- Output: ~400 tokens/call (Haiku Stage 1 mean)
- Per-call cost: $1.25 × 0.009 + $10.00 × 0.0004 = **~$0.015** without caching, **~$0.005** with caching of the 2,600-token template

### Total Gemini cost

- Expected correct trajectories across 4 datasets:
  - GPQA-D: 198 × ~80% = ~158
  - AIME: 60 × ~90% = ~54
  - HLE: 2,500 × ~30% = ~750
  - ARC-AGI: 400 × ~10% = ~40
  - **Total: ~1,000 correct trajectories**
- Without caching: 1000 × $0.015 = **~$15**
- With caching: 1000 × $0.005 = **~$5**

### Code changes needed

Existing autorater abstraction in `cotdiv/autoraters/legibility_coverage.py`
takes a `GraderClient` Protocol — provider-agnostic. `cotdiv/models/clients.py`
already has a `GeminiClient` stub that calls `google-genai`. **Estimated work:
~30 lines** to flesh out the Gemini path properly + add context-caching
config. Plus ~20 lines in `run_qwen3_gpqa.py` (or its Path D sibling) to
swap in the Gemini autorater behind a `--autorater` flag.

---

## 4. Cost model

### Per-dataset breakdown

Assumptions:
- GPU: 2×H200 FP8 ≈ $9.08/hr (option B above)
- Per-question wall: 150 s expected (75 s optimistic / 250 s pessimistic)
- Cold start: ~5 min (one-time per dataset, since Modal scales down between
  datasets unless we batch them in one execution)

| dataset | questions | expected GPU-hours | Modal ($) | Gemini ($) | total ($) |
|---|---|---|---|---|---|
| GPQA-Diamond | 198 | 8.3 | 75 | 2.5 | 78 |
| AIME | 60 | 2.5 | 23 | 0.8 | 24 |
| HLE | 2,500 | 104 | 945 | 11 | 956 |
| ARC-AGI | 400 | 17 | 154 | 0.6 | 155 |
| **TOTAL** | **3,158** | **132** | **$1,197** | **~$15** | **~$1,212** |

**This is dramatically higher than the earlier "$200–$280" estimate.** The
earlier number assumed I was misremembering HLE size; HLE at 2,500
questions dominates everything.

### Optimistic / expected / pessimistic

| scenario | per-q wall | total Modal | total |
|---|---|---|---|
| Optimistic (75 s/q, FP8 2×H200) | — | $600 | **$615** |
| Expected (150 s/q, FP8 2×H200) | — | $1,200 | **$1,215** |
| Pessimistic (250 s/q, bf16 4×H200, more timeouts) | — | $4,000 | **$4,015** |

### Modal cap implications

- Stage 1 spent ~$22.20 on Modal in one billing cycle.
- Workspace cap auto-raises in $50 increments after each successful charge.
  Currently cap = $100 (per screenshot 2026-04-20).
- To execute Path D expected ($1,200 Modal) we need cap ≥ ~$1,300:
  - Cap raises ~$50 per charge cycle. From $100, that's ~24 raise cycles.
  - **In practice**: Modal raises caps faster as your charge history builds.
    [ASSUMPTION] After ~$200–300 of charges, cap should reach $400+. After
    ~$800, cap should be limit-of-payment-method.
- **Action before Path D:** run a small intermediate workload (~$50) to
  trigger the first auto-raise; verify cap moves; iterate. Or contact Modal
  support to raise cap to $1,500 in advance.

### Fixed vs variable

| item | dataset-size dependent? |
|---|---|
| Modal image build | one-time ($0) |
| Cold-start GPU time | one per dataset (~$1 each, fixed) |
| Per-question Qwen3 inference | **variable**, dominates |
| Gemini autorater per call | **variable**, much smaller share |
| Code changes | one-time engineering, ~3–5 days |

---

## 5. Risk register

### vLLM Qwen3-235B FP8 stability

- Three closed bugs from late 2025, no documented "fixed-in" version visible
  in issue search.
- Mitigation: confirm fix versions via vLLM CI/changelog before deploy. If
  no clear fix, fall back to bf16 (4×H200, ~$18/hr) — costs ~2× more
  but de-risks the most critical infrastructure.

### HLE gating timeline

- HLE gate requires use-case justification per CAIS terms. **[ASSUMPTION]**
  delay 1–7 days from application to access. Apply at the start of the
  Path D timeline, not mid-execution.

### ARC-AGI format transfer

- No canonical HF dataset, no canonical text-encoding scheme, paper doesn't
  specify. We will need to make and document choices that may not match
  the paper.
- Mitigation: cite Chollet's original GitHub repo as source-of-truth, use
  the simplest text-encoding (rows of comma-separated digits, blank line
  between input and output grids), document explicitly.

### Inference timeout

- Stage 1 5-min cap fired once on 198 questions (Qwen3-14B, p95 latency
  171s). Path D's larger model with p95 estimated at ~3× that = ~9 minutes,
  routinely above a 5-min cap.
- **Proposal: raise cap to 600s (10 min) for Path D.** If 10 min still
  fires often (>10/dataset), reassess at the first dataset's checkpoint.

### Sample-size ambiguity

- Paper does not specify HLE, ARC-AGI, AIME sample counts. Our pooled
  number will weight by **our** counts, not the paper's. Even with
  identical autorater output, our pooled mean can differ from
  97.33% / 95.27% purely because of dataset-mix weighting differences.
- **Fundamental implication:** "reproduction within ±2pp" is potentially
  unattainable because the comparison isn't apples-to-apples. Honest
  framing: "applied paper's methodology to our dataset choices, observed
  X / Y" rather than "reproduced 97.33%."

### FP8 quantization drift on legibility/coverage

- **[ASSUMPTION]** FP8 should not significantly affect output token
  distribution → autorater scores should not differ from bf16. But this
  has not been measured for Qwen3-235B-Thinking specifically.
- Mitigation: run a 50-sample subset on bf16 before the full FP8 run;
  compare per-sample autorater scores. If divergence > 0.2 mean on either
  axis, run the full thing in bf16.

---

## 6. Execution sequence

### Suggested ordering (easiest-first)

1. **GPQA-Diamond** — pipeline already validated in Stage 1. Just swap
   model + autorater. Lowest risk. ~$78, ~2 hr wall-clock.
2. **AIME** — numeric extraction is mechanical, small dataset. ~$24, ~1 hr.
3. **HLE** — biggest cost item; enter only after #1 + #2 confirm cost
   per-q estimate is accurate. ~$956, ~26 hr (potentially overnight runs).
4. **ARC-AGI** — highest format risk; do last after pipeline is otherwise
   stable so format work is the only uncertainty. ~$155, ~4 hr.

### Checkpoint structure

After each dataset:
- Compute per-dataset legibility/coverage mean ± SD.
- Verify accuracy is in the expected band (sanity check on inference setup).
- Update cost-actual vs estimate; if actual / estimate > 1.5×, pause the
  remaining sequence and reassess.
- **Do not chain dataset N → N+1 automatically.** Manual go-ahead between each.

### Calendar (starting May 7 2026)

| date | activity | wall-clock |
|---|---|---|
| May 7 | Apply for HLE gate (parallel: write dataset loaders for AIME, ARC-AGI; write Gemini integration) | 2 days dev |
| May 9 | HLE gate status check; deploy Path D Modal image | 1 day |
| May 10 | GPQA-Diamond run + report (Dataset 1 checkpoint) | half day |
| May 11 | AIME run + report (Dataset 2 checkpoint) | half day |
| May 12–14 | HLE run (chunked overnight, possibly 2–3 sessions) + report | 2–3 days |
| May 15 | ARC-AGI run + report (Dataset 4 checkpoint) | 1 day |
| May 16 | Pooled aggregation + paper-comparison delta analysis | half day |

Total calendar: ~10 days from May 7 → May 16. Engineering work parallelized
with HLE gating delay.

### Critical-path item

**HLE gating + HLE compute cost.** HLE is 79% of the total Modal spend
and gates total cost. If HLE access denied or its cost runs higher than
$1,200 alone, Path D becomes unfeasible at our budget level.

---

## 7. Decision points

### Abort conditions during execution

- **Per-question latency p50 > 8 min** on Dataset 1 (GPQA-Diamond) → stop,
  reassess GPU config or quantization. Continuing on $9/hr GPUs at 8 min/q
  pessimistic compounds quickly.
- **Inference-timeout rate > 10% in any dataset** → stop, raise cap or
  diagnose.
- **Modal cumulative spend > $400 before any complete dataset** → stop,
  reassess cost model.
- **Any FP8 numerical divergence detected** in 50-sample bf16 vs FP8 spot
  check → stop, switch to bf16, re-cost.
- **Gemini parse-failure rate > 5% on any dataset** → stop, investigate
  prompt or rate-limit issue.

### Partial-completion scenarios

- **2/4 datasets done (GPQA-D + AIME):** ~$100 spent, ~218 trajectories
  total. Pooled number across 2 datasets is publishable as
  "applied 2510.23966 methodology to two of four paper datasets." Not a
  full reproduction but a meaningful intermediate result.
- **3/4 done (GPQA-D + AIME + ARC-AGI, no HLE):** ~$255 spent,
  ~252 trajectories. Skips HLE which is the biggest contributor; pooled
  number is heavily skewed. Honest framing: "GPQA-D + AIME + ARC-AGI
  pooled, HLE excluded due to gating/budget."
- **HLE only (skip the others):** $956 spent on the most informative
  dataset; the paper's pooled number is dominated by HLE anyway, since it's
  ~80% of question count. **Counterintuitively, HLE-only might be the most
  paper-faithful single-dataset reproduction.**

### Fallback framing

If pooled mean diverges > ±2pp from paper:

> "We attempted reproduction of 2510.23966's pooled Qwen3-235B-A22B-Thinking
> legibility (97.33%) and coverage (95.27%) on the four cited datasets. Our
> measurements yielded **X% / Y%**. Sources of divergence we identify:
> (a) the paper does not specify HLE/ARC-AGI/AIME sample counts or
> selection, so our dataset mix differs from theirs; (b) we used Claude
> Haiku 4.5 in addition to Gemini 2.5 Pro as a cross-rater experiment;
> (c) we used FP8 quantization (paper does not specify quant). Differences
> on individual datasets are: [breakdown]."

This is a defensible, honest framing whether or not the headline number
matches.

### Sunk-cost tripwire

**$200 spent before any complete dataset result lands** → stop and
report, regardless of whether anything is technically broken. This
indicates per-question costs are higher than modeled and the full Path D
will exceed our $1,200 expected estimate by enough that we should
re-decide rather than push forward.

---

## Open questions for human review

1. **HLE sample count:** if the paper truly used all 2500 questions, our
   reproduction needs to do the same. If Path D budget doesn't accommodate
   full HLE (~$956 alone), we need to either (a) get cap raised to $1,500+,
   (b) accept incomplete HLE and document the subset, or (c) abandon Path D
   in favor of GPQA-D + AIME pooled.
2. **ARC-AGI version:** without knowing if the paper used v1 or v2, we'll
   pick one and document. Want guidance on default.
3. **Multimodal HLE:** Qwen3-235B-A22B-Thinking is text-only. ~25% of HLE
   is multimodal. **[ASSUMPTION]** filter to text-only and report the
   fraction excluded; alternative is to record all rows but flag image
   rows as "model can't see this" and accept low accuracy.
4. **Gemini cross-rater experiment:** Stage 1 used Haiku-only. Path D's
   spec includes "Haiku + Gemini cross-rater." Confirm: do we run BOTH
   raters (doubles autorater cost from $15 → $30, still negligible) or
   only Gemini for paper-faithfulness?

---

## Appendix: source links

- Paper: https://arxiv.org/html/2510.23966v1
- Qwen3-235B model card: https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507
- vLLM releases: https://github.com/vllm-project/vllm/releases
- vLLM Qwen3-235B FP8 issues: https://github.com/vllm-project/vllm/issues?q=Qwen3-235B-A22B-Thinking+FP8
- Modal pricing: https://modal.com/pricing
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- HLE dataset: https://huggingface.co/datasets/cais/hle
- AIME 2024 candidate: https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
- AIME official: https://huggingface.co/datasets/AI-MO/aimo-validation-aime
- ARC-AGI source: https://github.com/fchollet/ARC-AGI
- Stage 1 results (for cost extrapolation baseline): `benchmarks/results/qwen3_14b_gpqa_full/`
