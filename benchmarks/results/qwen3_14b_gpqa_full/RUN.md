# Stage 1 run — Qwen3-14B on GPQA-Diamond, Haiku 4.5 autorater

**Date:** 2026-04-20 (initial run) and 2026-04-21 (resume + completion)
**Model:** [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B) (thinking mode enabled)
**Dataset:** [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa), config `gpqa_diamond`, split `train` — 198 expert-curated questions, gated (requires HF license acceptance + `HF_TOKEN`)
**Autorater:** Anthropic `claude-haiku-4-5` with the verbatim Emmons & Zimmermann (arXiv [2510.23966](https://arxiv.org/abs/2510.23966)) Appendix C prompt
**Prompt SHA-256:** `ac1e0ac4044b0a64816bc3c4424e547a462e64d18cf0f5555dec7ed5b42aaa67` (`emmons_zimmermann_v1`)

## Reproduce

Two-step: deploy the Modal inference app, then run the driver against it.

```bash
# 1. Deploy Qwen3-14B vLLM server to Modal (one-time; re-use across runs)
modal deploy scripts/modal_qwen3_14b.py

# 2. Run the driver against the deployed endpoint
python scripts/run_qwen3_gpqa.py \
    --output-dir benchmarks/results/qwen3_14b_gpqa_full
```

Required environment variables (via `.env` or exported):

- `ANTHROPIC_API_KEY` — for the Haiku 4.5 autorater
- `HF_TOKEN` — for the gated GPQA-Diamond dataset download and for Modal's Qwen3-14B weight fetch (Modal secret: `hf-token`)
- `modal setup` must have been run at least once so `~/.modal.toml` has valid tokens

The driver supports `--limit N` (first N questions, used for the 5-question real dry-run) and `--start-from N` (resume an interrupted run at index N; validates row-count and question-id parity against existing `results.jsonl`).

## Inference configuration

All values are the Qwen3-14B HF model-card recommendations for thinking mode (defaults in `scripts/run_qwen3_gpqa.py:GenerationConfig`):

| setting | value | source |
|---|---|---|
| `temperature` | 0.6 | HF card, Usage §Thinking Mode |
| `top_p` | 0.95 | HF card |
| `top_k` | 20 | HF card |
| `min_p` | 0.0 | HF card |
| `max_tokens` | 32768 | HF card (default; 81920 recommended for complex math/coding) |
| `seed` | 0 | cot-suite choice for reproducibility; paper does not prescribe |
| `enable_thinking` | `True` | explicit via `tokenizer.apply_chat_template(...)` |

**Autorater run parameters:**
- `n_samples` = 1 per correct trajectory (not the paper's 5). Justified by the variance pilot on 2026-04-20: three sequential Haiku calls on the same trajectory converged exactly (0-point spread on the clean 4/4 case, 1-point legibility spread on the boundary case). See `docs/autorater-notes.md`.
- `MAX_AUTORATER_TOKENS = 2048` as of commit `21a0a38`. Was 1024 during this run; bump applied afterward to preempt the two parse failures documented below.

## Cost

| item | value | source |
|---|---|---|
| Modal H100 spend | **~$21.6** across the split runs (dry-run $0.59 + crash-loop $0.14 + first full-run leg $7.23 + resume leg $13.6) | `modal billing report --for today` taken 2026-04-20 and 2026-04-21 |
| Anthropic Haiku spend | **~$1.30 estimated** (120 autorater calls + ~10 earlier diagnostic/smoke calls) | Estimated from per-call usage observed in the 2026-04-20 smoke test (~$0.018 / 2-call batch → ~$0.009 per call); actual total below the Anthropic $40 threshold that paused and was raised that evening |
| **Total Stage 1** | **~$22.9** | — |

Budget ceiling was $35 (Stage 1 spec). Came in at ~65% of ceiling.

## Wall-clock

- **Total run time:** 184 minutes (3h 4m) for the 117-question resume leg on a warm container; initial 81-question leg ~2h; combined ~5h elapsed including downtime between legs.
- **Per-question latency** (all 198, seconds): min 0 (the 1 timeout row), p25 44.9, **p50 73.9**, p75 120.5, p90 148.8, p95 171.7, p99 211.7, max 245.6, mean 83.3.
- **Cold start:** 153.6 s observed (container schedule + Qwen3-14B weight download + vLLM engine init + CUDA graph capture). Recorded once per leg; 1 true cold start on this run.

## Results

### Accuracy

- **Correct / usable: 122 / 197 = 61.93%** (97.5% CI roughly [54.8%, 68.7%])
- Public Qwen3-14B baseline on GPQA-Diamond is reported at 55–60%; our 62% lands at the upper edge of plausible, consistent with the baseline within sampling error for n=197.
- Denominator excludes the 1 inference-timeout row (`gpqa_diamond_127`). Inclusive accuracy is 122 / 198 = 61.62%.

### Autorater distribution (n=120 correct+parseable trajectories)

**Legibility — mean 3.608 ± 0.612**

| score | count | pct |
|---|---|---|
| 4 | 81 | 67.5% |
| 3 | 31 | 25.8% |
| 2 | 8 | 6.7% |
| 1 | 0 | — |
| 0 | 0 | — |

**Coverage — mean 3.417 ± 0.856**

| score | count | pct |
|---|---|---|
| 4 | 74 | 61.7% |
| 3 | 27 | 22.5% |
| 2 | 14 | 11.7% |
| 1 | 5 | 4.2% |
| 0 | 0 | — |

Coverage has the wider spread (SD 0.86 vs 0.61 on legibility). Qwen3-14B occasionally produces skeletal or procedurally thin CoTs that Haiku correctly flags as incomplete even when the final answer is right.

### Thinking-token distribution (all 198)

| pct | tokens |
|---|---|
| min | 0 (timeout row) |
| p25 | 3,326 |
| p50 | 6,014 |
| p75 | 9,609 |
| p90 | 11,995 |
| p95 | 13,423 |
| p99 | 19,261 |
| max | 19,318 |
| mean | 6,582 |

Never hit the `max_tokens=32768` ceiling. The p99 at 19k leaves ~40% headroom — no question was truncated mid-reasoning.

### Post-hoc rationalization (Arcuschin 2503.08679 detector)

Ran `scripts/run_post_hoc_rationalization.py` against the full 197-row usable set as a secondary analysis. Uses an LLM-as-judge single-call detector (Haiku 4.5) with a cot-suite-authored prompt — **not** verbatim from the paper; see `AUDIT.md` for the provenance caveat.

- **Correct + diverged + unacknowledged: 7 / 122 = 5.7%** (the Arcuschin implicit post-hoc pattern on correct answers)
- Correct + diverged + acknowledged: 1 / 122 (reasoning-to-final flip that the model explicitly called out — not the pathology)
- Incorrect + diverged + unacknowledged: 12 / 74 = 16.2%
- Judge mean confidence on correct+unack cases: 0.84; on incorrect+unack: 0.92
- Judge errors: 1 / 197 (JSON parse failure on an unescaped character in the judge's response; raw body preserved in the per-row output jsonl)
- Per-row output: `benchmarks/results/qwen3_14b_gpqa_full/post_hoc_rationalization.jsonl`

## Known gaps and deferred work

1. **Gemini 2.5 Pro cross-rater validation — attempted 2026-04-21, blocked.** `scripts/run_gemini_cross_rater.py` exists and was run; all 120 calls returned `400 INVALID_ARGUMENT: API key not valid` because the `GOOGLE_API_KEY` in the local `.env` was still the 7-character `AIza...` placeholder, not a real Google AI Studio key. Failed attempt documented in `gemini_cross_rater.jsonl` (120 error rows preserved as evidence). Matching the paper's Gemini-2.5-Pro autorater is deferred to v0.1.x; re-run once a real key is wired in.

2. **`gpqa_diamond_127` inference timeout.** Qwen3-14B produced zero output within the 5-minute per-question cap. Row preserved in `results.jsonl` with `inference_timeout=True`; excluded from accuracy / autorater denominators. The question's correct answer is recorded so a future re-run can target just this index via `--start-from 127 --limit 128`.

3. **2 parse failures at Q66 and Q143.** Haiku's Appendix C JSON response exceeded the 1024 output-token cap mid-JSON, causing `KeyError: legibility_score` in our parser. Raw bodies captured per Fix #2 (commit `ccbd8df`). Bumped to 2048 in commit `21a0a38` — should resolve the class of failure but **this run was not rerun retroactively**, so these two trajectories remain unrated in the 120-row autorater denominator. A Q66 diagnostic re-call on 2026-04-21 confirmed Haiku emits 4/4/4 on reparse when token budget is sufficient.

4. **Temperature 0.6 means runs are non-deterministic even with seed=0.** A re-run with identical command should reproduce Qwen3-14B's per-question latency distribution within ~10% but exact answers, accuracies, and autorater scores will differ. For a locked reproduction, pin a specific `seed` + temperature=0 (greedy) — but the HF card explicitly warns against greedy decoding in thinking mode, so this is a known tension.

5. **Published Qwen3-14B GPQA-Diamond numbers are not a clean comparison target.** The 55–60% public range is collected under varying inference configs (thinking budget, sample counts, prompt template). Our 61.93% is plausible but not directly benchmarked against a particular published setup.

6. **Single-shot autorater vs the paper's 5-shot.** Our variance pilot showed 0-1 point spread on the two trajectories we sampled; this does not generalize to the full distribution. A subsample 5-shot run (~20 questions) should be done before any "cot-suite autorater variance is negligible" claim.

## Planned CLI reproduction (v0.1.0)

Once `cot-suite`'s CLI is smoke-tested, the reproduction command becomes:

```bash
cot-suite evaluate \
    --model qwen3-14b-thinking \
    --dataset gpqa-diamond \
    --autorater haiku-4.5 \
    --output-dir benchmarks/results/qwen3_14b_gpqa_full
```

As of this writing the top-level CLI surface is the `cotsuite` script (entry point: `cotsuite.cli:app`); the higher-level `cot-suite evaluate ...` sub-command is a v0.2 item and the current `cotsuite` script exposes lower-level `score`, `eval`, and `prompts` commands.

## Artifacts

- `results.jsonl` — 198 rows (per-question schema). One line per GPQA-Diamond question in dataset order. Each row contains question, CoT, final answer, correctness, autorater scores, timing, and parse-error details.
- `summary.json` — aggregated statistics plus the canonical autorater prompt hash and generation config.
- `post_hoc_rationalization.jsonl` — 197 rows from the Arcuschin detector pass. Schema: `{question_id, is_correct, final_answer, cot_conclusion, diverged, acknowledged, confidence, judge_reasoning, raw_response, error}`.
- `gemini_cross_rater.jsonl` — 120 error rows from the 2026-04-21 cross-rater attempt. Kept as evidence of the deferred work.
