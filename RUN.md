# RUN.md — Stage 1 & Stage 2 benchmark plan

Two-stage execution against GPQA-Diamond with the 2510.23966
legibility + coverage autorater. Staged because Modal's $100 workspace
budget auto-raises after the first successful charge (threshold $50),
so a smaller first run "seasons" the cap before the larger second run.

Both stages use the verbatim 2510.23966 Appendix C prompt shipped at
`src/cotdiv/autoraters/prompts/emmons_zimmermann_v1.txt` (canonical
SHA-256 `ac1e0ac4…`). Only the **target model** and the **autorater set**
change between stages.

## Stage 1 — Qwen3-14B on GPQA-Diamond

**Framing.** Applies 2510.23966's methodology to a smaller open-weights
model the paper did not evaluate. Pipeline validation + scale-down data
point. NOT a reproduction of Table 1.

**Target:** `Qwen/Qwen3-14B` via vLLM 0.19.1 on a single Modal H100.
Thinking mode enabled via the HF-recommended defaults (`temperature=0.6`,
`top_p=0.95`, `top_k=20`, `min_p=0`, `max_tokens=32768`, `seed=0`).

**Dataset:** `Idavidrein/gpqa` config `gpqa_diamond` — 198 questions,
gated (requires `HF_TOKEN` with license accepted).

**Autorater:** Claude Haiku 4.5, single-shot (n=1). Matches the smoke
test; justified by the 2026-04-20 variance runs (3/3 convergence on the
4/4 case, cov-spread=0 on the boundary case).

**Expected cost:** $10–$30 Modal + ~$0.50 Anthropic.
**Budget ceiling before abort:** $50 for the full 198-question run.

**Checkpoints (do not skip):**
1. Modal auth + workspace budget cap confirmed — ✅ 2026-04-20 ($100).
2. Structural stub dry-run (`--dry-run`, zero outbound) — verify wiring.
3. 5-question real dry-run — extrapolate full-run cost; abort if >$50.
4. Full 198-question run — report per §"What to report" below.

## Stage 2 — Qwen3-235B-A22B-Thinking on GPQA-Diamond

**Framing.** Reproduction of 2510.23966's Qwen3 row restricted to
GPQA-Diamond, with cross-rater validation. Note: the paper's Table 1 is
pooled across HLE + GPQA-Diamond + ARC-AGI + AIME with no per-dataset
breakdown (verified 2026-04-20). A single-dataset reproduction cannot
match the pooled mean. True pooled reproduction is in `BLOCKERS.md` as a
post-v0.x milestone.

**Target:** `Qwen/Qwen3-235B-A22B-Thinking` — MoE, requires 4–8× H100
on Modal. Same inference settings as Stage 1 (Qwen3's thinking-mode
defaults).

**Autoraters (both):** Claude Haiku 4.5 AND Gemini 2.5 Pro. Turns the
autorater divergence from Stage 1 into a cross-rater validation
experiment. Report both columns side-by-side.

**Expected cost:** $30–$80 Modal + ~$2 Anthropic + ~$2 Google.

**Preconditions before Stage 2 executes:**
1. Stage 1 completes with parseable results.
2. Modal cap has risen to ≥$200 via Stage 1's successful charge
   (user verifies in Modal dashboard; auto-raise trigger is $50 usage).
3. Explicit user go-ahead. Do not auto-proceed from Stage 1.

## What to report at each full-run completion

Per the task spec:
- Actual Modal run cost.
- Actual Anthropic (+ Google in Stage 2) autorater cost.
- Target-model accuracy on GPQA-Diamond (sanity-check our inference vs
  the model's own published numbers).
- Mean ± SD for legibility and coverage across correct trajectories.
- Our numbers vs paper numbers with delta (Stage 2 only; Stage 1 has no
  paper comparator).
- Count and example of any trajectories where the autorater returned
  non-parseable output.
- Total wall-clock time.

## What NOT to do in either stage

- Do not add a second target model. Stage 1 = 14B only, Stage 2 = 235B
  only. No leaderboard creep.
- Do not add a second dataset. GPQA-Diamond only.
- Do not switch the autorater prompt. Verbatim Appendix C, SHA-verified.
- Do not ensemble autorater calls (n>1). Variance data doesn't justify
  the cost.
- Do not run Stage 2 until Stage 1 completes AND user approves.
