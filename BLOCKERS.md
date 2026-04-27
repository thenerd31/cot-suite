# BLOCKERS.md

Preconditions for any release that claims reproduction of specific paper
numbers. Every entry is a concrete gap between current implementation and
the source paper's methodology that would invalidate a published-number
reproduction claim.

**Mutation policy:** entries are resolved in-place (add a `Resolved in
commit <hash>` line; don't delete). Resolving a blocker unblocks exactly
the claim it was paired with — not others.

---

- **Length-weighted AOC alignment** — `early_answering` and `mistake_injection`
  weight by `round(f * n)` (sentence count at prefix fraction). Paper
  weights by token count through sentence k — materially different for
  variable-length sentence distributions.
  - *Blocks:* "reproduces Lanham Table 2 AOCs within stated error bars."
  - *Paper:* Lanham 2307.13702 §3.1–3.2.

- **Mistake-generator model class** — the ValueError guards in `5a81ac7`
  force callers to pass a separate model, but do not guarantee a non-RLHF
  base model. Per the paper, the mistake generator must be a pretrained
  (non-RLHF-tuned) LM to avoid the RLHF model over-correcting its own
  proposed corruption. The API accepts any string; live reproduction needs
  a stronger contract (e.g., default to `qwen/qwen3-14b-base`, or a
  registry-time check on known RLHF-tuned model ids).
  - *Blocks:* "reproduces Lanham mistake-injection AOCs."
  - *Paper:* Lanham 2307.13702 §3.2.

- **`scripts/verify_keys.py` — one-shot preflight for external API keys.**
  Third-time placeholder-key incident (Modal tokens 2026-04-20, Google
  AI Studio 2026-04-21, OpenAI 2026-04-23) each cost wasted runtime,
  log noise, and a false negative on a run that couldn't actually
  execute. A 30-line script that makes one minimal API call to each
  configured provider (Anthropic, OpenAI, Google AI Studio, HuggingFace,
  Modal) and fails with a provider-specific error for placeholder /
  expired / unauthorized keys. Should be runnable as
  `python -m cotsuite.verify_keys` or `cot-suite verify-keys` and called
  at the top of every spend-incurring script. v0.1.1 deliverable.
  - *Blocks:* "runs cleanly from a fresh .env without wasted spend."
  - *Resolved in commit:* `8821b85` — `src/cotsuite/verify_keys.py` ships
    checks for Anthropic / OpenAI / HuggingFace / Modal (token info);
    `--budget-check` degrades to warning for non-admin keys (end-user
    sk-ant-... keys can't hit the admin billing endpoint). Wired into
    `run_qwen3_gpqa.py`, `validate_b1_lanham.py`, `validate_b3_chen.py`,
    `validate_b4_arcuschin.py` as import-time `require_keys(...)` calls.
    Google AI Studio check deferred — no live Gemini runs on the queue.

- **True reproduction of 2510.23966 pooled four-dataset numbers — v0.2
  deliverable.** The paper's Table 1 headline (Qwen3-235B: Legibility
  97.33 ± 0.18%, Coverage 95.27 ± 0.39%) is pooled across HLE,
  GPQA-Diamond, ARC-AGI, and AIME — **no per-dataset breakdown is in
  the paper** (verified by WebFetch 2026-04-20). A single-dataset
  reproduction cannot match the pooled mean; there is no middle-ground
  "partial reproduction" claim worth making. The *only* path to a tight
  paper-comparison claim is running all four datasets × Qwen3-235B-A22B-
  Thinking with Gemini 2.5 Pro autorater and computing a pooled mean.
  Estimated cost $200–$280 (one model × four datasets × 4–8× H100 hours
  + Gemini API autorater). Upgraded from "post-v0.x milestone" to
  **"v0.2 priority"** on 2026-04-20 after this became the only defensible
  reproduction framing. Gated on Modal workspace cap seasoning to $300+
  after Stage 1's successful charge.
  - *Blocks:* "reproduces 2510.23966 Table 1 within stated error bars"
    — the library's headline reproduction claim.
  - *Paper:* Emmons & Zimmermann et al. 2510.23966, Table 1 + §5.1.
