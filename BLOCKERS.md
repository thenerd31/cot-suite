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
