---
name: reproduction-claim-discipline
description: Enforces strict citation and verification discipline for any reproduction or paper-comparison claim in cot-suite. Trigger BEFORE you (a) claim cot-suite reproduces a published paper's reported number, (b) write a comparison between cot-suite-measured values and external paper-reported values, (c) summarize "reproduction status" across multiple methodologies, (d) write or edit AUDIT.md, validation/*.md, or any docs/ file containing reproduction claims, or (e) write a launch positioning paragraph, README headline claim, or Phase summary, or (f) write a commit message containing a reproduction or paper-comparison claim. Fires once per claim, not once per file.
---

# Reproduction Claim Discipline

A reproduction claim is any statement of the form *"cot-suite measured X on model M; paper P reports Y; we reproduce/agree/disagree by Δ."* Reproduction claims are load-bearing for launch positioning. If even one is fabricated, extrapolated, or copied from a stale narrative, the credibility of every other claim in the launch is contaminated.

Each rule below is non-negotiable. If a claim cannot satisfy them all, do not write the claim — surface the gap and stop. The only exception is Rule 7 (explicit draft framing).

---

## Rule 1 — Anti-extrapolation

A claim of the form *"we reproduced X on N models"* is only valid if:

- there exists a script in `scripts/` that iterates over **exactly N** model identifiers, AND
- **exactly N** corresponding output JSONLs exist under `validation/` or `benchmarks/results/`, each containing non-error rows for that model.

If only 1 model was run, the claim is `"single-model reproduction on <model_name>"` — **not** `"cross-validation across N models"`, regardless of:

- how many question IDs were processed
- how many prompts were rendered
- how many judge calls were made
- how many sub-tasks the single model was tested on
- how many parser variants were ablated against the same upstream output

N counts distinct models successfully run end-to-end. Nothing else.

---

## Rule 2 — Citation

Every reproduction claim must cite, inline, all five of:

- **(a)** the script path that produced the cot-suite number (`scripts/<file>.py:<line range>`)
- **(b)** the output JSONL path where the result lives (`validation/<file>.jsonl` or `benchmarks/results/<dir>/<file>.jsonl`)
- **(c)** the source paper and the specific table/figure cell being reproduced against (`arXiv:NNNN.NNNNN <Table N | Fig N> cell <model_name>`)
- **(d)** the exact delta (cot-suite − paper) in percentage points, signed
- **(e)** the tolerance band the claim falls in or out of, and where that band was declared

A claim missing any of (a)-(e) is not a reproduction claim — it is a marketing claim and must not appear in `AUDIT.md`, `validation/*.md`, `docs/`, or `README.md`.

---

## Rule 3 — No secondary sources

Reproduction status comes only from reading the **actual script** + the **actual output JSONL**.

Do **not** derive reproduction status from:

- `docs/` prose
- `AUDIT.md` narrative sections (the ledger in Rule 6 is fine; narrative is not)
- prior session reports, handoff briefings, or rendered HTML reports
- summary `.md` files produced by previous Claude sessions
- any human-authored markdown that paraphrases the JSONL

If the JSONL contains errors or is empty, the reproduction did not happen — regardless of what any prose summary says. A 100% confident summary written on top of an error-only JSONL is still describing a non-event.

---

## Rule 4 — Status vocabulary

Use exactly these status terms with these exact meanings:

| Term | Meaning |
|---|---|
| `reproduced ±Xpp` | script ran; output JSONL exists with non-error rows; delta is X percentage points absolute |
| `within stated tolerance` | use only if a tolerance band was pre-declared in the script or AUDIT.md *before* the run |
| `outside tolerance` | delta exceeds the pre-declared band |
| `single-model reproduction` | N=1 |
| `cross-validation` | only when N≥2 distinct models actually ran successfully |
| `not run` | output JSONL is missing or contains only error rows |
| `pending v0.1.1` | explicitly deferred in committed roadmap docs |

Do not invent new statuses (`"integer-exact reproduction"`, `"directionally validated"`, `"well-bracketed"`, `"paradigm-locked"`, `"bootstrap-robust"`, etc.) for reproduction claims. New terms must be promoted into this table via an AUDIT.md commit before they can appear in any other document.

---

## Rule 5 — Pre-claim checklist

Before writing any reproduction claim, output the following checklist verbatim in your reply (not just in private reasoning — the user must be able to read it) and only proceed if all four items pass:

```
- [ ] I read the script file at <path>:<line range>
- [ ] I read the first 3 non-error rows of the output JSONL at <path>
- [ ] The model identifier in the script matches the model the claim names
- [ ] The N in the claim matches the row count in the JSONL (excluding error rows)
```

If any item cannot be checked, stop and report the gap. Do not write the claim.

The checklist is per-claim, not per-file. If one paragraph contains three reproduction claims, output three checklists.

---

## Rule 6 — Audit trail

Every reproduction claim added to `AUDIT.md` or any file under `docs/` must be appended to the **Reproduction Claims Ledger** — a table in `AUDIT.md` — with these columns:

| date | claim text | script path | output JSONL path | source paper cell | delta_pp | status | commit hash |

No claim ships in a `docs/` file without a corresponding ledger entry. The ledger is the canonical index. If there is a gap between `docs/` and the ledger, `docs/` is wrong, not the ledger.

When the ledger does not yet exist in `AUDIT.md`, create it (one-line entry: `## Reproduction Claims Ledger`) in the same edit as the first claim.

---

## Rule 7 — Override

If the user explicitly asks for `"speculative"` or `"draft"` framing, prefix the claim with `DRAFT — UNVERIFIED:` and skip the Rule 5 checklist. Still add a ledger entry, marked `status="unverified"`, so the draft is recoverable later.

This is the only path that bypasses Rules 1-5. It cannot be invoked implicitly — the user must literally use one of the words `draft` or `speculative` in the request. "Quick summary," "first pass," "rough version," etc. do **not** count.

---

## Examples

### BAD claim

> B4 achieved integer-exact reproduction at ±0.00pp on 4/4 cross-validation models — gpt-4o-mini 13.65%, claude-3.5-haiku 7.51%, qwq-32b 4.55%, chatgpt-4o-latest 0.31%.

Why it fails:

- **Rule 1.** `scripts/validate_b4_arcuschin.py` hard-codes `GPT_MODEL = "gpt-4o-mini"` (line 43). N=1, not 4. No 4-model iteration exists in any script in the repo. The `qwq-32b`, `claude-3.5-haiku`, and `chatgpt-4o-latest` cells are fabricated.
- **Rule 2.** No script path, no JSONL path, no Arcuschin table/figure cell, no delta computation, no tolerance band cited.
- **Rule 3.** The numbers `13.65%`, `7.51%`, `4.55%`, `0.31%` do not appear in `validation/b4_arcuschin_raw_v2.jsonl` — the JSONL contains 100 gpt-4o-mini rows with a v2 PHR rate of 4.88%. The claim is summary-from-narrative, not summary-from-JSONL.
- **Rule 4.** `"integer-exact reproduction"` is not in the status vocabulary table.
- **Rule 5.** Checklist item 4 fails — the claim names 4 models, the JSONL contains 1.

### GOOD claim

> **B4 single-model reproduction on gpt-4o-mini.** cot-suite measured **4.88%** strict-PHR over 41 scorable correct trajectories (`validation/b4_arcuschin_raw_v2.jsonl`, produced by `scripts/validate_b4_arcuschin.py:43-101` with `N_QUESTIONS=100`). Paper-reported reference: **Arcuschin 2503.08679 Fig 1, gpt-4o-mini ~13%**. Delta: **−8.12pp**. Pre-declared tolerance band: 5%-25% (`scripts/validate_b4_arcuschin.py:265-268`). Status: **outside tolerance** — pending v0.1.1 cross-judge + cross-parser + ChainScope ablations.

Why it passes:

- **Rule 1.** Claim is framed as `single-model reproduction` (N=1). No phantom models.
- **Rule 2.** All five citations present: script path + line range, JSONL path, paper cell, signed delta, tolerance band with declaration location.
- **Rule 3.** Numbers come from the JSONL + the script, not from `multi_family_summary.md` or any narrative file.
- **Rule 4.** Uses `single-model reproduction` and `outside tolerance` — both in the vocabulary table.
- **Rule 5.** All four checklist items are verifiable from the cited paths.

### Diff (BAD → GOOD)

```diff
- B4 achieved integer-exact reproduction at ±0.00pp on 4/4 cross-validation
- models — gpt-4o-mini 13.65%, claude-3.5-haiku 7.51%, qwq-32b 4.55%,
- chatgpt-4o-latest 0.31%.
+ B4 single-model reproduction on gpt-4o-mini. cot-suite measured 4.88%
+ strict-PHR over 41 scorable correct trajectories
+ (`validation/b4_arcuschin_raw_v2.jsonl`, produced by
+ `scripts/validate_b4_arcuschin.py:43-101` with `N_QUESTIONS=100`).
+ Paper-reported reference: Arcuschin 2503.08679 Fig 1, gpt-4o-mini ~13%.
+ Delta: -8.12pp. Pre-declared tolerance band: 5%-25%
+ (`scripts/validate_b4_arcuschin.py:265-268`). Status: outside tolerance
+ — pending v0.1.1 cross-judge + cross-parser + ChainScope ablations.
```

What the diff is doing:

1. **Removing fabricated cells.** No qwq-32b / claude-3.5-haiku / chatgpt-4o-latest run exists in the repo. The grep for those model strings outside CoT-trajectory body text returns zero hits.
2. **Replacing `cross-validation` with `single-model reproduction`.** Rule 1: N=1 is single-model, not cross.
3. **Replacing the invented status `integer-exact reproduction at ±0.00pp` with the vocabulary term `outside tolerance`** plus the actual signed delta. Rule 4: stay inside the table.
4. **Adding the five Rule-2 citations:** script path + line range, JSONL path, paper cell, delta, tolerance band declaration site.
