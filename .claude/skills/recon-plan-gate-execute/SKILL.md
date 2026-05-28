---
name: recon-plan-gate-execute
description: Use this skill whenever starting work on a new paper integration, writing or modifying scripts under scripts/validate_*.py, running operations that cost >$5 in API spend, writing code that produces JSONLs under validation/ or benchmarks/results/, or closing out a Phase milestone. Enforces the four-stage recon-plan-gate-execute pattern that the cot-suite project has empirically found prevents over-extrapolation and wasted spend.
---

# Recon → Plan → Gate → Execute

Every new paper integration, validation script, or methodology addition to cot-suite must move through four stages with explicit user approval between each. This pattern was adopted after observing repeated cases where compressed workflows (recon+plan in one message, or plan+execute in one message) produced overclaimed results and wasted spend.

## Stage 0 — Reconnaissance (read-only, no spend, no code changes)

Required before any new paper integration begins. Produces a one-page recon report covering:

- The source paper, its exact methodology section (cite section numbers), and what it claims to measure.
- Released artifacts: code repo URL, dataset location, supplementary materials. Verify each is accessible without authentication.
- Whether the paper's reported cells can be re-computed from its own released artifacts. If a paper does not reproduce against its own released data, cot-suite cannot reproduce it either — flag this as a blocker.
- What cot-suite would need to add: new scorer, new dataset loader, new judge prompt, new metric.
- Estimated cost in USD across the full validation. Estimated runtime.
- At least one explicit risk that could make this not work (deprecated checkpoint, missing dataset, incompatible licensing, etc).

The recon report is delivered to the user. No code is written in Stage 0.

## Stage 1 — Plan Presentation

After user approves the recon report, produce a plan containing:

- **What the cot-suite implementation produces** (one paragraph).
- **Target paper cells with deltas declared in advance.** Each cell: paper-reported value, cot-suite-expected value range, tolerance band. Tolerance bands must be declared *before* the run, not after.
- **Success criterion as a single sentence**, falsifiable by a single number. Example of valid: "Within ±5pp on the released gpt-4o-mini cell of ~13%." Example of invalid: "Reproduces the paper's findings."
- **Failure case as a single sentence.** What happens if the success criterion is not met? Does the project deferral to v0.1.1, or is it a launch blocker?
- **Commit targets.** What gets committed where: script path, output JSONL path, AUDIT.md ledger entry, validation/<paper>.md updates.
- **Cost estimate** broken down: judge calls, model-under-test calls, dataset access, infrastructure.

Stop after the plan. Wait for explicit user approval. Do not begin Stage 2 until the user types an affirmative response.

## Stage 2 — Gated Execution

After user approves the plan:

- Run the smallest possible validation first. Dry-run with N=10 if cheap, or single model on single subtask, before scaling to the full budget.
- Verify infrastructure works on the small run before scaling. This catches model-identifier typos, deprecated checkpoints, dataset access failures, and prompt-template bugs at $1 spend rather than $50 spend.
- If the small run produces unexpected results — out-of-band deltas, parse failures above 1%, unexpected error patterns — stop and report. Do not scale up to the full budget. The plan's failure case applies.

## Stage 3 — Report With Provenance

The report after Stage 2 completion must include:

- **Script path** that produced the data, with commit hash.
- **Output JSONL path(s)** with row counts and error-row counts.
- **Actual deltas** per cell against pre-declared tolerance.
- **Status against pre-declared tolerance**, using exactly the vocabulary from the reproduction-claim-discipline skill.
- **Raw numbers**, not summaries. If 3 cells were checked, report 3 deltas — not "all within band."
- **AUDIT.md ledger entry** added in the same commit as the validation result.

If the reproduction failed to meet pre-declared tolerance, this is reported as a finding. Do not massage the wording to make a failure look like a success. "Out of band — pending v0.1.1 ablation" is a valid finding. "Reproduced within stretch tolerance" is not.

## Anti-Pattern: Stage Compression

Under no circumstances should this workflow be compressed:

- "Recon and plan in one message" — violation. Recon must be reviewed before planning.
- "Plan and execute in one message" — violation. Plan must be approved before execution.
- "Execute and report in one message" — acceptable only if Stage 2's small run already passed gating; the report still requires explicit Stage 3 structure.
- "I'll just run a quick test first" — violation if the test costs >$1 or writes to validation/ or benchmarks/results/. Quick tests still require a plan.

Each stage requires user input before the next begins. The user can shortcut this only with the explicit phrase "skip to <stage>" — and even then, the skipped stages are noted in the resulting commit message so the audit trail records the deviation.
