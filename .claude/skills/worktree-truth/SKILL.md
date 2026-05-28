---
name: worktree-truth
description: Use this skill whenever summarizing "what has been done" on a phase or methodology, describing the state of cot-suite to the user or in an external document, answering "where are we on X", writing a context-transfer briefing, or producing any handoff document. Forces all status claims to be grounded in the actual filesystem state (git ls-files, git status, file contents at HEAD), not in conversation memory, prior briefings, or AUDIT.md narrative.
---

# Worktree Is Ground Truth

The only authoritative source for what exists in cot-suite is the actual filesystem state at the current HEAD: `git ls-files`, `git status --porcelain`, and the contents of files read directly. Prior conversation, briefing documents, AUDIT.md narrative sections, prior session reports, and memory of past work are commentary on the worktree — not authoritative.

This skill exists because the cot-suite project has been hit at least once by a context-handoff briefing that overclaimed reproduction status by an order of magnitude (e.g., "integer-exact ±0.00pp on 4/4 cross-validation models" when the repo contained 1-model N=100 reproduction with delta 3.7-8.1pp). The overclaim was caught by reading the actual worktree. Going forward, every status claim is verified against the worktree first.

## Pre-Summary Checks (Mandatory)

Before producing any "status of X" summary, run these checks and include the results in the response:

1. `git log --oneline -10` — report the actual recent commits, not what you remember being committed.
2. `git status --porcelain` — report whether the tree is clean. If there is uncommitted work, list it; uncommitted work changes the status answer.
3. For any methodology-status claim, read the actual script in `scripts/validate_*.py` to verify:
   - The exact model identifier(s) it iterates over.
   - The exact N (sample count) it uses.
   - The exact subtasks or datasets it covers.
   - Do not infer these from docs/. Read the script.
4. For any reproduction-status claim, read the actual output JSONL:
   - First 3 non-error rows to verify what's there.
   - Row count + error-row count via `wc -l` and a quick filter.
   - If the JSONL contains only error rows, the reproduction did not happen — regardless of what any prose summary in docs/ claims.

## No-Memory-Reliance Rule

If a previous conversation, a prior briefing, or any other non-filesystem source claims a result that you cannot find in `git ls-files`, in committed JSONLs, or in worktree files, you must explicitly flag this in your response:

> "Claimed by <source> but not verifiable from the current worktree."

Do not paper over the gap. Do not invent plausible cells, plausible numbers, or plausible model identifiers to fill it. Do not assume the work exists "somewhere" or "in an uncommitted branch" without checking — check explicitly with `git branch -a` and `git stash list`.

## Reconciliation Protocol

When the user provides a briefing, handoff document, or prior-session report that includes claims about repo state, your first action is to verify those claims against the worktree. Before responding to the user's actual question, produce a Reconciliation section that lists:

- Each claim from the briefing about repo state (file existence, run completion, numerical results).
- The corresponding worktree fact (exists / doesn't exist / partial / different value).
- Deltas between briefing and worktree.

Only proceed to the user's actual question after the Reconciliation section is complete. If there are material deltas, ask the user how to proceed before continuing.

## No-Confabulation Rule

If you cannot find the artifact that a claim refers to, do not generate plausible content to fill the gap. The correct response is explicit non-existence:

> "Artifact for claim <X> not found in worktree. Searched: <list of paths and search tokens>. Result: 0 hits."

This applies to model identifiers ("the claim names claude-3.5-haiku; no claude-3.5-haiku run exists in validation/"), to sample sizes ("the claim cites n=25,194; no JSONL of that size exists in the repo"), and to specific numerical cells ("the claim cites 13.65% on gpt-4o-mini; the actual gpt-4o-mini cell in validation/b4_arcuschin_raw.jsonl is 9.30%").

## What "Ground Truth" Means in Practice

- `git ls-files` + file contents = ground truth for what cot-suite contains.
- Committed JSONLs in `validation/` and `benchmarks/results/` = ground truth for what cot-suite has measured.
- Scripts in `scripts/` = ground truth for how those measurements were produced.
- AUDIT.md = annotated narrative *about* ground truth; useful as a guide but not authoritative on its own.
- docs/ = positioning and explanation; never authoritative on status.
- Prior conversation, briefings, prior strategist memory = useful context, never authoritative.

When the narrative and the worktree disagree, the worktree wins. Always.
