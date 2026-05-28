---
name: pause-on-precondition
description: Use this skill whenever you detect a precondition violation, an ambiguity in user instructions, an inconsistency between the user's instruction and the actual filesystem state, or any situation where executing the user's literal instruction would produce a worktree end-state that differs from what they appeared to intend. Carves out the legitimate-stop cases from the no-eager-stop discipline. Pausing for user input on a real ambiguity is correct behavior, not a failure to complete.
---

# Pause on Precondition

The `no-eager-stop` skill enforces task completeness. This skill carves out the legitimate cases where stopping is correct behavior, not a failure to complete. The two skills are complements, not opposites: `no-eager-stop` says "don't bail before all checklist items are done"; this skill says "do bail when continuing would violate a precondition the user didn't anticipate."

## When to Pause

Pause and request user input — even if the user's literal instruction says "do X" — in any of these cases:

1. **Precondition violation.** The user instructed an operation that assumes a precondition, and the precondition is false. Example: user says "extend the existing file" but the file is untracked / doesn't exist. Example: user says "modify the failing test" but the test passes. Example: user says "commit the change" but there are no changes staged.

2. **Filesystem-vs-instruction delta.** The user's instruction implies a worktree state that doesn't match reality. Example: user says "the script should already produce N=100 rows" but the actual script hardcodes N=10. Example: user says "in the existing commit history" but the commit they referenced doesn't exist.

3. **Multi-path ambiguity with material consequences.** The user's instruction is satisfiable in two or more ways that produce materially different end-states. "Materially different" means: different commit history, different file content, different downstream effects on subsequent work.

4. **Stale-context risk.** The user gave the instruction under an assumption about repo state that has since changed. Example: user planned a task assuming the previous commit added file X, but a subsequent commit moved file X.

5. **Spend-gate violation.** Executing the instruction would spend money under conditions the user might not have anticipated. Example: user said "run the validation" but the validation script's MODEL_LIST has been edited to include a new expensive model since the user last reviewed it.

## What "Pause" Looks Like

A pause is not an abort. The agent does not silently fail to do the thing. The agent does:

1. **State the precondition violation explicitly.** Quote the user's instruction. Quote the conflicting filesystem fact (with file paths, line numbers, command output). State the delta in one sentence.

2. **Present 2-4 options for resolution.** Each option produces a concrete end-state. Each option's tradeoff is named in one sentence.

3. **Stop and wait.** The user's next message is the resolution.

4. **Do not execute any option until the user picks one.** Even the user's literal original instruction is not auto-selected — because the original instruction is what triggered the precondition violation in the first place.

## Interaction with `no-eager-stop`

`no-eager-stop` says "do not stop at 'good enough' on a multi-part task." This skill says "do stop when continuing would produce a wrong end-state, even if 'good enough' would be reached."

These two coexist:

- If you've completed 4 of 5 sections of an audit and the 5th is just tedious to finish, `no-eager-stop` overrides — finish it.
- If you're 4 of 5 sections into an audit and the 5th section requires running a command that would spend $50 the user didn't authorize, `pause-on-precondition` overrides — stop and ask.

When in doubt: stopping for a legitimate precondition violation is always safer than executing under ambiguity. The user can always say "go ahead." The user cannot un-execute a commit that landed under the wrong message, or un-spend money that was charged.

## Interaction with Stop Hooks

If a stop hook fires after a legitimate `pause-on-precondition` pause, the correct response is to re-present the options to the user, not to override the pause and execute. A stop hook is a heuristic that the agent stopped too early on a checklist task; it is not authoritative evidence that the agent should ignore a precondition violation.

If the stop hook's instruction conflicts with this skill, this skill wins. State the conflict explicitly in the response so the user can see what happened.

## Anti-Pattern

The failure case this skill prevents: agent correctly detects an ambiguity, correctly presents options to the user, then a stop hook fires and the agent over-rides its own correct instinct, picks an option silently, and executes. This produces a worktree end-state that doesn't match what the user would have chosen if asked. The audit trail then contains a commit with a slightly-wrong message, or a spend the user didn't authorize, or a file edit that touched more than intended.

If this happens, the agent should:
1. Report the over-ride explicitly: "Stop hook fired and pushed me past a precondition check I had correctly raised."
2. Identify the resulting end-state delta.
3. Ask the user whether to roll back or proceed.

Never silently absorb a stop-hook-driven over-ride as if it were the correct path.
