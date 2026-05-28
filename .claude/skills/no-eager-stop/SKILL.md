---
name: no-eager-stop
description: Use this skill on any multi-part task, any task with an explicit "report on the following sections:" structure, any audit task, any reconciliation task, or any task requesting "everything" or "all" of some category. Prevents early-stopping at "good enough" before all requested items are completed.
---

# No Eager Stop

Multi-part tasks must be completed in full, not at "good enough." Audit and reconciliation tasks especially require completeness: partial reports leave dangling questions that the user cannot easily detect, and dangling questions in cot-suite reproduction work compound into overclaimed launch numbers.

## Checklist-Match Rule

Before ending any response to a multi-part task, compare the response against the original request's explicit structure:

- Did the request ask for sections 1-N? Verify all N are present.
- Did the request ask for "every X" or "all Y"? Verify every X is covered, or explicitly state the subset and the reason.
- Did the request ask for specific command outputs? Verify each was run and its output included.

If any item is missing, complete it before stopping. Do not stop after the easy items and leave the hard ones for the user to notice.

## No Partial Reporting

If a request asks for "every numerical claim in AUDIT.md" or "all reproduction status across methodologies," do not summarize a subset. Either:

- List all items, or
- Explicitly state: "I am reporting N of M items because <reason>." The reason must be stronger than "for brevity." Acceptable reasons: "the remaining M-N items are duplicates of the first N", "the remaining items are out of scope per <criterion>". Unacceptable reasons: "the rest are similar", "for readability."

## Raw Output Preservation

For audit, reconciliation, status-report, or git-state tasks, do not paraphrase command outputs. Quote them verbatim. The user needs the raw output to verify against their own mental model of the repo. Paraphrased outputs hide bugs:

- `git status --porcelain` output: quote the exact lines, or state "(empty)" if zero lines.
- `git log --oneline -10` output: quote all 10 lines.
- `grep` results: quote the actual matched lines with paths and line numbers.
- File contents: quote the exact bytes, do not summarize.

## Completion Check

End every multi-part task with a one-line confirmation in one of two forms:

- **Completed:** `<list of sections covered>`
- **Incomplete:** `<what's missing>` — `<why>`

The completion check is required even when the task is fully complete. It exists so the user can verify at a glance that nothing was skipped.

## Anti-Pattern: Stop-Hook Repair

If a stop hook fires and forces continuation of an incomplete task, this is a failure of the no-eager-stop discipline. The repair pattern (continue, then issue corrections) is acceptable as a last-resort backstop but should not be the normal mode. Aim for first-pass completion. The stop hook is the safety net, not the workflow.

## When in Doubt, Continue

If you are uncertain whether a section is complete, err on the side of including more detail rather than less. Audit tasks especially: missing items are far worse than redundant items. If the user wanted brevity, they would have asked for brevity.
