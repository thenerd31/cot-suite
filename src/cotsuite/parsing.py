r"""Canonical MCQ answer-letter extraction for cot-suite.

Used by both the multi-family scaling driver
(``scripts/run_qwen3_gpqa.py``) and the v0.1 Inspect AI scorer
(``cotsuite.inspect.scorers.post_hoc_rationalization``).
A single canonical implementation here means future bugfixes touch one
site, not two.

# Layered anchored extraction

Layers in priority order (most specific → least specific). Last match
within each layer wins. Fall through to the next layer if no match.
Return ``""`` (unscorable) if nothing matches at any layer.

1. ``\\boxed{X}`` — last match
2. ``Final Answer: X`` / ``Final answer: X`` (mandatory colon/dash)
3. Generic ``Answer: X`` (mandatory colon/dash)
4. Bare-letter line fallback, scoped to the last 500 chars only
5. ``""`` (unscorable)

# Why layered

A model emitting ``\\boxed{D}`` should not be overridden by a spurious
mid-prose ``Answer: A`` match. Confidence layering prevents that.
Within each layer, last-match wins because the model's last commitment
is what counts — early "leaning toward A" prose should not override a
final "Answer: D".

# Why mandatory colon

The previous parser used ``[:\\-]?`` (optional) and matched prose like
``the answer choices`` (capturing 'c' from "**c**hoices") and
``Final Answer\\n\\nAnswer: D`` (capturing 'A' from the second
"Answer" word). Mandatory colon-or-dash anchors the match to a formal
commitment.

# Why scoped line fallback

The bare-letter fallback ``^[A-D]$`` is broad enough to fire on any
single-letter line anywhere in the output (option restatements,
sub-bullet labels, etc). Scoping to the last 500 chars limits it to
the model's actual closing region.

See ``tests/test_parser_bug_diagnosis.py`` for the failure-mode
reproductions and post-fix expectations this implementation satisfies.
"""

from __future__ import annotations

import re

# Filler that can appear around the captured letter without changing
# meaning: spaces, tabs, asterisks (markdown bold), and the special
# latex `\text{...}` wrapper that Qwen3-Thinking emits inside `\boxed{}`.
# `[ \t\*]` excludes newlines deliberately — the colon must be on the
# same line as "Answer", or the regex over-eats across paragraph breaks
# and recreates the original parser bug.
_NW = r"[ \t\*]*"  # non-newline whitespace / markdown bold

# `\boxed{X}` plus the `\boxed{\text{X}}` latex variant Qwen3-Thinking
# emits. Both forms allowed, with optional markdown bold around the letter.
_BOXED_RE = re.compile(
    r"\\boxed\{\s*(?:\\text\{)?\s*"
    r"\*{0,2}([A-Da-d])\*{0,2}"
    r"\s*(?:\})?\s*\}",
)

# "Final Answer: X" anchored. Allows markdown headers like
# `### **Final Answer:** **D**` because `[^\n]*?` between "Final" and
# ":" tolerates intervening markdown but `[^\n]` keeps it on one line.
_FINAL_ANSWER_RE = re.compile(
    r"Final[ \t\*]+Answer\b[^\n]*?[:\-]" + _NW + r"\(?([A-Da-d])\)?",
    re.IGNORECASE,
)

# Generic "Answer: X" with MANDATORY colon-or-dash. `[ \t\*]*` allows
# markdown bold (`**Answer:** X`) but not newlines (which was the
# original bug — `\s*` ate `\n\n` and captured 'A' from the next
# "Answer" word).
_GENERIC_ANSWER_RE = re.compile(
    r"\bAnswer\b" + _NW + r"[:\-]" + _NW + r"\(?([A-Da-d])\)?",
    re.IGNORECASE,
)

# Bare-letter line: a line containing only "A" / "(B)" / "**C**" etc.
_LINE_RE = re.compile(
    r"^[ \t]*\*{0,2}\(?([A-Da-d])\)?\*{0,2}[ \t]*$",
    re.MULTILINE,
)

_LINE_FALLBACK_WINDOW = 500


def extract_answer_letter(content: str) -> str:
    """Extract the final MCQ letter (A-D) from a model's output string.

    Returns the upper-case letter on match, or the empty string ``""``
    if no formal commitment can be found. ``""`` is the honest unscorable
    signal — downstream code should treat it as NaN, not as a spurious
    label.

    Args:
        content: The model's full response text. Typically
            ``raw_model_content`` from a benchmark JSONL row.

    Returns:
        Upper-case letter ``"A"``..``"D"``, or ``""`` if no anchored
        commitment is present.
    """
    if not content:
        return ""

    # Layers 1-3: priority-ordered, last-match within each.
    for pattern in (_BOXED_RE, _FINAL_ANSWER_RE, _GENERIC_ANSWER_RE):
        matches = list(pattern.finditer(content))
        if matches:
            return matches[-1].group(1).upper()

    # Layer 4: bare-letter line, scoped to the last 500 chars only.
    tail = content[-_LINE_FALLBACK_WINDOW:]
    matches = list(_LINE_RE.finditer(tail))
    if matches:
        return matches[-1].group(1).upper()

    return ""
