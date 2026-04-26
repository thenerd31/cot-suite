"""Re-split DeepSeek-R1-Distill outputs after Stage-3 inference.

The DeepSeek-R1-Distill chat template injects `<think>` into the
prompt, so the model only emits `</think>` (closing tag) followed by
the final answer. The original Modal sibling's `_split_thinking`
required BOTH tags and so returned ``reasoning=""``, ``content=full``
for every row. The PHR detector then errored on every correct
trajectory because reasoning was empty.

This script re-parses each row's ``raw_model_content`` using the
correct DeepSeek splitter:
    everything before `</think>`  → reasoning
    everything after `</think>`   → content
and writes the corrected row back to results.jsonl. Then re-extracts
the answer letter from the corrected ``content`` and updates
``thinking_tokens``. Idempotent — re-running on a fixed file is a
no-op.

Usage:
    python scripts/fix_deepseek_split.py \\
        --results benchmarks/results/ds_r1_distill_qwen_14b_full/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ANSWER_RE = re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*\(?([A-Da-d])\)?", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Da-d])\s*\}")
_LINE_RE = re.compile(r"^\s*\(?([A-Da-d])\)?\s*$", re.MULTILINE)


def extract_answer_letter(content: str) -> str:
    for pattern in (_BOXED_RE, _ANSWER_RE, _LINE_RE):
        match = pattern.search(content)
        if match:
            return match.group(1).upper()
    return ""


def split_deepseek(text: str) -> tuple[str, str]:
    """Split DeepSeek-R1-Distill output: reasoning before `</think>`, answer after."""
    close = "</think>"
    idx = text.find(close)
    if idx == -1:
        # Both tags missing — treat as no thinking emitted (rare).
        return "", text
    reasoning = text[:idx].strip()
    content = text[idx + len(close):].strip()
    return reasoning, content


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    fixed = 0
    for r in rows:
        text = r.get("raw_model_content", "") or ""
        reasoning, content = split_deepseek(text)
        if not reasoning:
            continue  # both tags missing or empty body — leave row as-is
        if r.get("raw_cot") == reasoning and r.get("raw_model_content") == content:
            continue  # already in fixed form (idempotent rerun)
        r["raw_cot"] = reasoning
        r["raw_model_content"] = content
        # Re-extract answer letter from the corrected content; fall back to
        # the full original text if content alone doesn't yield a letter.
        new_letter = extract_answer_letter(content) or extract_answer_letter(text)
        if new_letter:
            r["final_answer"] = new_letter
            r["is_correct"] = (new_letter == r["correct_answer"])
        # Approximate thinking_tokens by character ratio (cheap proxy);
        # exact would require re-tokenizing. For monitorability metrics we
        # only need a reasonable estimate, not exact token counts.
        if r.get("completion_tokens"):
            ratio = len(reasoning) / max(1, len(text))
            r["thinking_tokens"] = int(r["completion_tokens"] * ratio)
        fixed += 1

    print(f"fixed {fixed}/{len(rows)} rows")
    if args.dry_run:
        print("(dry-run; no write)")
        return 0
    args.results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
