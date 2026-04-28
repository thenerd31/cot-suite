"""Reconstruct per-question option maps for the GPQA-Diamond run.

The run-driver in ``scripts/run_qwen3_gpqa.py`` shuffles the four
options per question with ``rng = random.Random(i); rng.shuffle(opts)``
where ``i`` is the dataset row index. The shuffle is deterministic but
the option mapping is NOT preserved into ``results.jsonl`` — only
``question_text``, ``correct_answer``, and ``final_answer`` are
serialized.

This module rebuilds the same shuffle from the gated HuggingFace
dataset and exposes ``options_for(question_id)`` so the v2 normalizer
in ``cotsuite.normalize_cot`` can map judge ``cot_conclusion`` strings
(e.g. ``"Option 2 only"``, ``"0"``) to canonical option letters per
the original ordering.

Usage:

    >>> from scripts.reconstruct_options import options_for
    >>> options_for("gpqa_diamond_116")
    {'A': 'Mutant 1', 'B': 'Mutant 2', 'C': '...', 'D': '...'}

CLI:

    PYTHONPATH=. python scripts/reconstruct_options.py gpqa_diamond_116

Cached at module level after first dataset load (~1s) so repeated
calls are O(1).

Requires HF_TOKEN — gated dataset license handshake.
"""

from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from pathlib import Path

GPQA_HF_PATH = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"
GPQA_SPLIT = "train"


@lru_cache(maxsize=1)
def _load_options() -> dict[str, dict[str, str]]:
    """Load GPQA-Diamond once; return ``{question_id: {A: text, ...}}``."""
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        # Try loading .env (for local dev convenience)
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGINGFACE_HUB_TOKEN="):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k, v.strip().strip('"').strip("'"))
                    break
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError(
            "HF_TOKEN not set. GPQA-Diamond is a gated dataset. "
            "Set HF_TOKEN in env or .env to reconstruct option maps."
        )

    from datasets import load_dataset

    ds = load_dataset(GPQA_HF_PATH, GPQA_CONFIG, split=GPQA_SPLIT)
    out: dict[str, dict[str, str]] = {}
    for i, row in enumerate(ds):
        correct = row["Correct Answer"]
        incorrect = [row[f"Incorrect Answer {j}"] for j in (1, 2, 3)]
        options = [correct, *incorrect]
        # Match the run-driver's shuffle exactly.
        rng = random.Random(i)
        rng.shuffle(options)
        letter_to_text = dict(zip("ABCD", options, strict=True))
        out[f"gpqa_diamond_{i:03d}"] = letter_to_text
    return out


def options_for(question_id: str) -> dict[str, str]:
    """Return ``{A: text, B: text, C: text, D: text}`` for a question_id.

    Raises KeyError if the question_id isn't in GPQA-Diamond.
    """
    return _load_options()[question_id]


def correct_letter_for(question_id: str) -> str:
    """Return the correct letter for a question_id (looked up against options)."""
    options = options_for(question_id)
    # The dataset row's "Correct Answer" field maps to one of A-D after the shuffle;
    # we need to identify which by re-walking the dataset, OR by rebuilding the
    # same shuffle and tracking which index landed where. The deterministic shuffle
    # plus the dataset row ordering is what defines this. Easier: use the run-driver's
    # results.jsonl correct_answer field — but we don't want a circular dep. Re-scan.
    from datasets import load_dataset

    ds = load_dataset(GPQA_HF_PATH, GPQA_CONFIG, split=GPQA_SPLIT)
    i = int(question_id.split("_")[-1])
    row = ds[i]
    correct_text = row["Correct Answer"]
    for letter, text in options.items():
        if text == correct_text:
            return letter
    raise RuntimeError(f"correct answer not found in option map for {question_id}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/reconstruct_options.py <question_id>")
        print("Example: python scripts/reconstruct_options.py gpqa_diamond_116")
        return 2
    qid = sys.argv[1]
    options = options_for(qid)
    print(f"# {qid}")
    for letter, text in options.items():
        print(f"  {letter}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
