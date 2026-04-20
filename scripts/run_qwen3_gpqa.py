"""Drive Qwen3-14B across GPQA-Diamond, filter to correct answers, autorate.

# Experimental setup grounded in public sources (pinned 2026-04-20)
# =================================================================
#
# PAPER (Emmons & Zimmermann et al., arXiv 2510.23966):
#   - Tested `Qwen3 235B` (Table 1, §5.1). NOT Qwen3-14B.
#   - Table 1 reports POOLED legibility/coverage across HLE, GPQA-Diamond,
#     ARC-AGI, AIME: 97.33% ± 0.18% / 95.27% ± 0.39%. No per-dataset
#     breakdown in the paper (verified via arXiv HTML fetch).
#   - Generation hyperparameters (temperature, top_p, max tokens, thinking
#     toggle) are NOT documented in §5.1.
#
# OUR STAGE 1 CHOICES (pragmatic — paper does not prescribe):
#   - Model: Qwen3-14B (not paper variant; smaller open-weights checkpoint
#     for pipeline validation on our $100 Modal cap).
#   - Inference settings: Qwen3 HF model card defaults for thinking mode
#     (temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=32768,
#     enable_thinking=True). Source: huggingface.co/Qwen/Qwen3-14B
#     model card "Usage" section, fetched 2026-04-20.
#   - seed=0 for reproducibility (paper does not specify; we lock for
#     our own re-runs).
#   - Autorater: Claude Haiku 4.5 (NOT Gemini 2.5 Pro as in the paper) —
#     matches our smoke test, enables apples-to-apples comparison against
#     Stage 2. Cross-rater validation vs paper's Gemini-2.5-Pro deferred
#     to Stage 2.
#   - Single-shot autorater (n=1), not paper's n=5. Justified by the
#     2026-04-20 variance runs: 3/3 convergence on the 4/4 case,
#     cov-spread=0 on the boundary case. See docs/autorater-notes.md.
#
# FRAMING: "applies 2510.23966 methodology to a smaller open-weights
# model the paper did not evaluate — pipeline validation + scale-down
# data point." This is NOT a reproduction of Table 1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from cotdiv.autoraters.legibility_coverage import LegibilityCoveragePrompt

AUTORATER_MODEL = "claude-haiku-4-5"
GPQA_HF_PATH = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"  # 198 expert-curated questions (gated — needs HF_TOKEN)
MAX_AUTORATER_TOKENS = 1024


@dataclass
class GenerationConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    max_tokens: int = 32768
    seed: int = 0


@dataclass
class Sample:
    question_id: str
    question_text: str
    correct_answer: str  # MCQ letter 'A'-'D'
    option_letter_to_text: dict[str, str]


@dataclass
class RunRow:
    """One row in results.jsonl. Written for every sample, correct or not."""

    question_id: str
    question_text: str
    raw_cot: str
    final_answer: str
    correct_answer: str
    is_correct: bool
    autorater_legibility: int | None
    autorater_coverage: int | None
    autorater_justification: str | None
    thinking_tokens: int
    latency_seconds: float
    # Bookkeeping
    raw_model_content: str
    completion_tokens: int
    autorater_raw: str | None
    parse_error: str | None


# =============================================================================
# Dataset loading
# =============================================================================


def load_gpqa_diamond(limit: int | None, *, stub: bool = False) -> list[Sample]:
    """Load GPQA-Diamond; shuffle deterministically; optionally truncate.

    In ``stub=True`` mode returns hardcoded in-file fixtures — no network,
    no HF_TOKEN, no dataset license handshake. Five plausible GPQA-Diamond-
    shaped questions, enough to exercise the full pipeline wiring.
    """
    if stub:
        samples = _fixture_gpqa_samples()
    else:
        from datasets import load_dataset

        ds = load_dataset(GPQA_HF_PATH, GPQA_CONFIG, split="train")
        samples = []
        for i, row in enumerate(ds):
            correct = row["Correct Answer"]
            incorrect = [row[f"Incorrect Answer {j}"] for j in (1, 2, 3)]
            options = [correct, *incorrect]
            rng = random.Random(i)  # deterministic permutation per-question
            rng.shuffle(options)
            letter_to_text = dict(zip("ABCD", options, strict=True))
            correct_letter = next(
                letter for letter, text in letter_to_text.items() if text == correct
            )
            samples.append(
                Sample(
                    question_id=f"gpqa_diamond_{i:03d}",
                    question_text=row["Question"],
                    correct_answer=correct_letter,
                    option_letter_to_text=letter_to_text,
                ),
            )
    if limit is not None:
        samples = samples[:limit]
    return samples


def _fixture_gpqa_samples() -> list[Sample]:
    """Five fixture samples for --dry-run. Not derived from the real dataset;
    structurally shaped like GPQA-Diamond MCQs so the pipeline code path is
    exercised identically. DO NOT use for any published number."""
    raw = [
        (
            "A photon with wavelength 500 nm has what energy in eV? "
            "(Use h=6.626e-34 J·s, c=3e8 m/s, 1 eV=1.602e-19 J.)",
            "2.48 eV",
            ["1.24 eV", "3.72 eV", "4.96 eV"],
        ),
        (
            "In a CRISPR-Cas9 knock-in experiment, which repair pathway "
            "mediates precise insertion of a donor template?",
            "Homology-directed repair",
            [
                "Non-homologous end joining",
                "Microhomology-mediated end joining",
                "Base excision repair",
            ],
        ),
        (
            "For a Carnot engine operating between 600 K and 300 K, "
            "what is the maximum theoretical efficiency?",
            "0.50",
            ["0.33", "0.67", "1.00"],
        ),
        (
            "Which quantum number determines the magnetic projection of orbital "
            "angular momentum for an electron in a hydrogen atom?",
            "m_l",
            ["n", "l", "m_s"],
        ),
        (
            "In population genetics, the Hardy-Weinberg equilibrium is violated by which of the following?",
            "Non-random mating",
            ["Large population size", "No migration", "No mutation"],
        ),
    ]
    samples = []
    for i, (q, correct, incorrect) in enumerate(raw):
        options = [correct, *incorrect]
        rng = random.Random(i)
        rng.shuffle(options)
        letter_to_text = dict(zip("ABCD", options, strict=True))
        correct_letter = next(letter for letter, text in letter_to_text.items() if text == correct)
        samples.append(
            Sample(
                question_id=f"fixture_gpqa_{i:03d}",
                question_text=q,
                correct_answer=correct_letter,
                option_letter_to_text=letter_to_text,
            ),
        )
    return samples


def format_mcq_prompt(sample: Sample) -> str:
    """Standard four-letter MCQ prompt, deterministic option order."""
    lines = [sample.question_text, ""]
    for letter in "ABCD":
        lines.append(f"({letter}) {sample.option_letter_to_text[letter]}")
    lines.append("")
    lines.append(
        "Think step-by-step, then state your final answer on the last line as 'Answer: <letter>'."
    )
    return "\n".join(lines)


_ANSWER_RE = re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*\(?([A-Da-d])\)?", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Da-d])\s*\}")
_LINE_RE = re.compile(r"^\s*\(?([A-Da-d])\)?\s*$", re.MULTILINE)


def extract_answer_letter(content: str) -> str:
    """Parse final MCQ letter from the model's content. Returns '' on failure."""
    for pattern in (_BOXED_RE, _ANSWER_RE, _LINE_RE):
        match = pattern.search(content)
        if match:
            return match.group(1).upper()
    return ""


# =============================================================================
# Stubs for --dry-run (zero outbound)
# =============================================================================


class StubQwen:
    """Deterministic fake Qwen3-14B. Produces plausible CoT + correct answer."""

    async def generate_remote(self, **kwargs) -> dict:
        question = kwargs.get("question", "")
        letter = _stub_pick_letter(question)
        reasoning = (
            "Stub reasoning. Considering the question and the provided options, "
            "the most consistent answer under standard interpretation is "
            f"option ({letter}). Brief justification: stub pathway used for "
            "pipeline dry-run; no real inference performed."
        )
        content = f"After analysis, the answer is ({letter})."
        return {
            "reasoning": reasoning,
            "content": content,
            "raw_text": f"<think>\n{reasoning}\n</think>\n\n{content}",
            "prompt_tokens": 420,
            "completion_tokens": 96,
            "thinking_tokens": 72,
            "finish_reason": "stop",
            "wall_clock_s": 0.01,
            "model_id": "stub/qwen3-14b",
            "vllm_version": "stub",
            "sampling": {
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "top_k": kwargs.get("top_k"),
                "min_p": kwargs.get("min_p"),
                "max_tokens": kwargs.get("max_tokens"),
                "seed": kwargs.get("seed"),
            },
        }


class StubAutorater:
    """Deterministic fake Haiku autorater — returns 4/4 and a one-line justification."""

    async def complete(self, prompt: str) -> str:
        return json.dumps(
            {
                "justification": "Stub justification — dry-run path, no API call made.",
                "legibility_score": 4,
                "coverage_score": 4,
            },
        )


def _stub_pick_letter(question: str) -> str:
    # deterministic per-question: pick the same "correct" letter our dataset
    # shuffle would place at index 0 for that question's seed. For --dry-run
    # we don't need ground-truth alignment — we just need a letter.
    digest = sum(ord(c) for c in (question[:100] or "x"))
    return string.ascii_uppercase[digest % 4]


# =============================================================================
# Real clients
# =============================================================================


def qwen_generate_fn(*, stub: bool):
    """Return an async `(question, cfg) -> dict` that dispatches to Modal or stub."""
    if stub:
        stub_engine = StubQwen()

        async def _stub(question: str, cfg: GenerationConfig) -> dict:
            return await stub_engine.generate_remote(
                question=question,
                **asdict(cfg),
            )

        return _stub

    # Lazy import so --dry-run doesn't require modal installed in the call chain.
    from scripts.modal_qwen3_14b import Qwen3Server

    server = Qwen3Server()

    async def _real(question: str, cfg: GenerationConfig) -> dict:
        # Modal .remote.aio returns an awaitable.
        return await server.generate.remote.aio(
            question=question,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    return _real


def autorater_fn(*, stub: bool):
    """Return an async `(rendered_prompt) -> (leg, cov, justification, raw)`."""
    prompt = LegibilityCoveragePrompt.load()

    if stub:
        stub_client = StubAutorater()

        async def _stub(rendered: str) -> tuple[int, int, str, str]:
            raw = await stub_client.complete(rendered)
            leg, cov, justification = prompt.parse(raw)
            return leg, cov, justification, raw

        return _stub

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()

    async def _real(rendered: str) -> tuple[int, int, str, str]:
        response = await client.messages.create(
            model=AUTORATER_MODEL,
            max_tokens=MAX_AUTORATER_TOKENS,
            messages=[{"role": "user", "content": rendered}],
        )
        raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        leg, cov, justification = prompt.parse(raw)
        return leg, cov, justification, raw

    return _real


# =============================================================================
# Main pipeline
# =============================================================================


async def run_one(
    sample: Sample,
    cfg: GenerationConfig,
    *,
    qwen,
    autorater,
    autorater_prompt: LegibilityCoveragePrompt,
) -> RunRow:
    mcq_prompt = format_mcq_prompt(sample)
    gen = await qwen(mcq_prompt, cfg)

    extracted = extract_answer_letter(gen["content"]) or extract_answer_letter(gen["raw_text"])
    is_correct = extracted == sample.correct_answer

    leg: int | None = None
    cov: int | None = None
    justification: str | None = None
    autorater_raw: str | None = None
    parse_error: str | None = None

    if is_correct:
        rendered = autorater_prompt.render(
            question=mcq_prompt,
            explanation=gen["reasoning"],
            answer=gen["content"],
        )
        try:
            leg, cov, justification, autorater_raw = await autorater(rendered)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"

    return RunRow(
        question_id=sample.question_id,
        question_text=sample.question_text,
        raw_cot=gen["reasoning"],
        final_answer=extracted,
        correct_answer=sample.correct_answer,
        is_correct=is_correct,
        autorater_legibility=leg,
        autorater_coverage=cov,
        autorater_justification=justification,
        thinking_tokens=gen["thinking_tokens"],
        latency_seconds=gen["wall_clock_s"],
        raw_model_content=gen["content"],
        completion_tokens=gen["completion_tokens"],
        autorater_raw=autorater_raw,
        parse_error=parse_error,
    )


def aggregate(rows: list[RunRow]) -> dict:
    n_total = len(rows)
    n_correct = sum(r.is_correct for r in rows)
    rated = [r for r in rows if r.autorater_legibility is not None]
    parse_failures = [r for r in rows if r.parse_error is not None]

    def _mean_sd(vals):
        if not vals:
            return None, None
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return statistics.fmean(vals), statistics.stdev(vals)

    leg_mean, leg_sd = _mean_sd([r.autorater_legibility for r in rated])
    cov_mean, cov_sd = _mean_sd([r.autorater_coverage for r in rated])

    total_latency = sum(r.latency_seconds for r in rows)
    total_thinking_tokens = sum(r.thinking_tokens for r in rows)
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "n_rated": len(rated),
        "n_parse_failures": len(parse_failures),
        "parse_failure_ids": [r.question_id for r in parse_failures],
        "legibility_mean": leg_mean,
        "legibility_sd": leg_sd,
        "coverage_mean": cov_mean,
        "coverage_sd": cov_sd,
        "total_inference_wall_s": total_latency,
        "total_thinking_tokens": total_thinking_tokens,
    }


async def run_pipeline(
    *,
    stub: bool,
    limit: int | None,
    output_dir: Path,
    cfg: GenerationConfig,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_gpqa_diamond(limit=limit, stub=stub)

    qwen = qwen_generate_fn(stub=stub)
    autorater = autorater_fn(stub=stub)
    autorater_prompt = LegibilityCoveragePrompt.load()

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    rows: list[RunRow] = []
    t0 = time.monotonic()
    with results_path.open("w") as fh:
        for i, sample in enumerate(samples):
            row = await run_one(
                sample,
                cfg,
                qwen=qwen,
                autorater=autorater,
                autorater_prompt=autorater_prompt,
            )
            rows.append(row)
            fh.write(json.dumps(asdict(row)) + "\n")
            fh.flush()
            print(
                f"  [{i + 1}/{len(samples)}] {sample.question_id} "
                f"correct={row.is_correct} "
                f"leg={row.autorater_legibility} cov={row.autorater_coverage} "
                f"latency={row.latency_seconds:.2f}s",
                file=sys.stderr,
            )
    total_wall = time.monotonic() - t0

    summary = aggregate(rows)
    summary["wall_clock_seconds"] = total_wall
    summary["autorater_prompt_sha256"] = autorater_prompt.sha256
    summary["autorater_model"] = "stub" if stub else AUTORATER_MODEL
    summary["stub_mode"] = stub
    summary["generation_config"] = asdict(cfg)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub both Qwen and autorater. Zero outbound. Validates pipeline wiring.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions (for real 5-question dry-run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/qwen3_14b_gpqa_diamond"),
    )
    # Inference settings — default to Qwen3 HF card recommendations.
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write("ERROR: ANTHROPIC_API_KEY not set; either --dry-run or export key.\n")
        sys.exit(2)
    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        sys.stderr.write(
            "ERROR: HF_TOKEN not set. GPQA-Diamond is a gated dataset — accept the "
            "license at https://huggingface.co/datasets/Idavidrein/gpqa then export HF_TOKEN.\n",
        )
        sys.exit(2)

    summary = asyncio.run(
        run_pipeline(
            stub=args.dry_run,
            limit=args.limit,
            output_dir=args.output_dir,
            cfg=cfg,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
