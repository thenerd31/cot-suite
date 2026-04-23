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

from cotmon.autoraters.legibility_coverage import LegibilityCoveragePrompt

AUTORATER_MODEL = "claude-haiku-4-5"
GPQA_HF_PATH = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"  # 198 expert-curated questions (gated — needs HF_TOKEN)
# Raised 1024 → 2048 on 2026-04-21 — two Stage 1 correct-answer trajectories
# (Q143 + one other) produced JSON-truncated Haiku responses when the
# justification ran long. 2048 empirically covers the observed p99
# justification length (~1600 chars / ~400 tokens) with margin. If a future
# run sees this threshold hit again, bump to 4096.
MAX_AUTORATER_TOKENS = 2048


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
    autorater_raw_response: str | None
    parse_error: str | None
    # Fix #1 2026-04-21: true when the Modal call was skipped due to
    # >5-min cap or exhausted retry budget. When True, generation fields
    # (raw_cot, final_answer, thinking_tokens, etc.) are placeholders.
    inference_timeout: bool = False
    inference_retry_count: int = 0


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


def qwen_generate_fn(*, stub: bool, modal_app: str | None = None):
    """Return an async `(question, cfg) -> dict` that dispatches to Modal or stub.

    Args:
        stub: if True, return a fake deterministic engine (no network).
        modal_app: name of the already-deployed Modal app whose ``Qwen3Server``
            class to target (e.g. ``"cotdiv-qwen3-14b"``, ``"cotdiv-qwen3-8b"``).
            Required when ``stub=False``; ignored in stub mode.
    """
    if stub:
        stub_engine = StubQwen()

        async def _stub(question: str, cfg: GenerationConfig) -> dict:
            return await stub_engine.generate_remote(
                question=question,
                **asdict(cfg),
            )

        return _stub

    if not modal_app:
        raise ValueError(
            "qwen_generate_fn(stub=False) requires a modal_app name (e.g. "
            "'cotdiv-qwen3-14b'). None provided.",
        )

    # Lazy import so --dry-run doesn't require modal installed in the call chain.
    import modal
    import modal.exception

    # Look up the class in the already-deployed app rather than re-defining it.
    # Requires a prior `modal deploy scripts/modal_qwen3_<size>.py` — the app
    # name is parameterized so the same driver runs against 8B / 14B / 32B.
    qwen3_server_cls = modal.Cls.from_name(modal_app, "Qwen3Server")
    server = qwen3_server_cls()

    # Fix #1 2026-04-21: wrap .remote.aio() in (retry + 5-min cap).
    # Modal's default client uses OUTPUTS_TIMEOUT=55s per internal poll
    # and retries via grpclib; transient network hiccups still surface as
    # modal.exception.ConnectionError ("Deadline exceeded"). First full-run
    # crashed at Q82 on exactly this. We retry on ConnectionError +
    # asyncio.TimeoutError up to `retry_attempts` times with exponential
    # backoff, and cap total per-question wall at `max_per_question_seconds`.
    retry_attempts = 3
    max_per_question_seconds = 300.0
    retriable: tuple[type[BaseException], ...] = (
        modal.exception.ConnectionError,
        asyncio.TimeoutError,
        ConnectionError,
    )

    async def _one_call(question: str, cfg: GenerationConfig) -> dict:
        return await server.generate.remote.aio(
            question=question,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    async def _real(question: str, cfg: GenerationConfig) -> dict:
        deadline = time.monotonic() + max_per_question_seconds
        last_exc: BaseException | None = None
        for attempt in range(1, retry_attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _InferenceTimeoutError(
                    f"exceeded {max_per_question_seconds:.0f}s per-question cap "
                    f"after {attempt - 1} attempts; last error: {last_exc!r}",
                    attempts=attempt - 1,
                )
            try:
                result = await asyncio.wait_for(_one_call(question, cfg), timeout=remaining)
                result["_retry_count"] = attempt - 1
                return result
            except retriable as exc:
                last_exc = exc
                backoff = min(2 ** (attempt - 1) * 2.0, 30.0)
                print(
                    f"  [warn] Modal call failed (attempt {attempt}/{retry_attempts}): "
                    f"{type(exc).__name__}: {exc}; retrying in {backoff:.1f}s",
                    file=sys.stderr,
                )
                try:
                    await asyncio.wait_for(
                        asyncio.sleep(backoff),
                        timeout=max(0.0, deadline - time.monotonic()),
                    )
                except TimeoutError:
                    raise _InferenceTimeoutError(
                        f"5-min cap hit while backing off after attempt {attempt}; "
                        f"last error: {last_exc!r}",
                        attempts=attempt,
                    ) from last_exc
        raise _InferenceTimeoutError(
            f"exhausted {retry_attempts} retries; last error: {last_exc!r}",
            attempts=retry_attempts,
        )

    return _real


class _InferenceTimeoutError(Exception):
    """Raised when the Modal inference call exceeds the per-question time cap
    or retry budget. Caller should catch and emit an inference_timeout=True row."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def autorater_fn(*, stub: bool):
    """Return an async `(rendered_prompt) -> (raw, leg, cov, justification, parse_error)`.

    Always returns the raw autorater body first, even when parse fails, so a
    failed row preserves the evidence needed to diagnose what Haiku did wrong.
    Fix #2 2026-04-21.
    """
    prompt = LegibilityCoveragePrompt.load()

    def _parse_safely(
        raw: str,
    ) -> tuple[str, int | None, int | None, str | None, str | None]:
        try:
            leg, cov, justification = prompt.parse(raw)
        except Exception as exc:
            return raw, None, None, None, f"{type(exc).__name__}: {exc}"
        return raw, leg, cov, justification, None

    if stub:
        stub_client = StubAutorater()

        async def _stub(
            rendered: str,
        ) -> tuple[str, int | None, int | None, str | None, str | None]:
            raw = await stub_client.complete(rendered)
            return _parse_safely(raw)

        return _stub

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()

    async def _real(
        rendered: str,
    ) -> tuple[str, int | None, int | None, str | None, str | None]:
        response = await client.messages.create(
            model=AUTORATER_MODEL,
            max_tokens=MAX_AUTORATER_TOKENS,
            messages=[{"role": "user", "content": rendered}],
        )
        raw = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        return _parse_safely(raw)

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

    # Fix #1 2026-04-21: inference path may time out or exhaust retries.
    # On timeout, emit an inference_timeout row without attempting the
    # autorater (no CoT to rate).
    try:
        gen = await qwen(mcq_prompt, cfg)
    except _InferenceTimeoutError as exc:
        print(
            f"  [timeout] {sample.question_id}: {exc}",
            file=sys.stderr,
        )
        return RunRow(
            question_id=sample.question_id,
            question_text=sample.question_text,
            raw_cot="",
            final_answer="",
            correct_answer=sample.correct_answer,
            is_correct=False,
            autorater_legibility=None,
            autorater_coverage=None,
            autorater_justification=None,
            thinking_tokens=0,
            latency_seconds=0.0,
            raw_model_content="",
            completion_tokens=0,
            autorater_raw_response=None,
            parse_error=f"inference_timeout: {exc}",
            inference_timeout=True,
            inference_retry_count=exc.attempts,
        )

    extracted = extract_answer_letter(gen["content"]) or extract_answer_letter(gen["raw_text"])
    is_correct = extracted == sample.correct_answer

    leg: int | None = None
    cov: int | None = None
    justification: str | None = None
    autorater_raw_response: str | None = None
    parse_error: str | None = None

    if is_correct:
        rendered = autorater_prompt.render(
            question=mcq_prompt,
            explanation=gen["reasoning"],
            answer=gen["content"],
        )
        # Fix #2 2026-04-21: autorater returns raw body regardless of parse
        # outcome, so parse failures don't lose the evidence we need to
        # diagnose what Haiku actually returned.
        try:
            autorater_raw_response, leg, cov, justification, parse_error = await autorater(
                rendered,
            )
        except Exception as exc:
            # Only reached if the Anthropic network call itself failed,
            # not a schema parse error (those land in parse_error below
            # with the raw body captured).
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
        autorater_raw_response=autorater_raw_response,
        parse_error=parse_error,
        inference_timeout=False,
        inference_retry_count=gen.get("_retry_count", 0),
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


def _recover_trailing_partial_write(results_path: Path) -> int:
    """Trim a malformed trailing line on results.jsonl, return rows dropped.

    A Modal container SIGKILL, OOM, or host crash can leave a half-written
    final line. Detecting this at resume time is cheap: try json.loads on
    the last non-empty line. If it raises, rewrite the file without that
    line and log it. Well-formed lines earlier in the file are untouched.
    """
    if not results_path.exists():
        return 0
    with results_path.open() as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]
    if not lines:
        return 0
    try:
        json.loads(lines[-1])
        return 0
    except json.JSONDecodeError:
        sys.stderr.write(
            f"  [resume] last line of {results_path} is not well-formed JSON "
            f"(likely partial write from a crashed container); dropping it.\n",
        )
        results_path.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""))
        return 1


def _validate_resume(
    results_path: Path,
    samples: list[Sample],
    start_from: int,
) -> list[RunRow]:
    """Read the existing results.jsonl and validate it matches samples[:start_from].

    On any mismatch, raise ValueError rather than silently append. This is
    Fix #3's row-count + question_id invariant check.
    """
    if not results_path.exists():
        raise ValueError(
            f"--start-from {start_from} specified but results.jsonl does not "
            f"exist at {results_path}",
        )
    _recover_trailing_partial_write(results_path)
    existing: list[dict] = []
    with results_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                existing.append(json.loads(line))
    if len(existing) != start_from:
        raise ValueError(
            f"--start-from {start_from} but results.jsonl has "
            f"{len(existing)} rows. Refusing to resume with ambiguous boundary.",
        )
    for i, row in enumerate(existing):
        if row["question_id"] != samples[i].question_id:
            raise ValueError(
                f"question_id mismatch at index {i}: "
                f"results.jsonl has {row['question_id']!r}, "
                f"current iteration would produce {samples[i].question_id!r}. "
                "Dataset order drifted; cannot safely append.",
            )
    # Round-trip through RunRow so aggregate() sees the same shape on both
    # paths. Ignore extra keys; tolerate pre-Fix-#2 rows that carry
    # `autorater_raw` instead of `autorater_raw_response` by renaming on
    # load.
    field_names = {f.name for f in RunRow.__dataclass_fields__.values()}
    preserved: list[RunRow] = []
    for row in existing:
        if "autorater_raw" in row and "autorater_raw_response" not in row:
            row["autorater_raw_response"] = row.pop("autorater_raw")
        filtered = {k: v for k, v in row.items() if k in field_names}
        # Supply defaults for Fix #1/#2 fields added after these rows were
        # written — pre-fix rows are treated as: not timed out, 0 retries.
        filtered.setdefault("inference_timeout", False)
        filtered.setdefault("inference_retry_count", 0)
        filtered.setdefault("autorater_raw_response", None)
        preserved.append(RunRow(**filtered))
    return preserved


async def run_pipeline(
    *,
    stub: bool,
    limit: int | None,
    output_dir: Path,
    cfg: GenerationConfig,
    modal_app: str | None = None,
    start_from: int = 0,
    load_real_dataset: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Dataset load axis is independent of inference stub axis:
    #   --dry-run: both stubbed (original behavior)
    #   --stub-inference-only: real dataset, stub inference (Fix #3 validation)
    #   neither: both real (production runs)
    samples = load_gpqa_diamond(
        limit=limit,
        stub=stub and not load_real_dataset,
    )

    qwen = qwen_generate_fn(stub=stub, modal_app=modal_app)
    autorater = autorater_fn(stub=stub)
    autorater_prompt = LegibilityCoveragePrompt.load()

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    # Fix #3 2026-04-21: resume path.
    preserved_rows: list[RunRow] = []
    if start_from > 0:
        preserved_rows = _validate_resume(results_path, samples, start_from)
        print(
            f"  [resume] validated {len(preserved_rows)} preserved rows; "
            f"appending from index {start_from}",
            file=sys.stderr,
        )
    open_mode = "a" if start_from > 0 else "w"

    rows: list[RunRow] = list(preserved_rows)
    first_client_wall: float | None = None
    first_server_wall: float | None = None
    t0 = time.monotonic()
    with results_path.open(open_mode) as fh:
        for i in range(start_from, len(samples)):
            sample = samples[i]
            call_t0 = time.monotonic()
            row = await run_one(
                sample,
                cfg,
                qwen=qwen,
                autorater=autorater,
                autorater_prompt=autorater_prompt,
            )
            if i == start_from and not stub:
                first_client_wall = time.monotonic() - call_t0
                first_server_wall = row.latency_seconds
            rows.append(row)
            fh.write(json.dumps(asdict(row)) + "\n")
            fh.flush()
            print(
                f"  [{i + 1}/{len(samples)}] {sample.question_id} "
                f"correct={row.is_correct} "
                f"leg={row.autorater_legibility} cov={row.autorater_coverage} "
                f"latency={row.latency_seconds:.2f}s "
                f"thinking_tokens={row.thinking_tokens}"
                f"{' [timeout]' if row.inference_timeout else ''}",
                file=sys.stderr,
            )
    total_wall = time.monotonic() - t0

    summary = aggregate(rows)
    summary["wall_clock_seconds"] = total_wall
    summary["autorater_prompt_sha256"] = autorater_prompt.sha256
    summary["autorater_model"] = "stub" if stub else AUTORATER_MODEL
    summary["stub_mode"] = stub
    summary["generation_config"] = asdict(cfg)
    if first_client_wall is not None and first_server_wall is not None:
        summary["first_call_client_wall_s"] = first_client_wall
        summary["first_call_server_wall_s"] = first_server_wall
        summary["cold_start_approx_s"] = max(0.0, first_client_wall - first_server_wall)
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
        "--stub-inference-only",
        action="store_true",
        help=(
            "Stub Qwen + autorater but LOAD THE REAL GPQA dataset. Requires "
            "HF_TOKEN. Used for validating --start-from resume semantics "
            "against the real dataset's question_id ordering without GPU spend."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N questions (for real 5-question dry-run).",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help=(
            "Resume from question index N. Validates results.jsonl has "
            "exactly N rows with matching question_ids before appending. "
            "Hard-errors on mismatch rather than silently overlapping."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/qwen3_14b_gpqa_diamond"),
    )
    parser.add_argument(
        "--modal-app",
        type=str,
        required=True,
        help=(
            "Name of the deployed Modal app to target — e.g. "
            "'cotdiv-qwen3-14b', 'cotdiv-qwen3-8b'. Required even in "
            "stub modes; pass any string (e.g. 'stub') for --dry-run."
        ),
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

    fully_stubbed = args.dry_run and not args.stub_inference_only
    stub_inference = args.dry_run or args.stub_inference_only

    # Preflight: verify every key we're about to spend against. --dry-run
    # skips preflight (offline smoke test). Otherwise fail loudly up-front
    # rather than mid-run after the first autorater / modal / HF call.
    if not fully_stubbed:
        from cotmon.verify_keys import require_keys
        providers = []
        if not stub_inference:
            providers.append("anthropic")  # autorater
            providers.append("modal")      # vLLM inference
        providers.append("huggingface")    # gated GPQA-Diamond dataset
        require_keys(providers)

    summary = asyncio.run(
        run_pipeline(
            stub=stub_inference,
            load_real_dataset=not fully_stubbed,
            limit=args.limit,
            output_dir=args.output_dir,
            cfg=cfg,
            modal_app=args.modal_app,
            start_from=args.start_from,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
