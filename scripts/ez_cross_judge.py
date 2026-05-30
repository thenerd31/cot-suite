"""E-Z cross-classifier kappa validation (785 committed autorater trajectories).

Re-scores the Haiku-rated GPQA-Diamond autorater trajectories through additional
judges (gpt-4o, claude-sonnet-4-6, gemini-2.5-pro) on the byte-identical
Emmons-Zimmermann Appendix C prompt + the reconstructed ``mcq_prompt`` Haiku was
scored on, then computes pairwise quadratic-weighted Cohen's kappa (legibility +
coverage) plus model-ranking-reversal.

The run is SPLIT-FRIENDLY (``--judges``): the fast judges land first, gpt-4o
appends on its slow clock later (Tier-1 30k TPM → ~2.6 h, token-rate-bound).
Append is keyed by (model_dir, question_id); kappa recomputes over whichever
judges are present. Re-running is resumable — a judge already scored on a
trajectory is skipped.

Two headlines:
  1. Rater-substitution fidelity — Haiku-vs-Gemini kappa: does the Haiku
     substitute agree with E-Z's Gemini rater (Gemini 2.5 Pro is E-Z's
     original rater; Haiku 4.5 was cot-suite's substitute)? NOT a
     reproduction of E-Z's reported cells.
  2. Cross-classifier sensitivity — does switching judge reorder the 8-model
     ranking? (Young 2603.20172.)

mcq_prompt reconstruction: remote HF download is broken in-env (huggingface_hub
1.16 + httpx 0.28 'client closed'); we read a local curl'd gpqa_diamond.csv via
a load_dataset monkeypatch so the committed ``run_qwen3_gpqa.load_gpqa_diamond``
construction runs byte-exactly (Phase A verified, 24/24).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

from cotsuite.autoraters.legibility_coverage import LegibilityCoveragePrompt
from cotsuite.judges.kappa import cohen_kappa_quadratic, dominant_category_fraction
from cotsuite.judges.ranking_reversal import detect_ranking_reversals, ranking_reversal_summary
from cotsuite.models.clients import get_grader_client

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CSV = Path("/tmp/gpqa_diamond.csv")
CSV_URL = "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv"
OUT_DIR = ROOT / "benchmarks" / "results" / "ez_cross_judge"
SCORES_PATH = OUT_DIR / "cross_judge_scores.jsonl"
SUMMARY_PATH = OUT_DIR / "kappa_summary.json"
PROMPT_VERSION = "emmons_zimmermann_v1"
PROMPT_SHA = "ac1e0ac4044b0a64816bc3c4424e547a462e64d18cf0f5555dec7ed5b42aaa67"
MAX_TOKENS = 2048
NUM_CATEGORIES = 5  # labels 0-4; _to_labels clips to [0,K-1] so K MUST be 5 (4 would clip a real "4")
SPEND_CEILING = 60.0
PARSE_FAIL_HALT = 0.30

HAIKU = "haiku-4.5"  # committed; never re-scored
NEW_JUDGES = {
    "gpt-4o": "openai/gpt-4o",
    "sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
}
# Per-judge (input, output) USD/token (approx) + concurrency + inter-call pacing.
# gpt-4o is throttled to ~1 call / 13s to stay under the org's 30k TPM ceiling.
# gemini-2.5-pro is a thinking model: a long justification overruns 2048 and
# truncates the JSON (intermittent parse-fail), so it gets a larger cap. This is
# instrument-preserving — the cap only governs JSON completion, not the scores.
JUDGE_CFG = {
    "gpt-4o": {"price": (2.5e-6, 10e-6), "concurrency": 1, "min_interval": 13.0, "max_tokens": MAX_TOKENS},
    "sonnet-4.6": {"price": (3e-6, 15e-6), "concurrency": 4, "min_interval": 0.0, "max_tokens": MAX_TOKENS},
    "gemini-2.5-pro": {"price": (1.25e-6, 10e-6), "concurrency": 4, "min_interval": 0.0, "max_tokens": 8192},
}
SCORED_DIRS = sorted((ROOT / "benchmarks" / "results").glob("*_full"))
_SUFFIXES = ("_leg", "_cov", "_parse_fail", "_skip_reason", "_raw", "_parse_err")


class HaltError(RuntimeError):
    """Raised to stop the run on a pre-declared failure mode."""


# --------------------------------------------------------------------------
# mcq_prompt reconstruction (Phase-A-verified local-CSV path)
# --------------------------------------------------------------------------


def _ensure_csv() -> None:
    if LOCAL_CSV.exists() and LOCAL_CSV.stat().st_size > 1000:
        return
    import os

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
    subprocess.run(
        ["curl", "-fsSL", "-H", f"Authorization: Bearer {token}", CSV_URL, "-o", str(LOCAL_CSV)],
        check=True,
    )


def build_qid_to_mcq() -> dict[str, str]:
    """Return {question_id: reconstructed mcq_prompt} via the committed loader."""
    _ensure_csv()
    spec = importlib.util.spec_from_file_location("rqg", str(ROOT / "scripts" / "run_qwen3_gpqa.py"))
    rqg = importlib.util.module_from_spec(spec)
    sys.modules["rqg"] = rqg
    spec.loader.exec_module(rqg)

    import datasets

    _real = datasets.load_dataset

    def _patched(path, *a, **k):  # type: ignore[no-untyped-def]
        if path == rqg.GPQA_HF_PATH:
            return _real("csv", data_files=str(LOCAL_CSV), split="train")
        return _real(path, *a, **k)

    datasets.load_dataset = _patched
    samples = rqg.load_gpqa_diamond(limit=None)
    return {s.question_id: rqg.format_mcq_prompt(s) for s in samples}


def load_records() -> list[dict[str, Any]]:
    """Base 785 Haiku records, with any prior new-judge scores merged in."""
    records: list[dict[str, Any]] = []
    for d in SCORED_DIRS:
        for line in (d / "results.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("autorater_legibility") is None:
                continue
            records.append(
                {
                    "model_dir": d.name,
                    "question_id": r["question_id"],
                    "raw_cot": r["raw_cot"],
                    "raw_model_content": r["raw_model_content"],
                    "haiku_leg": int(r["autorater_legibility"]),
                    "haiku_cov": int(r["autorater_coverage"]),
                }
            )
    if SCORES_PATH.exists():  # merge prior new-judge scores (append/resume)
        by_key = {(r["model_dir"], r["question_id"]): r for r in records}
        for line in SCORES_PATH.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            rec = by_key.get((e["model_dir"], e["question_id"]))
            if rec is None:
                continue
            for j in NEW_JUDGES:
                for suf in _SUFFIXES:
                    if f"{j}{suf}" in e:
                        rec[f"{j}{suf}"] = e[f"{j}{suf}"]
    return records


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _est_cost(judge: str, rendered: str, raw: str) -> float:
    pin, pout = JUDGE_CFG[judge]["price"]
    return (len(rendered) // 4) * pin + (len(raw) // 4) * pout


def _status_of(exc: BaseException) -> int | None:
    return (
        getattr(exc, "status_code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
        or getattr(exc, "code", None)
    )


def _is_rate_limit(exc: BaseException) -> bool:
    return "RateLimit" in type(exc).__name__ or _status_of(exc) == 429


def _is_retryable(exc: BaseException) -> bool:
    """429s plus transient server/connection errors (5xx, overloaded, conn drops)."""
    if _is_rate_limit(exc):
        return True
    name = type(exc).__name__
    if any(
        s in name
        for s in ("ServerError", "APIConnection", "Timeout", "Overloaded", "ServiceUnavailable", "InternalServer")
    ):
        return True
    return _status_of(exc) in {500, 502, 503, 504, 529}


async def _complete_with_retry(client: Any, rendered: str, guard: dict[str, int]) -> str | None:
    """Call client.complete with backoff on 429s + transient server errors.

    Returns None on retry-exhausted transient failures (recorded as a skip, never
    re-raised into the gather); raises HaltError only at 5 consecutive terminal
    429s. Non-transient errors (auth, bad request) propagate.
    """
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    async def _attempt() -> str:
        return await client.complete(rendered)

    try:
        raw = await _attempt()
    except BaseException as exc:  # noqa: BLE001 — bookkeeping then skip/halt, never crash on a transient error
        if _is_retryable(exc):
            if _is_rate_limit(exc):
                guard["total_429"] += 1
                guard["consecutive"] += 1
                if guard["consecutive"] >= 5:
                    raise HaltError(
                        f"{guard['consecutive']} consecutive terminal 429s — halting (rate-limit ceiling)"
                    ) from exc
            else:
                guard["total_server"] += 1
            return None
        raise
    guard["consecutive"] = 0
    return raw


async def score_judge(
    judge: str,
    records: list[dict[str, Any]],
    qid_to_mcq: dict[str, str],
    prompt: LegibilityCoveragePrompt,
    spend: dict[str, float],
) -> None:
    """Score records through one judge in place.

    Idempotent: skips records that already carry an outcome for this judge
    (resume/append).
    """
    cfg = JUDGE_CFG[judge]
    client = get_grader_client(NEW_JUDGES[judge], max_tokens=cfg["max_tokens"])
    sem = asyncio.Semaphore(cfg["concurrency"])
    interval: float = cfg["min_interval"]
    done = {"n": 0, "fail": 0, "skip": 0, "already": 0}
    guard = {"consecutive": 0, "total_429": 0, "total_server": 0}
    clock = {"next": 0.0}

    async def one(rec: dict[str, Any]) -> None:
        if f"{judge}_parse_fail" in rec or rec.get(f"{judge}_skip_reason"):
            done["already"] += 1
            return
        if spend["total"] >= SPEND_CEILING:
            raise HaltError(f"spend ceiling ${SPEND_CEILING} reached (est ${spend['total']:.2f})")
        rendered = prompt.render(
            question=qid_to_mcq[rec["question_id"]],
            explanation=rec["raw_cot"],
            answer=rec["raw_model_content"],
        )
        async with sem:
            if interval > 0:  # proactive TPM pacing (gpt-4o)
                now = asyncio.get_event_loop().time()
                wait = clock["next"] - now
                if wait > 0:
                    await asyncio.sleep(wait)
                clock["next"] = asyncio.get_event_loop().time() + interval
            raw = await _complete_with_retry(client, rendered, guard)
        if raw is None:  # rate-limit skip
            rec[f"{judge}_leg"] = None
            rec[f"{judge}_cov"] = None
            rec[f"{judge}_parse_fail"] = True
            rec[f"{judge}_skip_reason"] = "transient_exhausted"
            done["skip"] += 1
            done["n"] += 1
            return
        spend[judge] += _est_cost(judge, rendered, raw)
        spend["total"] = sum(v for k, v in spend.items() if k != "total")
        try:
            leg, cov, _ = prompt.parse(raw)
            rec[f"{judge}_leg"] = leg
            rec[f"{judge}_cov"] = cov
            rec[f"{judge}_parse_fail"] = False
        except Exception as exc:  # noqa: BLE001 — record + count, never abort on one item
            rec[f"{judge}_leg"] = None
            rec[f"{judge}_cov"] = None
            rec[f"{judge}_parse_fail"] = True
            rec[f"{judge}_raw"] = raw
            rec[f"{judge}_parse_err"] = f"{type(exc).__name__}: {exc}"
            done["fail"] += 1
        done["n"] += 1
        if done["n"] % 100 == 0:
            _write_scores(records)  # checkpoint so a crash never discards completed work
            print(
                f"[ez]   {judge} {done['n']} new ({done['skip']} skip, 429s={guard['total_429']}, "
                f"5xx={guard['total_server']}) | est ${spend[judge]:.2f}",
                file=sys.stderr,
            )
        if done["n"] >= 50 and done["fail"] / done["n"] > PARSE_FAIL_HALT:
            raise HaltError(
                f"judge {judge!r} parse-fail rate {done['fail']}/{done['n']} > {PARSE_FAIL_HALT:.0%}"
            )

    await asyncio.gather(*(one(r) for r in records))
    print(
        f"[ez] {judge} DONE | new={done['n']} already={done['already']} "
        f"parse_fail={done['fail']} rate_skip={done['skip']} 429s={guard['total_429']} "
        f"5xx={guard['total_server']} | est ${spend[judge]:.2f} | running ${spend['total']:.2f}",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------
# Kappa + ranking reversal (over whichever judges are present)
# --------------------------------------------------------------------------


def present_new_judges(records: list[dict[str, Any]]) -> list[str]:
    return [j for j in NEW_JUDGES if any(r.get(f"{j}_leg") is not None for r in records)]


def _axis_kappa(records: list[dict[str, Any]], axis: str, judges: list[str]) -> dict[str, Any]:
    new = [j for j in judges if j != HAIKU]
    survivors = [
        r
        for r in records
        if r.get(f"haiku_{axis}") is not None
        and all(r.get(f"{j}_{axis}") is not None for j in new)
    ]
    cols = {HAIKU: [r[f"haiku_{axis}"] for r in survivors]}
    for j in new:
        cols[j] = [r[f"{j}_{axis}"] for r in survivors]
    pairwise = {
        f"{a}__{b}": cohen_kappa_quadratic(cols[a], cols[b], num_categories=NUM_CATEGORIES)
        for a, b in combinations(judges, 2)
    }
    return {
        "n_survivors": len(survivors),
        "pairwise_kappa": pairwise,
        "per_judge_mean": {j: (round(sum(c) / len(c), 3) if c else None) for j, c in cols.items()},
        "dominant_frac": {j: round(dominant_category_fraction(cols[j]), 3) for j in judges if cols[j]},
    }


def _ranking_reversals(records: list[dict[str, Any]], axis: str, judges: list[str]) -> dict[str, Any]:
    per_judge_subject: dict[str, dict[str, float]] = {j: {} for j in judges}
    for d in sorted({r["model_dir"] for r in records}):
        rows = [r for r in records if r["model_dir"] == d]
        for j in judges:
            key = "haiku" if j == HAIKU else j
            vals = [r[f"{key}_{axis}"] for r in rows if r.get(f"{key}_{axis}") is not None]
            if vals:
                per_judge_subject[j][d] = sum(vals) / len(vals)
    reversals = detect_ranking_reversals(per_judge_subject)
    return {
        "per_judge_model_means": {j: {k: round(v, 3) for k, v in m.items()} for j, m in per_judge_subject.items()},
        "summary": ranking_reversal_summary(per_judge_subject),
        "n_reversals": len(reversals),
        "reversals": [r.model_dump() for r in reversals],
    }


def compute_summary(records: list[dict[str, Any]], spend: dict[str, float]) -> dict[str, Any]:
    judges = [HAIKU, *present_new_judges(records)]
    out: dict[str, Any] = {"judges_present": judges, "axes": {}, "per_model": {}, "ranking_reversal": {}}
    for axis in ("leg", "cov"):
        out["axes"][axis] = _axis_kappa(records, axis, judges)
        out["ranking_reversal"][axis] = _ranking_reversals(records, axis, judges)
        out["per_model"][axis] = {
            d: _axis_kappa([r for r in records if r["model_dir"] == d], axis, judges)
            for d in sorted({r["model_dir"] for r in records})
        }
    if "gemini-2.5-pro" in judges:
        out["fidelity_haiku_vs_gemini"] = {
            "leg": out["axes"]["leg"]["pairwise_kappa"].get(f"{HAIKU}__gemini-2.5-pro"),
            "cov": out["axes"]["cov"]["pairwise_kappa"].get(f"{HAIKU}__gemini-2.5-pro"),
        }
    out["meta"] = {
        "n_records": len(records),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA,
        "num_categories": NUM_CATEGORIES,
        "max_tokens": {j: JUDGE_CFG[j]["max_tokens"] for j in NEW_JUDGES},
        "mcq_prompt_reconstruction": "Phase-A verified (question_text byte-match + correct-letter, 24/24)",
        "parse_fail_rate": {
            j: round(sum(1 for r in records if r.get(f"{j}_parse_fail")) / len(records), 4)
            for j in present_new_judges(records)
        },
        "gpt_4o_pending": "gpt-4o" not in judges,
    }
    return out


def _write_scores(records: list[dict[str, Any]]) -> None:
    """Atomically persist the per-trajectory scores (checkpoint; enables resume)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SCORES_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(SCORES_PATH)


def write_outputs(records: list[dict[str, Any]], spend: dict[str, float]) -> dict[str, Any]:
    _write_scores(records)
    summary = compute_summary(records, spend)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


async def _amain(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(NEW_JUDGES), help="comma-sep new judges to score this run")
    ap.add_argument("--dry-run", type=int, default=0, help="score N trajectories/judge then exit (liveness gate)")
    args = ap.parse_args(argv)
    judges_to_score = [j.strip() for j in args.judges.split(",") if j.strip()]
    for j in judges_to_score:
        if j not in NEW_JUDGES:
            raise HaltError(f"unknown judge {j!r}; valid: {list(NEW_JUDGES)}")

    prompt = LegibilityCoveragePrompt.load(PROMPT_VERSION)
    if prompt.sha256 != PROMPT_SHA:
        raise HaltError(f"prompt SHA mismatch: {prompt.sha256[:12]} != {PROMPT_SHA[:12]}")

    qid_to_mcq = build_qid_to_mcq()
    records = load_records()
    missing = [r["question_id"] for r in records if r["question_id"] not in qid_to_mcq]
    if missing:
        raise HaltError(f"{len(missing)} question_ids did not resolve (e.g. {missing[:3]})")
    print(
        f"[ez] {len(qid_to_mcq)} mcq_prompts; {len(records)} records; scoring {judges_to_score}",
        file=sys.stderr,
    )

    spend = {j: 0.0 for j in NEW_JUDGES}
    spend["total"] = 0.0

    if args.dry_run > 0:
        for judge in judges_to_score:
            client = get_grader_client(NEW_JUDGES[judge], max_tokens=JUDGE_CFG[judge]["max_tokens"])
            for rec in records[: args.dry_run]:
                rendered = prompt.render(
                    question=qid_to_mcq[rec["question_id"]],
                    explanation=rec["raw_cot"],
                    answer=rec["raw_model_content"],
                )
                raw = await client.complete(rendered)
                try:
                    leg, cov, _ = prompt.parse(raw)
                    v = f"OK leg={leg} cov={cov} ({len(raw)} chars, JSON complete)"
                except Exception as exc:  # noqa: BLE001
                    v = f"PARSE-FAIL {type(exc).__name__} | raw[:200]={raw[:200]!r}"
                print(f"[ez-dry] {judge} {rec['question_id']}: {v}", file=sys.stderr)
        return 0

    for judge in judges_to_score:
        print(f"[ez] scoring {judge}...", file=sys.stderr)
        await score_judge(judge, records, qid_to_mcq, prompt, spend)
        _write_scores(records)  # checkpoint after each judge completes

    summary = write_outputs(records, spend)
    fid = summary.get("fidelity_haiku_vs_gemini")
    print(
        f"[ez] complete. judges={summary['judges_present']} | "
        f"fidelity Haiku-vs-Gemini kappa={fid} | est ${spend['total']:.2f}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except HaltError as exc:
        print(f"[ez] HALT: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
