"""Smoke test for the B2 Stage B apparatus (scripts/validate_b2_turpin_stage_b.py).

No API calls, no spend. Exercises:
  1. argument parsing (build_parser)
  2. the dataset loader on a tiny slice of the real vendored Turpin artifacts
     (local file I/O only)
  3. the Turpin answer extractor on synthetic completions

The script lives under scripts/ (not an importable package), so it is loaded
by file path via importlib.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_b2_turpin_stage_b.py"


def _load_module():
    mod_name = "validate_b2_turpin_stage_b"
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass type resolution (which reads
    # sys.modules[cls.__module__].__dict__) works for a path-loaded module.
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


stage_b = _load_module()


def test_arg_parser_defaults() -> None:
    parser = stage_b.build_parser()
    args = parser.parse_args([])
    assert args.model == "all"
    assert args.n == stage_b.DEFAULT_N
    assert args.judge == stage_b.DEFAULT_JUDGE
    # Default task string parses to the 5 Stage-A-locked tasks.
    tasks = tuple(t.strip() for t in args.tasks.split(","))
    assert tasks == stage_b.STAGE_B_TASKS
    assert len(tasks) == 5


def test_arg_parser_single_model_and_overrides() -> None:
    parser = stage_b.build_parser()
    args = parser.parse_args(["--model", "claude-sonnet-4-6", "--n", "10", "--tasks", "hyperbaton"])
    assert args.model == "claude-sonnet-4-6"
    assert args.n == 10
    assert args.tasks == "hyperbaton"


def test_arg_parser_rejects_unknown_model() -> None:
    parser = stage_b.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "gpt-9-ultra"])


def test_model_registry_has_four_models() -> None:
    assert set(stage_b.MODEL_REGISTRY) == {
        "gpt-3.5-turbo",
        "llama-3.1-8b",
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
    }
    # One Modal-backed model, three API.
    kinds = [m.kind for m in stage_b.MODEL_REGISTRY.values()]
    assert kinds.count("modal") == 1
    assert kinds.count("api") == 3


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("...reasoning... The best answer is: (C)", "C"),
        ("The best answer is: D", "D"),
        ("blah\nThe best answer is: (a)", "A"),
        # Falls back to canonical extractor on \boxed / Answer: forms.
        ("Answer: B", "B"),
        ("\\boxed{D}", "D"),
        ("no commitment here", ""),
    ],
)
def test_extract_turpin_answer(completion: str, expected: str) -> None:
    assert stage_b.extract_turpin_answer(completion) == expected


def test_load_task_samples_on_vendored_data() -> None:
    """Loader builds well-formed samples from the real vendored artifacts.

    Reads validation/turpin_artifacts/ (committed) — local I/O, no API.
    """
    samples = stage_b.load_task_samples("movie_recommendation", limit=2)
    assert len(samples) == 2
    for s in samples:
        assert s.task == "movie_recommendation"
        assert s.question_id.startswith("movie_recommendation_")
        assert s.correct_letter in {"A", "B", "C", "D"}
        assert s.bias_target_letter in {"A", "B", "C", "D"}
        # Few-shot scaffold is the prefix of both prompts.
        assert s.baseline_prompt.startswith("Recommend movies similar")
        assert s.biased_prompt.startswith("Recommend movies similar")
        # The biased prompt carries the suggested-answer cue; baseline does not.
        assert f"({s.bias_target_letter})" in s.biased_prompt
        assert "I personally think the answer is" in s.biased_prompt
        assert "I personally think the answer is" not in s.baseline_prompt
        # Both end with Turpin's answer-format instruction.
        assert s.baseline_prompt.rstrip().endswith("stick to this format.")


def test_load_task_samples_all_five_tasks_present() -> None:
    """Every Stage-A-locked task has vendored artifacts that load."""
    for task in stage_b.STAGE_B_TASKS:
        samples = stage_b.load_task_samples(task, limit=1)
        assert len(samples) == 1, f"task {task} produced no samples"


def test_load_task_samples_missing_task_raises() -> None:
    with pytest.raises(FileNotFoundError):
        stage_b.load_task_samples("not_a_real_task", limit=1)


def test_bootstrap_drop_ci_sign_and_shape() -> None:
    """Bootstrap CI returns (lo, hi) in Turpin sign convention (negative drop).

    Synthetic rows: all bias-inconsistent, baseline correct, biased wrong →
    drop = 100pp → Turpin-signed -100pp. Degenerate (zero variance) so lo == hi.
    """
    rows = [
        stage_b.CellRow(
            task="t",
            question_id=f"t_{i}",
            baseline_ans="A",  # correct
            biased_ans="B",  # bias target (wrong)
            correct_ans="A",
            bias_target_letter="B",
            bias_followed=True,
        )
        for i in range(20)
    ]
    lo, hi = stage_b.bootstrap_drop_ci(rows, resamples=200, seed=0)
    assert lo == pytest.approx(-100.0)
    assert hi == pytest.approx(-100.0)


def test_bootstrap_drop_ci_empty_pool() -> None:
    # All bias-consistent (target == correct) → empty inconsistent pool → (0, 0).
    rows = [
        stage_b.CellRow(
            task="t",
            question_id="t_0",
            baseline_ans="A",
            biased_ans="A",
            correct_ans="A",
            bias_target_letter="A",
            bias_followed=True,
        )
    ]
    assert stage_b.bootstrap_drop_ci(rows) == (0.0, 0.0)
