"""Port of the Emmons & Zimmermann (2510.23966) legibility + coverage autorater.

The verbatim Appendix C prompt lives at
`cotsuite/autoraters/prompts/emmons_zimmermann_v1.txt`. Its canonical SHA-256
is committed alongside at `emmons_zimmermann_v1.sha256` and cross-referenced
in `docs/paper-refs/2510.23966-appendix-c.md`.

The prompt is versioned. To update — ship a new version (`emmons_zimmermann_v1_1`,
`emmons_zimmermann_v2`), never mutate the shipped file. This invariant is
enforced by the SHA-256 drift check in
`tests/test_appendix_c_prompt_integrity.py`.

The prompt uses `{question}`, `{explanation}`, `{answer}` as template
placeholders and contains JSON schema braces that would conflict with
`str.format()`. We render via literal `.replace()` so that only the three
named placeholders are substituted; every other `{...}` (the JSON schema at
the bottom of the prompt) passes through untouched.

The prompt's required output format is:

    {"justification": "...", "legibility_score": 0-4, "coverage_score": 0-4}
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING

from cotsuite.core.registry import register_metric
from cotsuite.core.schemas import MetricValue
from cotsuite.models.clients import get_grader_client

if TYPE_CHECKING:
    from cotsuite.core.trajectory import Trajectory


@dataclass(frozen=True)
class LegibilityCoveragePrompt:
    """A versioned legibility + coverage prompt template."""

    version: str
    template: str
    sha256: str

    @classmethod
    def load(cls, version: str = "emmons_zimmermann_v1") -> LegibilityCoveragePrompt:
        """Load a prompt by version name (no `.txt` suffix)."""
        pkg = resources.files("cotsuite.autoraters.prompts")
        template = (pkg / f"{version}.txt").read_text(encoding="utf-8")
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return cls(version=version, template=template, sha256=digest)

    @classmethod
    def canonical_sha256(cls, version: str = "emmons_zimmermann_v1") -> str:
        """Load the committed canonical SHA-256 for a prompt version."""
        pkg = resources.files("cotsuite.autoraters.prompts")
        return (pkg / f"{version}.sha256").read_text(encoding="utf-8").strip()

    def render(self, *, question: str, explanation: str, answer: str) -> str:
        """Substitute the three template placeholders.

        Uses literal `.replace()` rather than `str.format()` so that JSON
        schema braces elsewhere in the prompt (the `{...}` output block at
        the bottom of the Appendix C prompt) pass through untouched.
        """
        return (
            self.template.replace("{question}", question)
            .replace("{explanation}", explanation)
            .replace("{answer}", answer)
        )

    def parse(self, completion: str) -> tuple[int, int, str]:
        """Extract `(legibility_score, coverage_score, justification)` from output."""
        obj = _extract_first_json_object(completion)
        leg = int(obj["legibility_score"])
        cov = int(obj["coverage_score"])
        if not (0 <= leg <= 4 and 0 <= cov <= 4):
            raise ValueError(
                f"out-of-range scores: legibility_score={leg}, coverage_score={cov}",
            )
        return leg, cov, str(obj.get("justification", ""))


def _extract_first_json_object(completion: str) -> dict:
    """Locate and decode the first complete JSON object in `completion`.

    Uses `json.JSONDecoder.raw_decode` starting from the first `{` — robust
    to nested braces inside strings (unlike a naive regex) and to prose
    surrounding the object (either a fenced code block or a bare block).
    """
    start = completion.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in autorater output: {completion!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(completion[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"autorater output is not valid JSON: {completion!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"autorater output is not a JSON object: {completion!r}")
    return data


@register_metric("legibility")
async def _legibility_metric(
    trajectory: Trajectory,
    *,
    autorater: str = "google/gemini-2.5-pro",
    runs: int = 5,
    **_: object,
) -> MetricValue:
    leg, _, meta = await _score(trajectory, autorater=autorater, runs=runs)
    return MetricValue(name="legibility", metadata=meta, **leg)


@register_metric("coverage")
async def _coverage_metric(
    trajectory: Trajectory,
    *,
    autorater: str = "google/gemini-2.5-pro",
    runs: int = 5,
    **_: object,
) -> MetricValue:
    _, cov, meta = await _score(trajectory, autorater=autorater, runs=runs)
    return MetricValue(name="coverage", metadata=meta, **cov)


async def legibility_coverage(
    trajectory: Trajectory,
    *,
    autorater: str = "google/gemini-2.5-pro",
    runs: int = 5,
    prompt_version: str = "emmons_zimmermann_v1",
) -> tuple[MetricValue, MetricValue]:
    """Score a trajectory on legibility AND coverage in one autorater pass.

    Returns two `MetricValue` objects. Each is the mean over `runs` samples
    from the autorater; `stddev` and `stderr` are populated when `runs > 1`.
    """
    leg, cov, meta = await _score(
        trajectory,
        autorater=autorater,
        runs=runs,
        prompt_version=prompt_version,
    )
    return (
        MetricValue(name="legibility", metadata=meta, **leg),
        MetricValue(name="coverage", metadata=meta, **cov),
    )


async def _score(
    trajectory: Trajectory,
    *,
    autorater: str,
    runs: int,
    prompt_version: str = "emmons_zimmermann_v1",
) -> tuple[dict, dict, dict]:
    prompt_obj = LegibilityCoveragePrompt.load(prompt_version)
    client = get_grader_client(autorater)

    first_user = next((t for t in trajectory.turns if t.role == "user"), None)
    question_text = first_user.text if first_user else ""
    rendered = prompt_obj.render(
        question=question_text or "",
        explanation=trajectory.reasoning_text,
        answer=trajectory.final_answer or "",
    )

    legs: list[int] = []
    covs: list[int] = []
    justifications: list[str] = []
    for _ in range(runs):
        completion = await client.complete(rendered)
        leg, cov, justification = prompt_obj.parse(completion)
        legs.append(leg)
        covs.append(cov)
        justifications.append(justification)

    meta = {
        "autorater": autorater,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_obj.sha256,
        "justifications": justifications,
        "n_runs": runs,
    }
    return _agg("legibility", legs), _agg("coverage", covs), meta


def _agg(name: str, values: list[int]) -> dict:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return {
        "value": mean,
        "stddev": sd,
        "stderr": se,
        "n_runs": len(values),
    }
