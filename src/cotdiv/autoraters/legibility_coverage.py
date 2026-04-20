"""Port of the Emmons & Zimmermann (2510.23966) legibility + coverage autorater.

The Appendix C prompt is stored verbatim at
`cotdiv/autoraters/prompts/emmons_zimmermann_v1.txt`. The prompt is versioned;
its SHA256 is stored in metadata for reproducibility. To update the prompt,
ship a new version (v1_1, v2) — do not mutate the shipped file.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from importlib import resources
from string import Template
from typing import TYPE_CHECKING

from cotdiv.core.registry import register_metric
from cotdiv.core.schemas import MetricValue
from cotdiv.models.clients import get_grader_client

if TYPE_CHECKING:
    from cotdiv.core.trajectory import Trajectory


@dataclass(frozen=True)
class LegibilityCoveragePrompt:
    """A versioned legibility + coverage prompt template."""

    version: str
    template: str
    sha256: str

    @classmethod
    def load(cls, version: str = "emmons_zimmermann_v1") -> LegibilityCoveragePrompt:
        """Load a prompt by version name (no `.txt` suffix)."""
        pkg = resources.files("cotdiv.autoraters.prompts")
        template = (pkg / f"{version}.txt").read_text(encoding="utf-8")
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return cls(version=version, template=template, sha256=digest)

    def render(self, *, prompt: str, reasoning: str, answer: str) -> str:
        """Format the template against one trajectory's inputs.

        Uses `string.Template` ($-substitution) so braces inside the prompt
        (e.g., the JSON output example) are never reinterpreted as fields.
        """
        return Template(self.template).substitute(
            prompt=prompt,
            reasoning=reasoning,
            answer=answer,
        )

    def parse(self, completion: str) -> tuple[int, int, str]:
        """Extract `(legibility, coverage, rationale)` from an autorater response."""
        match = re.search(r"\{.*?\}", completion, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in autorater output: {completion!r}")
        data = json.loads(match.group(0))
        leg = int(data["legibility"])
        cov = int(data["coverage"])
        if not (0 <= leg <= 4 and 0 <= cov <= 4):
            raise ValueError(f"out-of-range scores: legibility={leg}, coverage={cov}")
        return leg, cov, data.get("rationale", "")


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
    prompt_text = first_user.text if first_user else ""
    rendered = prompt_obj.render(
        prompt=prompt_text or "",
        reasoning=trajectory.reasoning_text,
        answer=trajectory.final_answer or "",
    )

    legs: list[int] = []
    covs: list[int] = []
    rationales: list[str] = []
    for _ in range(runs):
        completion = await client.complete(rendered)
        leg, cov, rat = prompt_obj.parse(completion)
        legs.append(leg)
        covs.append(cov)
        rationales.append(rat)

    meta = {
        "autorater": autorater,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_obj.sha256,
        "rationales": rationales,
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
