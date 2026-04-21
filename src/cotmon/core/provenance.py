"""Provenance labels for tests, cues, and metrics.

Every test/cue/metric must carry a `Provenance` — either a paper citation
with `verified_against_pdf` gating whether we trust the replication, or an
extension marker for cotmon-original work that lives under `tests/extensions/`.

This is the structural separation between "papers we replicate" and "ideas we
invented." Unverified-or-extension work MUST NOT live in the primary
namespaces (`tests.turpin_counterfactual`, `tests.chen_cue_injection`,
`tests.lanham.*`) until the corresponding PDF has been cross-checked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    """Where a test/cue/metric came from.

    Paper-based replication:
        Provenance(arxiv_id="2307.13702", section="§3.1", verified_against_pdf=False)

    CoT-Divergence extension (not from any one paper):
        Provenance(arxiv_id=None, notes="Inspired by Sharma et al. sycophancy...")

    The `verified_against_pdf` flag flips to True only when a human has opened
    the referenced PDF and confirmed the implementation matches. Any catalog
    entry with `arxiv_id is not None and verified_against_pdf is False` is a
    candidate shortcut.
    """

    arxiv_id: str | None
    section: str | None = None
    verified_against_pdf: bool = False
    notes: str = ""

    def is_extension(self) -> bool:
        return self.arxiv_id is None

    def is_unverified_replication(self) -> bool:
        return self.arxiv_id is not None and not self.verified_against_pdf
