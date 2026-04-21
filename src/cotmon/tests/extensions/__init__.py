"""CoT-Divergence extensions — tests/cues that are NOT direct replications.

Anything in this subpackage is one of:

    (a) inspired by published work but not a verbatim reproduction
    (b) a cotmon-original measurement not in any one paper
    (c) a best-guess implementation waiting for PDF cross-check that will
        either be promoted back into the primary namespace (verified) or
        stay here (unverifiable)

Every entry carries a `Provenance` with `arxiv_id=None` (pure extension) or
`arxiv_id=<id>, verified_against_pdf=False` (pending verification).

Promotion rule: move an entry out of `extensions/` and into the primary
namespace only when:
    1. The paper PDF has been cross-checked by a human, and
    2. The implementation matches the paper's template within one short
       paragraph of explanation, and
    3. `provenance.verified_against_pdf = True` in the catalog entry.
"""

from cotmon.tests.extensions.authority_bias import AUTHORITY_BIAS
from cotmon.tests.extensions.chen_authority_cue import CHEN_AUTHORITY_CUE
from cotmon.tests.extensions.chen_visual_pattern_cue import (
    CHEN_VISUAL_PATTERN_SIMPLIFIED,
)

__all__ = [
    "AUTHORITY_BIAS",
    "CHEN_AUTHORITY_CUE",
    "CHEN_VISUAL_PATTERN_SIMPLIFIED",
]
