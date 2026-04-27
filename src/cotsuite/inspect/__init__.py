"""Inspect AI integration: scorers and solvers.

This module is registered as a `[project.entry-points.inspect_ai]`
endpoint named ``cotsuite``, so importing it eagerly at install time
populates Inspect AI's process-global registry. After
``pip install cot-suite``, users can run any registered scorer via:

    inspect eval some_task --scorer cotsuite/cot_legibility_coverage

or import directly:

    from cotsuite.inspect.scorers import cot_legibility_coverage
"""

# Eager import so each `@scorer(...)` decorator fires and registers
# into Inspect's registry the moment cotsuite.inspect is imported by
# Inspect's entry-point loader.
from cotsuite.inspect import scorers  # noqa: F401

__all__ = ["scorers"]
