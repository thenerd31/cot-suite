"""CLI shim: `python scripts/verify_keys.py` → `python -m cotsuite.verify_keys`.

All logic lives in `src/cotsuite/verify_keys.py` so run scripts can import
`from cotsuite.verify_keys import require_keys` for import-time assertion.
"""

from __future__ import annotations

import sys

from cotsuite.verify_keys import main

if __name__ == "__main__":
    sys.exit(main())
