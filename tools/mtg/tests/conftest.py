"""Make the WORKTREE packages importable for the test suite (temporary, for worktree runs).

The shared venv's editable installs point at the main checkout, so in a worktree we prepend
the worktree package roots to sys.path — the same thing PYTHONPATH
= .../tools/mtg/src:.../packages/core/src does. This file is not part of the feature and is
removed after verification.
"""

import sys
from pathlib import Path

_WT = Path(__file__).resolve().parents[3]  # .../wt-mtg-conv
for rel in ("tools/mtg/src", "packages/core/src"):
    _p = _WT / rel
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
