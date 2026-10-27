"""``import deepcontrail.core`` must not pull in torch, albumentations or tensorflow."""

from __future__ import annotations

import subprocess
import sys


def test_core_import_loads_no_heavy_dependency():
    code = (
        "import sys, deepcontrail.core; "
        "print(sorted(m for m in ('torch', 'albumentations', 'tensorflow') "
        "if sys.modules.get(m) is not None))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", (
        f"deepcontrail.core must stay torch-free at import time, loaded: {result.stdout.strip()}"
    )
