"""``import deepcontrail.instance`` must stay light: no torch, no optional extra."""

from __future__ import annotations

import subprocess
import sys


def test_instance_import_loads_no_heavy_or_optional_dependency():
    heavy = ("torch", "sklearn", "pycocotools", "optuna", "deepcontrail.segmentation")
    code = (
        "import sys, deepcontrail.instance; "
        f"print(sorted(m for m in {heavy!r} if sys.modules.get(m) is not None))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]", (
        "deepcontrail.instance needs only core, numpy, scipy, pydantic and PyYAML at import "
        f"time (extras load on use), loaded: {result.stdout.strip()}"
    )


def test_instance_package_reports_its_version():
    import deepcontrail.instance

    assert deepcontrail.instance.__version__, "the package exposes its installed version"
