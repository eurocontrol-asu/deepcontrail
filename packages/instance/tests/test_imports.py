"""``import deepcontrail.instance`` must stay light: no torch, no optional extra."""

from __future__ import annotations

import subprocess
import sys

import pytest


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


@pytest.mark.parametrize("name", ["evaluation", "tuning"])
def test_script_helpers_are_not_public_modules(name):
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"deepcontrail.instance.{name}")

    assert importlib.import_module(f"deepcontrail.instance._{name}"), (
        "the scripts' helpers stay available as a private module"
    )


def test_public_api_exports_scene_features_not_instance_features() -> None:
    import deepcontrail.instance as instance

    assert "scene_features" in instance.__all__
    assert not hasattr(instance, "instance_features"), "removed: it matched masks by identity"
