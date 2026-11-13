"""The deepcontrail package installs the whole library, at its own version."""

from __future__ import annotations

import importlib
from importlib.metadata import requires, version

import pytest

LIBRARY = ("deepcontrail-core", "deepcontrail-segmentation", "deepcontrail-instance")


@pytest.mark.parametrize(
    "module", ["deepcontrail.core", "deepcontrail.segmentation", "deepcontrail.instance"]
)
def test_installing_deepcontrail_gives_every_library_package(module: str):
    imported = importlib.import_module(module)

    assert imported.__version__ == version("deepcontrail"), "the packages are released together"


@pytest.mark.parametrize("extra", ["pycocotools", "sklearn"])
def test_installing_deepcontrail_gives_the_instance_extras(extra: str):
    importlib.import_module(extra)


def test_deepcontrail_requires_the_library_but_not_the_converter():
    required = " ".join(requires("deepcontrail") or [])

    assert all(name in required for name in LIBRARY)
    assert "deepcontrail-opencontrails" not in required, "the converter (TensorFlow) is a tool"
