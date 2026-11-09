"""Tests for scripts/evaluate.py — threshold handling for single-model and ensemble runs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluate.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tta_test_threshold_uses_the_given_threshold():
    """A threshold given on the command line wins over the validation search."""
    script = _load_script()

    result = script.tta_test_threshold(0.4, {"final/val_tta_threshold": 0.6}, 0.5)

    assert result == 0.4, "an explicit --threshold must decide the test TTA evaluation"


def test_tta_test_threshold_falls_back_to_the_validation_search():
    """With no given threshold, reuse the one searched on validation, never on test."""
    script = _load_script()

    result = script.tta_test_threshold(None, {"final/val_tta_threshold": 0.6}, 0.5)

    assert result == 0.6, (
        "the validation-searched threshold must be reused, not re-searched on test"
    )


def test_check_ensemble_threshold_refuses_a_search_on_the_test_split():
    """Searching the threshold on test would bias the reported score: refuse it."""
    script = _load_script()

    with pytest.raises(SystemExit):
        script.check_ensemble_threshold("test", None)


def test_check_ensemble_threshold_allows_validation_or_a_given_threshold():
    """Validation searches freely; a given threshold on test is honoured."""
    script = _load_script()

    script.check_ensemble_threshold("val", None)
    script.check_ensemble_threshold("test", 0.5)
