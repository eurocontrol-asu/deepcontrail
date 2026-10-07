"""Tests that training package imports work and base imports stay torch-free."""

from __future__ import annotations

import importlib
import sys


def test_import_contrail_segmentation_module() -> None:
    from deepcontrail.training import ContrailSegmentationModule

    assert ContrailSegmentationModule is not None


def test_import_evaluate_global() -> None:
    from deepcontrail.training import evaluate_global

    assert callable(evaluate_global)


def test_import_training_config() -> None:
    from deepcontrail.training import TrainingConfig

    assert TrainingConfig is not None


def test_import_find_optimal_threshold() -> None:
    from deepcontrail.training import find_optimal_threshold

    assert callable(find_optimal_threshold)


def test_import_constants() -> None:
    from deepcontrail.training import (
        CHECKPOINT_ROOT,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
    )

    assert CHECKPOINT_ROOT is not None
    assert MLFLOW_TRACKING_URI is not None
    assert MLFLOW_EXPERIMENT_NAME is not None


def test_import_load_configs() -> None:
    from deepcontrail.training import load_configs

    assert callable(load_configs)


def test_deepcontrail_data_works_without_torch() -> None:
    saved = sys.modules.pop("torch", "NOT_LOADED")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        if "deepcontrail.data" in sys.modules:
            importlib.reload(sys.modules["deepcontrail.data"])
        else:
            import deepcontrail.data  # noqa: F401
    finally:
        if saved == "NOT_LOADED":
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
