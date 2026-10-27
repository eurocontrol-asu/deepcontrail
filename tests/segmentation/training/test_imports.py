"""Tests that training package imports work and base imports stay torch-free."""

from __future__ import annotations

import subprocess
import sys


def test_import_contrail_segmentation_module() -> None:
    from deepcontrail.segmentation.training import ContrailSegmentationModule

    assert ContrailSegmentationModule is not None


def test_import_evaluate_global() -> None:
    from deepcontrail.segmentation.training import evaluate_global

    assert callable(evaluate_global)


def test_import_training_config() -> None:
    from deepcontrail.segmentation.training import TrainingConfig

    assert TrainingConfig is not None


def test_import_find_optimal_threshold() -> None:
    from deepcontrail.segmentation.training import find_optimal_threshold

    assert callable(find_optimal_threshold)


def test_import_constants() -> None:
    from deepcontrail.segmentation.training import (
        CHECKPOINT_ROOT,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
    )

    assert CHECKPOINT_ROOT is not None
    assert MLFLOW_TRACKING_URI is not None
    assert MLFLOW_EXPERIMENT_NAME is not None


def test_import_load_configs() -> None:
    from deepcontrail.segmentation.training import load_configs

    assert callable(load_configs)


def test_segmentation_data_pipeline_imports_without_torch() -> None:
    code = "import sys, deepcontrail.segmentation; print('torch' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", (
        "deepcontrail.segmentation (config + dataset) must import without torch; "
        "only models/losses/training need it"
    )
