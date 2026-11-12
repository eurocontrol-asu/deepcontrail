"""Tests that training package imports work and base imports stay torch-free."""

from __future__ import annotations

import subprocess
import sys


def test_import_segmentation_module() -> None:
    from deepcontrail.segmentation.training import SegmentationModule

    assert SegmentationModule is not None


def test_import_evaluate_global() -> None:
    from deepcontrail.segmentation.training import evaluate_global

    assert callable(evaluate_global)


def test_import_training_config() -> None:
    from deepcontrail.segmentation.training import TrainingConfig

    assert TrainingConfig is not None


def test_import_find_optimal_threshold() -> None:
    from deepcontrail.segmentation.training import find_optimal_threshold

    assert callable(find_optimal_threshold)


def test_training_api_does_not_expose_the_experiment_script_settings() -> None:
    import deepcontrail.segmentation.training as training

    script_only = {"resolve_run_paths", "CHECKPOINT_ROOT_ENV", "MLFLOW_TRACKING_URI_ENV"}

    assert not script_only & set(training.__all__), (
        "where scripts/train.py writes its runs is the script's business, not the library's"
    )
    assert not any(hasattr(training, name) for name in script_only)


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
