"""Training infrastructure for DeepContrail segmentation."""

from deepcontrail.segmentation.training.config import (
    CHECKPOINT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TrainingConfig,
    load_configs,
)
from deepcontrail.segmentation.training.evaluation import (
    evaluate_ensemble,
    evaluate_global,
    find_optimal_threshold,
)
from deepcontrail.segmentation.training.module import ContrailSegmentationModule
from deepcontrail.segmentation.training.tta import tta_predict

__all__ = [
    "CHECKPOINT_ROOT",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_TRACKING_URI",
    "ContrailSegmentationModule",
    "TrainingConfig",
    "evaluate_ensemble",
    "evaluate_global",
    "find_optimal_threshold",
    "load_configs",
    "tta_predict",
]
