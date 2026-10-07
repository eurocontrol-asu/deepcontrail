"""Training infrastructure for DeepContrail segmentation."""

from deepcontrail.training.config import (
    CHECKPOINT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TrainingConfig,
    load_configs,
)
from deepcontrail.training.evaluation import evaluate_global, find_optimal_threshold
from deepcontrail.training.module import ContrailSegmentationModule

__all__ = [
    "CHECKPOINT_ROOT",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_TRACKING_URI",
    "ContrailSegmentationModule",
    "TrainingConfig",
    "evaluate_global",
    "find_optimal_threshold",
    "load_configs",
]
