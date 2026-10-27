"""Training infrastructure for DeepContrail segmentation."""

from deepcontrail.segmentation.training.config import (
    CHECKPOINT_ROOT_ENV,
    MLFLOW_TRACKING_URI_ENV,
    TrainingConfig,
    load_configs,
    resolve_run_paths,
)
from deepcontrail.segmentation.training.evaluation import (
    evaluate_ensemble,
    evaluate_global,
    find_optimal_threshold,
)
from deepcontrail.segmentation.training.module import SegmentationModule
from deepcontrail.segmentation.training.run import (
    TrainingRun,
    build_dataloaders,
    build_logger,
    default_run_name,
    prepare_training,
    train,
)
from deepcontrail.segmentation.training.tta import tta_predict

__all__ = [
    "CHECKPOINT_ROOT_ENV",
    "MLFLOW_TRACKING_URI_ENV",
    "SegmentationModule",
    "TrainingConfig",
    "TrainingRun",
    "build_dataloaders",
    "build_logger",
    "default_run_name",
    "evaluate_ensemble",
    "evaluate_global",
    "find_optimal_threshold",
    "load_configs",
    "prepare_training",
    "resolve_run_paths",
    "train",
    "tta_predict",
]
