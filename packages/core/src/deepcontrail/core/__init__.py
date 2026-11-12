"""Shared, torch-free foundations: dataset format, config, composites, splits, metrics."""

from importlib.metadata import version

from deepcontrail.core.config import DatasetConfig
from deepcontrail.core.contract import (
    BandChannel,
    BandManifest,
    DatasetContractError,
    DatasetInfo,
    read_band_manifest,
    read_dataset_info,
    scene_stems,
)
from deepcontrail.core.false_color import (
    FalseColorChannel,
    FalseColorConfig,
    check_false_color,
    false_color_from_spectral,
    load_spectral_false_color,
)
from deepcontrail.core.metrics import dice_coefficient, iou_score
from deepcontrail.core.splits import context_frames, partition_stems, split_fingerprint
from deepcontrail.core.validate import Issue, ValidationReport, validate_dataset

__version__ = version("deepcontrail-core")

__all__ = [
    "BandChannel",
    "BandManifest",
    "DatasetConfig",
    "DatasetContractError",
    "DatasetInfo",
    "FalseColorChannel",
    "FalseColorConfig",
    "Issue",
    "ValidationReport",
    "__version__",
    "check_false_color",
    "context_frames",
    "dice_coefficient",
    "false_color_from_spectral",
    "iou_score",
    "load_spectral_false_color",
    "partition_stems",
    "read_band_manifest",
    "read_dataset_info",
    "scene_stems",
    "split_fingerprint",
    "validate_dataset",
]
