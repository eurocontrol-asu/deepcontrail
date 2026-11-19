"""Shared, torch-free foundations: dataset format, config, composites, splits, metrics."""

from importlib.metadata import version

from deepcontrail.core.config import DatasetConfig
from deepcontrail.core.contract import (
    BandChannel,
    BandManifest,
    DatasetContractError,
    DatasetInfo,
    context_ids,
    read_band_manifest,
    read_dataset_info,
    read_neighbours,
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
from deepcontrail.core.splits import (
    context_frames,
    partition_stems,
    read_stem_list,
    split_fingerprint,
)
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
    "context_ids",
    "dice_coefficient",
    "false_color_from_spectral",
    "iou_score",
    "load_spectral_false_color",
    "partition_stems",
    "read_band_manifest",
    "read_dataset_info",
    "read_neighbours",
    "read_stem_list",
    "scene_stems",
    "split_fingerprint",
    "validate_dataset",
]
