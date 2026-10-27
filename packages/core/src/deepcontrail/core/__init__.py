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
)
from deepcontrail.core.false_color import (
    FalseColorChannel,
    FalseColorConfig,
    check_false_color,
    false_color_from_spectral,
    load_spectral_false_color,
)
from deepcontrail.core.metrics import dice_coefficient, iou_score
from deepcontrail.core.splits import is_sequence_stem, partition_stems

__version__ = version("deepcontrail-core")

__all__ = [
    "BandChannel",
    "BandManifest",
    "DatasetConfig",
    "DatasetContractError",
    "DatasetInfo",
    "FalseColorChannel",
    "FalseColorConfig",
    "__version__",
    "check_false_color",
    "dice_coefficient",
    "false_color_from_spectral",
    "iou_score",
    "is_sequence_stem",
    "load_spectral_false_color",
    "partition_stems",
    "read_band_manifest",
    "read_dataset_info",
]
