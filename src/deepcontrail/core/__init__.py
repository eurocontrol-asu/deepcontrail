"""Shared, torch-free foundations: dataset config, spectral bands, composites, splits, metrics."""

from deepcontrail.core.bands import SPECTRAL_BAND_INFO, SPECTRAL_BAND_KEYS, SpectralBandInfo
from deepcontrail.core.config import DatasetConfig
from deepcontrail.core.false_color import (
    FalseColorChannel,
    FalseColorConfig,
    false_color_from_spectral,
    load_spectral_false_color,
)
from deepcontrail.core.metrics import dice_coefficient, iou_score
from deepcontrail.core.splits import is_sequence_stem, partition_stems

__all__ = [
    "SPECTRAL_BAND_INFO",
    "SPECTRAL_BAND_KEYS",
    "DatasetConfig",
    "FalseColorChannel",
    "FalseColorConfig",
    "SpectralBandInfo",
    "dice_coefficient",
    "false_color_from_spectral",
    "iou_score",
    "is_sequence_stem",
    "load_spectral_false_color",
    "partition_stems",
]
