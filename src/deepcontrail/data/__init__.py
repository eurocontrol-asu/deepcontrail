"""Data access for the DeepContrail segmentation datasets (Phase 1: GOES-16)."""

from deepcontrail.data.config import (
    SPECTRAL_BAND_KEYS,
    AugmentationConfig,
    DataConfig,
    FalseColorChannel,
    FalseColorConfig,
    NormalizeConfig,
)
from deepcontrail.data.goes_semantic import GOESSemanticDataset

GOESDataset = GOESSemanticDataset

__all__ = [
    "SPECTRAL_BAND_KEYS",
    "AugmentationConfig",
    "DataConfig",
    "FalseColorChannel",
    "FalseColorConfig",
    "GOESDataset",
    "GOESSemanticDataset",
    "NormalizeConfig",
]
