"""Data access for the DeepContrail segmentation datasets (Phase 1: GOES-16)."""

from deepcontrail.data.config import (
    SPECTRAL_BAND_INFO,
    SPECTRAL_BAND_KEYS,
    AugmentationConfig,
    DataConfig,
    FalseColorChannel,
    FalseColorConfig,
    NormalizeConfig,
    SpectralBandInfo,
)
from deepcontrail.data.goes_semantic import GOESSemanticDataset

GOESDataset = GOESSemanticDataset

__all__ = [
    "SPECTRAL_BAND_INFO",
    "SPECTRAL_BAND_KEYS",
    "AugmentationConfig",
    "DataConfig",
    "FalseColorChannel",
    "FalseColorConfig",
    "GOESDataset",
    "GOESSemanticDataset",
    "NormalizeConfig",
    "SpectralBandInfo",
]
