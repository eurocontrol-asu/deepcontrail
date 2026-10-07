"""Data access for the DeepContrail segmentation datasets (Phase 1: GOES-16)."""

from deepcontrail.data.config import AugmentationConfig, DataConfig, NormalizeConfig
from deepcontrail.data.goes_semantic import GOESSemanticDataset

GOESDataset = GOESSemanticDataset

__all__ = [
    "AugmentationConfig",
    "DataConfig",
    "GOESDataset",
    "GOESSemanticDataset",
    "NormalizeConfig",
]
