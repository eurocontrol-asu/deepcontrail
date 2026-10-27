"""Semantic segmentation of contrails (Phase 1: GOES-16).

The data pipeline imported here is torch-free; ``models``, ``losses`` and
``training`` are subpackages that need the training stack.
"""

from deepcontrail.segmentation.config import (
    AugmentationConfig,
    NormalizeConfig,
    SegmentationDataConfig,
)
from deepcontrail.segmentation.dataset import GOESSemanticDataset

GOESDataset = GOESSemanticDataset

__all__ = [
    "AugmentationConfig",
    "GOESDataset",
    "GOESSemanticDataset",
    "NormalizeConfig",
    "SegmentationDataConfig",
]
