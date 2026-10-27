"""Semantic segmentation of contrails in satellite imagery.

The data pipeline imported here is torch-free; ``models``, ``losses`` and
``training`` are subpackages that need the training stack.
"""

from importlib.metadata import version

from deepcontrail.segmentation.config import (
    AugmentationConfig,
    NormalizeConfig,
    SegmentationDataConfig,
)
from deepcontrail.segmentation.dataset import SegmentationDataset

__version__ = version("deepcontrail-segmentation")

__all__ = [
    "AugmentationConfig",
    "NormalizeConfig",
    "SegmentationDataConfig",
    "SegmentationDataset",
    "__version__",
]
