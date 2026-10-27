"""Semantic segmentation of contrails in satellite imagery.

The data pipeline imported here is torch-free; ``models``, ``losses`` and
``training`` are subpackages that need the training stack. ``train`` (from
``training``) is loaded on first access, so this import stays light.
"""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from deepcontrail.segmentation.config import (
    AugmentationConfig,
    NormalizeConfig,
    SegmentationDataConfig,
)
from deepcontrail.segmentation.dataset import SegmentationDataset

__version__ = version("deepcontrail-segmentation")

if TYPE_CHECKING:
    from deepcontrail.segmentation.training.run import train


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Load ``train`` lazily: it imports torch and Lightning."""
    if name == "train":
        from deepcontrail.segmentation.training.run import train

        return train
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "AugmentationConfig",
    "NormalizeConfig",
    "SegmentationDataConfig",
    "SegmentationDataset",
    "__version__",
    "train",
]
