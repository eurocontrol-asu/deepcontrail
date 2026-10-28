"""Semantic segmentation of contrails in satellite imagery.

The data pipeline imported here is torch-free; ``models``, ``losses`` and
``training`` are subpackages that need the training stack. ``train`` (from
``training``), ``load_pretrained`` and ``predict`` (from ``pretrained``) are loaded
on first access, so this import stays light.
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
    from deepcontrail.segmentation.pretrained import load_pretrained, predict
    from deepcontrail.segmentation.training.run import train

_LAZY = {
    "train": "deepcontrail.segmentation.training.run",
    "load_pretrained": "deepcontrail.segmentation.pretrained",
    "predict": "deepcontrail.segmentation.pretrained",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Load ``train``, ``load_pretrained`` and ``predict`` lazily: they import torch."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "AugmentationConfig",
    "NormalizeConfig",
    "SegmentationDataConfig",
    "SegmentationDataset",
    "__version__",
    "load_pretrained",
    "predict",
    "train",
]
