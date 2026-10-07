"""Pydantic configuration models for the data pipeline.

Serializable via ``model_dump()`` → YAML/JSON for experiment logging.
Uses only pydantic + pyyaml so that ``import deepcontrail.data`` stays torch-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class NormalizeConfig(BaseModel):
    """Channel-wise normalization statistics.

    Defaults to ImageNet mean/std, the standard choice when fine-tuning
    pretrained encoders (ResNet, EfficientNet, etc.).

    For model-specific normalization, training pipelines should query
    ``smp.encoders.get_preprocessing_params(encoder_name)`` and pass the
    resulting ``mean``/``std`` here to match the encoder's expectations.
    """

    model_config = ConfigDict(frozen=True)

    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)


class AugmentationConfig(BaseModel):
    """Spatial augmentation probabilities for the training pipeline."""

    model_config = ConfigDict(frozen=True)

    horizontal_flip: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    vertical_flip: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    random_rotate_90: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5


class DataConfig(BaseModel):
    """Complete data pipeline configuration.

    Captures every parameter needed to reproduce a dataset setup:
    root path, split, seed, augmentation, normalization, and DataLoader
    parameters (batch_size, num_workers).

    Parameters
    ----------
    root
        Path to the dataset root (contains ``images_semantic/`` and
        ``semantic_masks/`` subdirectories).
    split
        ``"train"``, ``"val"``, or ``"test"``.
    pixel_shift
        Sub-pixel translation applied to images at load time to correct a
        labeling misalignment in the GOES-16 dataset. Default 0.0 (no shift).
    seed
        RNG seed for reproducible augmentation.
    image_size
        Expected spatial size of images (height and width).
    batch_size
        Batch size for the DataLoader (not used by the Dataset itself).
    num_workers
        Number of DataLoader workers (not used by the Dataset itself).
    augmentation
        Augmentation config for training, or ``None`` for no augmentation.
    normalize
        Channel-wise normalization statistics.
    """

    model_config = ConfigDict(frozen=True)

    root: str
    split: Literal["train", "val", "test"] = "train"
    pixel_shift: Annotated[float, Field(ge=0.0)] = 0.0
    seed: Annotated[int, Field(ge=0)] = 42
    image_size: Annotated[int, Field(gt=0)] = 256
    batch_size: Annotated[int, Field(gt=0)] = 16
    num_workers: Annotated[int, Field(ge=0)] = 4
    augmentation: AugmentationConfig | None = None
    normalize: NormalizeConfig = NormalizeConfig()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        """Build a ``DataConfig`` from a plain dictionary.

        Pydantic handles nested dict → sub-model parsing automatically.
        """
        return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataConfig:
        """Load a ``DataConfig`` from the ``data:`` section of a YAML file."""
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw["data"])
