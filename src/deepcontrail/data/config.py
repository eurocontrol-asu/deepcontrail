"""Pydantic configuration models for the data pipeline.

Serializable via ``model_dump()`` → YAML/JSON for experiment logging.
Uses only pydantic + pyyaml so that ``import deepcontrail.data`` stays torch-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    """Spatial and photometric augmentation probabilities.

    Each probability defaults to ``0.0`` (disabled) except the three spatial
    flips/rotates, so existing YAML configs without the new fields keep their
    previous behaviour. Photometric transforms (``gaussian_blur``,
    ``brightness_contrast``) apply to the image only — albumentations
    co-transforms the mask for the spatial ones.
    """

    model_config = ConfigDict(frozen=True)

    # Spatial — applied to both image and mask.
    horizontal_flip: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    vertical_flip: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    random_rotate_90: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    shift_scale_rotate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    shift_limit: float = 0.0625
    scale_limit: float = 0.1
    rotate_limit: int = 45
    # Photometric / pixel-level — image only.
    gaussian_blur: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    blur_limit: tuple[int, int] = (3, 7)
    brightness_contrast: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


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
        ``"train"``, ``"val"``, or ``"test"``.  Both ``"train"`` and ``"val"``
        read from the ``train/`` directory on disk; the dataset partitions stems
        deterministically using ``val_fraction``.  ``"test"`` reads from ``test/``.
    mask_type
        Which label artifact to load. ``"binary"`` reads ``semantic_masks/``
        PNGs (strict consensus ground truth). ``"probability"`` reads
        ``probability_masks/`` and returns float32 in ``[0, 1]`` for
        soft-label training.
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
    val_fraction
        Fraction of training stems held out for validation (seeded by ``seed``).
        Must be > 0 when ``fold`` is not set.  Ignored when ``fold`` is set
        (fold partitioning takes over).
    fold
        K-fold cross-validation fold index, or ``None`` to disable CV.
    num_folds
        Total number of CV folds. Only used when ``fold`` is set.
    """

    model_config = ConfigDict(frozen=True)

    root: str
    split: Literal["train", "val", "test"] = "train"
    mask_type: Literal["binary", "probability"] = "binary"
    image_source: Literal["jpeg", "spectral"] = "jpeg"
    pixel_shift: Annotated[float, Field(ge=0.0)] = 0.0
    seed: Annotated[int, Field(ge=0)] = 42
    image_size: Annotated[int, Field(gt=0)] = 256
    batch_size: Annotated[int, Field(gt=0)] = 16
    num_workers: Annotated[int, Field(ge=0)] = 4
    augmentation: AugmentationConfig | None = None
    normalize: NormalizeConfig = NormalizeConfig()
    val_fraction: Annotated[float, Field(default=0.1, ge=0.0, lt=1.0)] = 0.1
    fold: int | None = None
    num_folds: Annotated[int, Field(default=4, ge=2)] = 4

    @model_validator(mode="after")
    def _validate_split_strategy(self) -> DataConfig:
        """Validate the split configuration.

        - ``fold`` must be in ``[0, num_folds)`` when set.
        - ``val_fraction`` must be > 0 when ``fold`` is not set.
        """
        if self.fold is not None and not (0 <= self.fold < self.num_folds):
            msg = f"fold {self.fold} out of range [0, {self.num_folds})"
            raise ValueError(msg)
        if self.val_fraction <= 0 and self.fold is None:
            msg = "val_fraction must be > 0 (use fold for CV instead)"
            raise ValueError(msg)
        return self

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
