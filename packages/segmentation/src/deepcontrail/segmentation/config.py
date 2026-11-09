"""Pydantic configuration of the segmentation data pipeline.

Uses only pydantic so that ``import deepcontrail.segmentation`` stays torch-free.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from deepcontrail.core.config import DatasetConfig


class NormalizeConfig(BaseModel):
    """Channel-wise normalization statistics.

    Defaults to ImageNet mean/std, the standard choice when fine-tuning
    pretrained encoders (ResNet, EfficientNet, etc.).

    For model-specific normalization, training pipelines should query
    ``smp.encoders.get_preprocessing_params(encoder_name)`` and pass the
    resulting ``mean``/``std`` here to match the encoder's expectations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class AugmentationConfig(BaseModel):
    """Spatial and photometric augmentation probabilities.

    Each probability defaults to ``0.0`` (disabled) except the three spatial
    flips/rotates, so existing YAML configs without the new fields keep their
    previous behaviour. Photometric transforms (``gaussian_blur``,
    ``brightness_contrast``) apply to the image only — albumentations
    co-transforms the mask for the spatial ones.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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


class SegmentationDataConfig(DatasetConfig):
    """Complete segmentation data pipeline configuration.

    Extends ``DatasetConfig`` (root, split, image construction, pixel shift and
    train/val partition) with the label, resize, augmentation, normalization
    and DataLoader parameters (batch_size, num_workers), so one flat ``data:``
    YAML section reproduces a dataset setup.

    Parameters
    ----------
    mask_type
        Which label artifact to load. ``"binary"`` reads ``semantic_masks/``
        PNGs (strict consensus ground truth). ``"probability"`` reads
        ``probability_masks/`` and returns float32 in ``[0, 1]`` for
        soft-label training.
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

    # Unlike the shared core DatasetConfig, the segmentation data section is complete here, so a
    # misspelt key is an error instead of a silently ignored setting.
    model_config = ConfigDict(frozen=True, extra="forbid")

    mask_type: Literal["binary", "probability"] = "binary"
    image_size: Annotated[int, Field(gt=0)] = 256
    batch_size: Annotated[int, Field(gt=0)] = 16
    num_workers: Annotated[int, Field(ge=0)] = 4
    augmentation: AugmentationConfig | None = None
    normalize: NormalizeConfig = NormalizeConfig()
