"""Pydantic configuration of the segmentation data pipeline.

Uses only pydantic so that ``import deepcontrail.segmentation`` stays torch-free.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class MemoryCorruptionConfig(BaseModel):
    """Training-time corruption of the memory, so the previous label looks like a prediction.

    At inference the memory is the model's own prediction for the previous frame: incomplete,
    shifted and soft. Each training item draws, in order: an empty memory (``empty_rate``); else
    contrails removed one by one (``drop_rate``), a random shift (``max_shift``), a Gaussian blur
    (``max_blur_sigma``) and a scaling of the values (``min_scale``).

    Parameters
    ----------
    empty_rate
        Probability that the whole memory is zeros (a sequence start, or a lost memory).
    drop_rate
        Probability that each contrail (connected region of the memory above zero) is removed.
    max_shift
        The memory moves by a random whole number of pixels in ``[-max_shift, max_shift]`` along
        each axis (contrails drift between frames); zeros enter at the edges.
    max_blur_sigma
        Gaussian blur with a sigma drawn uniformly in ``[0, max_blur_sigma]`` pixels.
    min_scale
        The values are multiplied by a factor drawn uniformly in ``[min_scale, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    empty_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.25
    drop_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    max_shift: Annotated[int, Field(ge=0)] = 8
    max_blur_sigma: Annotated[float, Field(ge=0.0)] = 1.5
    min_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 0.6


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
    crop_size
        Side of the square window the model sees, in native pixels. Set it for scenes larger
        than the training window: training reads one random crop per scene and item, validation
        and test read the non-overlapping tiles covering each scene (the last row/column of
        tiles is moved back to end at the scene edge). ``None`` (default) uses whole scenes.
    crops_per_scene
        With ``crop_size``, the number of independent random crops each training scene yields
        per epoch (the epoch length is scenes x crops_per_scene). Each item draws a fresh
        random origin, so ``crops_per_scene = 4`` makes an epoch see four times as many steps
        over the same data. Default 1 is today's behaviour. Ignored for validation and test,
        which read tiles. Requires ``crop_size``: a value > 1 without it is an error.
    train_tile_stride
        With ``crop_size``, training reads the fixed tiles of each scene at this stride (e.g.
        ``crop_size // 2`` for windows overlapping by half, as prediction does) instead of random
        crops; the epoch is every tile of every scene, shuffled by the DataLoader. ``None``
        (default) keeps random crops. Requires ``crop_size``; cannot be combined with
        ``crops_per_scene > 1``; must be ``<= crop_size``.
    context_frames
        ``k`` frames before and after each scene are stacked with it, giving
        ``3*(2k+1)`` channels in time order; needs ``metadata/neighbours.json`` (see
        :func:`deepcontrail.core.contract.read_neighbours`), and missing neighbours repeat the
        nearest frame. Default 0 keeps the plain 3-channel image.
    memory_prior
        The track-memory channel appended after the image channels (for a model with
        ``track_memory``). ``"none"`` (default): no extra channel. ``"labels"``: the previous
        frame's label at the same window (the soft label with ``mask_type: probability``, the
        consensus mask otherwise), zeros at a sequence start; needs ``metadata/neighbours.json``.
        ``"empty"``: zeros; validation and test use it during training, so no label of the
        previous frame leaks into model selection.
    memory_corruption
        With ``memory_prior: labels``, the training-time corruption of the memory (see
        ``MemoryCorruptionConfig``); ``None`` keeps the clean previous label.
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
    crop_size: Annotated[int, Field(gt=0)] | None = None
    crops_per_scene: Annotated[int, Field(ge=1)] = 1
    train_tile_stride: Annotated[int, Field(gt=0)] | None = None
    context_frames: Annotated[int, Field(ge=0)] = 0
    memory_prior: Literal["none", "labels", "empty"] = "none"
    memory_corruption: MemoryCorruptionConfig | None = None
    batch_size: Annotated[int, Field(gt=0)] = 16
    num_workers: Annotated[int, Field(ge=0)] = 4
    augmentation: AugmentationConfig | None = None
    normalize: NormalizeConfig = NormalizeConfig()

    @model_validator(mode="after")
    def _crops_per_scene_needs_crop_size(self) -> Self:
        """Reject ``crops_per_scene`` > 1 when no ``crop_size`` is set."""
        if self.crops_per_scene > 1 and self.crop_size is None:
            msg = "crops_per_scene needs crop_size"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _train_tile_stride_needs_crop_size(self) -> Self:
        """Validate ``train_tile_stride`` against ``crop_size`` and ``crops_per_scene``."""
        if self.train_tile_stride is None:
            return self
        if self.crop_size is None:
            msg = "train_tile_stride needs crop_size"
            raise ValueError(msg)
        if self.crops_per_scene > 1:
            msg = "train_tile_stride cannot be combined with crops_per_scene > 1"
            raise ValueError(msg)
        if self.train_tile_stride > self.crop_size:
            msg = "train_tile_stride must be <= crop_size"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _memory_corruption_needs_labels(self) -> Self:
        """Reject a memory corruption without a label memory to corrupt."""
        if self.memory_corruption is not None and self.memory_prior != "labels":
            msg = 'memory_corruption needs memory_prior: "labels"'
            raise ValueError(msg)
        return self
