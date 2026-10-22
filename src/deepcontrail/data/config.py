"""Pydantic configuration models for the data pipeline.

Serializable via ``model_dump()`` → YAML/JSON for experiment logging.
Uses only pydantic + pyyaml so that ``import deepcontrail.data`` stays torch-free.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

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


SpectralBand = Literal[
    "data_10um",
    "data_11um",
    "data_12um",
    "brightness_temperature_difference",
    "cloud_top",
    "co2",
    "low_vapor",
    "mid_vapor",
    "upper_vapor",
    "ozone",
]
"""OpenContrails band key.

``data_10um`` is ABI band 13 (10.3 µm); ``cloud_top`` is ABI band 11 (8.4 µm).
"""

SPECTRAL_BAND_KEYS: tuple[str, ...] = get_args(SpectralBand)
"""Channel order of every ``spectral_bands/{split}/{stem}.npy`` file, shape ``(H, W, 10)``."""


@dataclasses.dataclass(frozen=True)
class SpectralBandInfo:
    """What one OpenContrails band key holds.

    Parameters
    ----------
    abi_band
        GOES-16 ABI band number, or ``None`` for a derived channel.
    wavelength_um
        ABI central wavelength in µm, or ``None`` for a derived channel.
    name
        NOAA name of the band (or a description of the derived channel).
    derived_from
        ``(a, b)`` when the channel is ``a - b`` of two other keys.
    """

    abi_band: int | None
    wavelength_um: float | None
    name: str
    derived_from: tuple[str, str] | None = None


# Verified by exact match of OpenContrails TFRecord keys against Kaggle's labelled band_XX files.
SPECTRAL_BAND_INFO: dict[str, SpectralBandInfo] = {
    "data_10um": SpectralBandInfo(13, 10.33, "Clean IR longwave window"),
    "data_11um": SpectralBandInfo(14, 11.19, "IR longwave window"),
    "data_12um": SpectralBandInfo(15, 12.27, "Dirty IR longwave window"),
    "brightness_temperature_difference": SpectralBandInfo(
        None, None, "Split-window difference", derived_from=("data_11um", "data_12um")
    ),
    "cloud_top": SpectralBandInfo(11, 8.44, "Cloud-top phase"),
    "co2": SpectralBandInfo(16, 13.27, "CO2 longwave IR"),
    "low_vapor": SpectralBandInfo(10, 7.34, "Lower-level tropospheric water vapor"),
    "mid_vapor": SpectralBandInfo(9, 6.93, "Mid-level tropospheric water vapor"),
    "upper_vapor": SpectralBandInfo(8, 6.19, "Upper-level tropospheric water vapor"),
    "ozone": SpectralBandInfo(12, 9.61, "Ozone"),
}
"""Band number, wavelength and name of each ``SPECTRAL_BAND_KEYS`` entry."""


class FalseColorChannel(BaseModel):
    """One composite channel: a band, or the difference of two bands, scaled by ``bounds``.

    ``bands=(a,)`` gives ``a``; ``bands=(a, b)`` gives ``a - b``. The value is
    mapped linearly from ``bounds = (lo, hi)`` to ``[0, 1]`` and clipped.
    """

    model_config = ConfigDict(frozen=True)

    bands: tuple[SpectralBand] | tuple[SpectralBand, SpectralBand]
    bounds: tuple[float, float]

    @model_validator(mode="after")
    def _validate_bounds(self) -> FalseColorChannel:
        lo, hi = self.bounds
        if not lo < hi:
            msg = f"bounds must be increasing (lo < hi), got {self.bounds}"
            raise ValueError(msg)
        return self


class FalseColorConfig(BaseModel):
    """Bands and bounds of the 3-channel composite built from ``spectral_bands/``.

    The default is the recipe every model before this option was trained on:
    the ash RGB layout but with ``data_10um`` (ABI band 13) in green. The
    standard ash RGB uses ``cloud_top`` (ABI band 11) there instead.
    """

    model_config = ConfigDict(frozen=True)

    channels: tuple[FalseColorChannel, FalseColorChannel, FalseColorChannel] = (
        FalseColorChannel(bands=("data_12um", "data_11um"), bounds=(-4.0, 2.0)),
        FalseColorChannel(bands=("data_11um", "data_10um"), bounds=(-4.0, 5.0)),
        FalseColorChannel(bands=("data_11um",), bounds=(243.0, 303.0)),
    )


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
    image_source
        ``"jpeg"`` reads pre-rendered ``images_semantic/`` JPEGs. ``"spectral"``
        reads ``spectral_bands/`` and builds the composite with ``false_color``.
    false_color
        Bands and bounds of the composite when ``image_source="spectral"``.
        Defaults to the band13 legacy recipe.
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
    false_color: FalseColorConfig = FalseColorConfig()
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
