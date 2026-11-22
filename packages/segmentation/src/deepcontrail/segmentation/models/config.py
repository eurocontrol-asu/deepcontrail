"""Pydantic configuration model for segmentation architectures.

Serializable via ``model_dump()`` for MLflow logging.
Uses only pydantic + pyyaml — no torch dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Segmentation model configuration.

    Captures every parameter needed to reproduce a model architecture.
    Serializable via ``model_dump()`` for MLflow logging.

    Parameters
    ----------
    architecture
        Name of the smp architecture class (e.g. ``"Unet"``, ``"UnetPlusPlus"``).
    encoder_name
        Encoder backbone name. Supports ``tu-`` prefix for timm encoders
        (passed through to smp unchanged).
    encoder_weights
        Pretrained weights identifier, or ``None`` for random init.
    in_channels
        Number of input channels (3 for RGB).
    temporal_fusion
        How a temporal input (``in_channels = 3 x F``, ``F = 2k+1`` stacked frames; see the data
        option ``context_frames``) is combined. ``"input"`` (default): the frames are the channels
        of one image and the first convolution is widened. ``"features"``: every frame goes through
        the same 3-channel encoder and the frames' features are merged at every scale (late fusion,
        ``models/temporal.py``); ``in_channels`` must then be ``3 x F`` with ``F`` odd and >= 3.
    classes
        Number of output classes (1 for binary segmentation).
    decoder_attention_type
        Decoder attention type (e.g. ``"scse"``). Only used by architectures
        that accept it (Unet, UnetPlusPlus). Silently stripped for others.
    decoder_use_norm
        Decoder normalization. Only used by architectures that accept it.
        Silently stripped for others.
    extra_kwargs
        Additional keyword arguments passed directly to the smp constructor.
        Use for architecture-specific parameters (e.g. ``decoder_channels``,
        ``encoder_depth``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    architecture: str = "Unet"
    encoder_name: str = "efficientnet-b0"
    encoder_weights: str | None = "imagenet"
    in_channels: Annotated[int, Field(gt=0)] = 3
    temporal_fusion: Literal["input", "features"] = "input"
    classes: Annotated[int, Field(gt=0)] = 1
    decoder_attention_type: str | None = None
    decoder_use_norm: bool | str = "batchnorm"
    extra_kwargs: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check_temporal_fusion(self) -> ModelConfig:
        """Feature fusion needs an odd number (at least 3) of stacked 3-channel frames."""
        if self.temporal_fusion == "features":
            frames, rest = divmod(self.in_channels, 3)
            if rest or frames < 3 or frames % 2 == 0:
                msg = (
                    'temporal_fusion="features" needs in_channels = 3 x F with F an odd number of '
                    f"frames >= 3 (9, 15, ...); got in_channels={self.in_channels}"
                )
                raise ValueError(msg)
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Build a ``ModelConfig`` from a plain dictionary.

        Pydantic handles nested dict → sub-model parsing automatically.
        """
        return cls.model_validate(d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load a ``ModelConfig`` from the ``model:`` section of a YAML file."""
        with Path(path).open() as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw["model"])
