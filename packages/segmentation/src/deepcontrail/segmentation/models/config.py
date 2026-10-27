"""Pydantic configuration model for segmentation architectures.

Serializable via ``model_dump()`` for MLflow logging.
Uses only pydantic + pyyaml — no torch dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(frozen=True)

    architecture: str = "Unet"
    encoder_name: str = "efficientnet-b0"
    encoder_weights: str | None = "imagenet"
    in_channels: Annotated[int, Field(gt=0)] = 3
    classes: Annotated[int, Field(gt=0)] = 1
    decoder_attention_type: str | None = None
    decoder_use_norm: bool | str = "batchnorm"
    extra_kwargs: dict[str, Any] = {}

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
