"""Model factory for constructing segmentation models from config."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import segmentation_models_pytorch as smp

if TYPE_CHECKING:
    import torch.nn as nn

    from deepcontrail.segmentation.models.config import ModelConfig


def build_model(config: ModelConfig) -> nn.Module:
    """Construct a segmentation model from config.

    Returns a plain ``torch.nn.Module``. The training script is responsible
    for wrapping it in a Lightning module, optimizer, and scheduler.

    Parameters
    ----------
    config
        Model configuration specifying architecture, encoder, and decoder
        parameters.

    Returns
    -------
    nn.Module
        The constructed segmentation model.

    Raises
    ------
    AttributeError
        If ``config.architecture`` is not a valid smp architecture name.
    """
    arch_cls = getattr(smp, config.architecture)
    kwargs = _build_smp_kwargs(config, arch_cls)
    return arch_cls(**kwargs)  # type: ignore[no-any-return]


def _build_smp_kwargs(config: ModelConfig, arch_cls: type[Any]) -> dict[str, Any]:
    """Assemble smp constructor kwargs, stripping unsupported decoder params."""
    sig_params = set(inspect.signature(arch_cls.__init__).parameters.keys())

    kwargs: dict[str, Any] = {
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "in_channels": config.in_channels,
        "classes": config.classes,
    }

    if "decoder_attention_type" in sig_params and config.decoder_attention_type is not None:
        kwargs["decoder_attention_type"] = config.decoder_attention_type

    if "decoder_use_norm" in sig_params:
        kwargs["decoder_use_norm"] = config.decoder_use_norm

    kwargs.update(config.extra_kwargs)

    return kwargs
