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
    model = arch_cls(**kwargs)
    if arch_cls is smp.UnetPlusPlus and model.encoder.out_channels[1] == 0:
        _add_half_scale_stem(model, kwargs)
    return model  # type: ignore[no-any-return]


def _add_half_scale_stem(model: smp.UnetPlusPlus, kwargs: dict[str, Any]) -> None:
    """Give a Unet++ whose encoder starts at 1/4 scale a conv stem at 1/2 scale.

    Without it the nested decoder nodes at 1/2 scale have 0 channels and the model cannot run
    (transformer-style encoders such as CAFormer). The decoder is rebuilt with the same options,
    and it and the stem get smp's decoder initialisation, as a plain smp Unet++ decoder does.
    """
    from segmentation_models_pytorch.base.initialization import initialize_decoder
    from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

    from deepcontrail.segmentation.models.stem import HalfScaleStemEncoder

    defaults = {
        name: p.default
        for name, p in inspect.signature(smp.UnetPlusPlus.__init__).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    options = defaults | kwargs
    model.encoder = HalfScaleStemEncoder(model.encoder)
    model.decoder = UnetPlusPlusDecoder(
        encoder_channels=model.encoder.out_channels,
        decoder_channels=options["decoder_channels"],
        n_blocks=options["encoder_depth"],
        use_norm=options["decoder_use_norm"],
        attention_type=options["decoder_attention_type"],
        interpolation_mode=options["decoder_interpolation"],
    )
    initialize_decoder(model.decoder)
    initialize_decoder(model.encoder.stem)


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
