"""Tests for deepcontrail.segmentation.models.temporal — late temporal fusion."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


def _frames(seed: int) -> torch.Tensor:
    """Build a single random 3-channel 32x32 frame from a seeded generator."""
    import torch

    return torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def _resnet18_unet() -> nn.Module:
    """Build a random-initialised 3-channel resnet18 U-Net for the fusion tests."""
    from deepcontrail.segmentation.models import ModelConfig, build_model

    return build_model(
        ModelConfig(architecture="Unet", encoder_name="resnet18", encoder_weights=None)
    )


def test_temporal_fusion_encoder_starts_as_the_encoder_on_the_centre_frame(full_precision_matmul):
    import torch

    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    torch.manual_seed(0)
    encoder = _resnet18_unet().encoder.eval()
    wrapper = TemporalFusionEncoder(copy.deepcopy(encoder), frames=3).eval()

    f1, f2, f3 = _frames(1), _frames(2), _frames(3)
    stacked = torch.cat([f1, f2, f3], dim=1)

    with torch.no_grad():
        fused = wrapper(stacked)
        single = encoder(f2)

    assert len(fused) == len(single), "the wrapper returns one map per encoder scale"
    for index, (fused_scale, single_scale) in enumerate(zip(fused, single, strict=True)):
        assert torch.allclose(fused_scale, single_scale, atol=1e-6), (
            f"scale {index} must equal the encoder's on the centre frame before training"
        )


@pytest.mark.parametrize("frames", [1, 2, 4])
def test_temporal_fusion_encoder_rejects_an_even_or_single_frame_count(frames):
    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    encoder = _resnet18_unet().encoder

    with pytest.raises(ValueError, match="odd"):
        TemporalFusionEncoder(copy.deepcopy(encoder), frames=frames)


def test_temporal_fusion_encoder_rejects_a_wrong_channel_count():
    import torch

    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    wrapper = TemporalFusionEncoder(_resnet18_unet().encoder, frames=3)

    with pytest.raises(ValueError, match="stacked frames"):
        wrapper(torch.rand(1, 6, 32, 32))


def test_temporal_fusion_neighbour_weights_get_gradients():
    import torch
    import torch.nn as nn

    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    wrapper = TemporalFusionEncoder(_resnet18_unet().encoder, frames=3).train()
    x = torch.rand(2, 9, 32, 32, generator=torch.Generator().manual_seed(0))

    sum(f.mean() for f in wrapper(x)[1:]).backward()

    for fuse in wrapper.fuse:
        if not isinstance(fuse, nn.Conv2d):
            continue
        channels = fuse.weight.shape[0]
        neighbour_grad = torch.cat(
            [fuse.weight.grad[:, :channels], fuse.weight.grad[:, 2 * channels :]], dim=1
        )
        assert torch.count_nonzero(neighbour_grad) > 0, (
            "the neighbour frames start at zero but their weights must receive a gradient to learn"
        )


def test_build_model_features_fusion_wraps_the_encoder():
    import torch

    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    model = build_model(
        ModelConfig(
            architecture="Unet",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=9,
            temporal_fusion="features",
        )
    )

    assert isinstance(model.encoder, TemporalFusionEncoder), "features fusion wraps the encoder"
    assert model(torch.rand(2, 9, 32, 32)).shape == (2, 1, 32, 32), "the wrapped model still runs"


def test_build_model_features_fusion_unetplusplus_with_stem():
    import torch

    from deepcontrail.segmentation.models import ModelConfig, build_model

    model = build_model(
        ModelConfig(
            architecture="UnetPlusPlus",
            encoder_name="tu-convnext_atto",
            encoder_weights=None,
            in_channels=9,
            temporal_fusion="features",
        )
    )

    assert model(torch.rand(1, 9, 32, 32)).shape == (1, 1, 32, 32), (
        "a features-fusion Unet++ with a stem still runs on the stacked frames"
    )


def test_build_model_input_fusion_keeps_one_encoder_on_the_stacked_channels():
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    model = build_model(
        ModelConfig(
            architecture="Unet", encoder_name="resnet18", encoder_weights=None, in_channels=9
        )
    )

    assert not isinstance(model.encoder, TemporalFusionEncoder), "the default fusion does not wrap"
    assert model.encoder.conv1.weight.shape[1] == 9, "the first conv takes every stacked channel"


def test_model_config_temporal_fusion_defaults_to_input():
    from deepcontrail.segmentation.models.config import ModelConfig

    assert ModelConfig().temporal_fusion == "input"


@pytest.mark.parametrize("in_channels", [3, 6, 10, 12])
def test_model_config_features_fusion_rejects_in_channels_that_are_not_odd_frames(in_channels):
    from pydantic import ValidationError

    from deepcontrail.segmentation.models.config import ModelConfig

    with pytest.raises(ValidationError, match="temporal_fusion"):
        ModelConfig(in_channels=in_channels, temporal_fusion="features")


def test_model_config_features_fusion_round_trips_through_a_dict():
    from deepcontrail.segmentation.models.config import ModelConfig

    cfg = ModelConfig(in_channels=9, temporal_fusion="features")

    assert ModelConfig.from_dict(cfg.model_dump()) == cfg, "the fusion mode survives a dump/reload"
