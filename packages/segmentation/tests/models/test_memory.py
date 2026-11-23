"""Tests for deepcontrail.segmentation.models.memory — track memory injection."""

from __future__ import annotations

import copy

import pytest


def _resnet18_unet() -> object:
    """Build a random-initialised 3-channel resnet18 U-Net for the memory tests."""
    from deepcontrail.segmentation.models import ModelConfig, build_model

    return build_model(
        ModelConfig(architecture="Unet", encoder_name="resnet18", encoder_weights=None)
    )


def test_track_memory_encoder_starts_as_the_encoder(full_precision_matmul):
    import torch

    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.models.memory import TrackMemoryEncoder

    torch.manual_seed(0)
    encoder = build_model(
        ModelConfig(architecture="Unet", encoder_name="resnet18", encoder_weights=None)
    ).encoder.eval()
    wrapper = TrackMemoryEncoder(copy.deepcopy(encoder), image_channels=3).eval()

    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 3, 32, 32, generator=generator)
    memory = torch.rand(1, 1, 32, 32, generator=generator)

    with torch.no_grad():
        injected = wrapper(torch.cat([image, memory], dim=1))
        single = encoder(image)

    assert len(injected) == len(single), "the wrapper returns one map per encoder scale"
    for index, (injected_scale, single_scale) in enumerate(zip(injected, single, strict=True)):
        assert torch.allclose(injected_scale, single_scale, atol=1e-6), (
            f"scale {index} must equal the encoder's before the memory is trained"
        )


def test_track_memory_encoder_rejects_a_wrong_channel_count():
    import torch

    from deepcontrail.segmentation.models.memory import TrackMemoryEncoder

    wrapper = TrackMemoryEncoder(_resnet18_unet().encoder, image_channels=3)

    with pytest.raises(ValueError, match="memory channel"):
        wrapper(torch.rand(1, 3, 32, 32))


def test_track_memory_projection_learns_from_zero():
    import torch
    import torch.nn as nn

    from deepcontrail.segmentation.models.memory import TrackMemoryEncoder

    wrapper = TrackMemoryEncoder(_resnet18_unet().encoder, image_channels=3).train()
    x = torch.rand(2, 4, 32, 32, generator=torch.Generator().manual_seed(0))

    sum(f.mean() for f in wrapper(x)[1:]).backward()

    for inject in wrapper.inject:
        if not isinstance(inject, nn.Sequential):
            continue
        weight_grad = inject[-1].weight.grad
        assert weight_grad is not None, "the last convolution must receive a gradient"
        assert torch.count_nonzero(weight_grad) > 0, (
            "the memory starts at zero but its last convolution must receive a gradient to learn"
        )


def test_track_memory_changes_the_output_once_trained():
    import torch
    import torch.nn as nn

    from deepcontrail.segmentation.models.memory import TrackMemoryEncoder

    wrapper = TrackMemoryEncoder(_resnet18_unet().encoder, image_channels=3).train()
    image = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        for inject in wrapper.inject:
            if isinstance(inject, nn.Sequential):
                inject[-1].weight.fill_(0.01)

    with torch.no_grad():
        with_memory = wrapper(torch.cat([image, torch.ones(2, 1, 32, 32)], dim=1))[1]
        without_memory = wrapper(torch.cat([image, torch.zeros(2, 1, 32, 32)], dim=1))[1]

    assert not torch.allclose(with_memory, without_memory), (
        "a trained memory projection must reach the encoder features"
    )


def test_build_model_track_memory_wraps_the_encoder():
    import torch

    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.models.memory import TrackMemoryEncoder

    model = build_model(
        ModelConfig(
            architecture="Unet", encoder_name="resnet18", encoder_weights=None, track_memory=True
        )
    )

    assert isinstance(model.encoder, TrackMemoryEncoder), "track_memory wraps the encoder"
    assert model(torch.rand(2, 4, 32, 32)).shape == (2, 1, 32, 32), (
        "a memory model takes the image channels plus the memory channel and still runs"
    )


def test_build_model_track_memory_on_a_temporal_model():
    import torch

    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.models.memory import TrackMemoryEncoder
    from deepcontrail.segmentation.models.temporal import TemporalFusionEncoder

    model = build_model(
        ModelConfig(
            architecture="UnetPlusPlus",
            encoder_name="tu-convnext_atto",
            encoder_weights=None,
            in_channels=9,
            temporal_fusion="features",
            track_memory=True,
        )
    )

    assert isinstance(model.encoder, TrackMemoryEncoder), "track_memory is the outermost wrapper"
    assert isinstance(model.encoder.encoder, TemporalFusionEncoder), (
        "a memory temporal model wraps the fusion encoder"
    )
    assert model(torch.rand(1, 10, 32, 32)).shape == (1, 1, 32, 32), (
        "a memory temporal model takes the stacked frames plus the memory channel"
    )


def test_model_config_track_memory_defaults_to_false():
    from deepcontrail.segmentation.models.config import ModelConfig

    assert ModelConfig().track_memory is False
