"""Tests for SegmentationModule forward-pass resizing."""

from __future__ import annotations

import pytest
import torch

from deepcontrail.segmentation.training import SegmentationModule


class _DummySeg(torch.nn.Module):
    """A 1x1 convolution: output spatial size mirrors the input size."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _module(native_size: int = 256) -> SegmentationModule:
    model = _DummySeg()
    # A loss_fn is required by the module even though forward() ignores it.
    module = SegmentationModule(
        model=model,
        loss_fn=torch.nn.Identity(),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=native_size,
    )
    module.eval()
    return module


def test_forward_downscaled_output_is_upsampled_to_native():
    module = _module(native_size=256)

    out = module(torch.randn(2, 3, 384, 384))

    assert out.shape == (2, 1, 256, 256), f"got {out.shape}"


def test_forward_upscaled_output_is_downsampled_to_native():
    module = _module(native_size=256)

    out = module(torch.randn(2, 3, 512, 512))

    assert out.shape == (2, 1, 256, 256), f"got {out.shape}"


def test_forward_native_input_passes_through_unchanged():
    module = _module(native_size=256)

    out = module(torch.randn(2, 3, 256, 256))

    assert out.shape == (2, 1, 256, 256), f"got {out.shape}"


def test_forward_custom_native_size():
    module = _module(native_size=128)

    out = module(torch.randn(1, 3, 384, 384))

    assert out.shape == (1, 1, 128, 128), f"got {out.shape}"


def test_forward_output_shape_matches_mask_resolution():
    """The resized output is what loss/metrics (at mask resolution) consume."""
    module = _module(native_size=256)

    images = torch.randn(4, 3, 512, 512)  # image_size = 512
    masks = torch.randint(0, 2, (4, 256, 256))  # native mask resolution

    out = module(images)

    # Output (B,1,256,256) pairs with mask (B,256,256) for loss/metric calls.
    assert out.shape == (4, 1, 256, 256)
    assert masks.shape == (4, 256, 256)


def test_module_requires_the_native_size() -> None:
    import torch

    from deepcontrail.segmentation.training.module import SegmentationModule

    with pytest.raises(TypeError, match="native_size"):
        SegmentationModule(  # type: ignore[call-arg]
            model=torch.nn.Identity(),
            loss_fn=torch.nn.Identity(),
            lr=1e-3,
            weight_decay=0.0,
            scheduler="none",
            epochs=1,
        )
