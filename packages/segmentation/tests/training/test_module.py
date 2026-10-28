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


def test_training_step_logs_the_epoch_mean_and_the_per_step_loss(monkeypatch) -> None:
    from deepcontrail.segmentation.losses import LossConfig, build_loss

    module = SegmentationModule(
        model=_DummySeg(),
        loss_fn=build_loss(LossConfig(name="bce")),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=8,
    )
    logged: dict[str, dict] = {}
    monkeypatch.setattr(module, "log", lambda name, value, **kw: logged.setdefault(name, kw))
    batch = (torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.long))

    module.training_step(batch, 0)

    assert logged["train_loss"]["on_epoch"] is True, "train_loss stays the per-epoch mean"
    assert logged["train_loss"]["on_step"] is False
    assert logged["train_loss_step"]["on_step"] is True, "the per-step loss gives the fine curve"
    assert logged["train_loss_step"]["on_epoch"] is False
