"""Tests for D4 test-time augmentation."""

from __future__ import annotations

import torch


def test_d4_all_eight_elements() -> None:
    from deepcontrail.training.tta import D4_TRANSFORMS

    assert len(D4_TRANSFORMS) == 8, "D4 group has exactly 8 elements"


def test_d4_augment_deaugment_roundtrip() -> None:
    from deepcontrail.training.tta import D4_TRANSFORMS

    torch.manual_seed(42)
    x = torch.randn(1, 1, 8, 8)

    for i, (augment, deaugment) in enumerate(D4_TRANSFORMS):
        restored = deaugment(augment(x))
        assert torch.allclose(x, restored, atol=1e-6), (
            f"D4 element {i}: augment then deaugment must recover original"
        )


def test_d4_transforms_produce_different_outputs() -> None:
    from deepcontrail.training.tta import D4_TRANSFORMS

    torch.manual_seed(42)
    x = torch.randn(1, 1, 8, 8)

    outputs = [augment(x) for augment, _ in D4_TRANSFORMS]
    unique = len({o.data_ptr() for o in outputs})

    assert unique >= 2, "at least some D4 transforms should produce different tensors"


def test_tta_predict_returns_correct_shape() -> None:
    from deepcontrail.training.tta import tta_predict

    class DummyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])

    model = DummyModel().eval()
    images = torch.randn(2, 3, 16, 16)

    probs = tta_predict(model, images)

    assert probs.shape == (2, 16, 16), f"expected (2, 16, 16), got {probs.shape}"
    assert probs.dtype == torch.float32


def test_tta_predict_values_in_zero_one() -> None:
    from deepcontrail.training.tta import tta_predict

    class RandModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch.manual_seed(0)
            return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])

    model = RandModel().eval()
    images = torch.randn(2, 3, 8, 8)

    probs = tta_predict(model, images)

    assert probs.min() >= 0.0, "sigmoid probs must be >= 0"
    assert probs.max() <= 1.0, "sigmoid probs must be <= 1"


def test_tta_predict_identity_model_symmetric_input() -> None:
    from deepcontrail.training.tta import tta_predict

    class IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :1, :, :]

    model = IdentityModel().eval()
    sym = torch.ones(1, 3, 8, 8) * 0.5

    probs = tta_predict(model, sym)

    assert torch.allclose(probs, torch.sigmoid(torch.tensor(0.5)).expand_as(probs), atol=1e-5), (
        "TTA on D4-symmetric input with identity model should give uniform sigmoid(0.5)"
    )
