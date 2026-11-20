"""Tests for deepcontrail.segmentation.losses — soft clDice (Shit et al., CVPR 2021)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# soft_skeleton
# ---------------------------------------------------------------------------


def test_soft_skeleton_of_a_thick_line_is_thinner_and_inside_it():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.losses import soft_skeleton

    bar = torch.zeros(1, 1, 15, 15)
    bar[0, 0, 4:11, :] = 1.0  # 7-px-wide horizontal bar, rows 4..10, centre row 7
    skel = soft_skeleton(bar, iterations=3)

    assert skel.sum() < bar.sum(), "the skeleton must have less mass than the bar it thins"

    outside = torch.ones(1, 1, 15, 15, dtype=torch.bool)
    outside[0, 0, 4:11, :] = False
    assert skel[outside].max().item() == 0.0, "the skeleton must be zero outside the bar"

    centre = skel[0, 0, 7, :]
    assert centre.max().item() == skel.max().item(), "the skeleton must peak along the centre row"


def test_soft_skeleton_of_empty_input_is_zero():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.losses import soft_skeleton

    x = torch.zeros(1, 1, 8, 8)

    skel = soft_skeleton(x, iterations=3)

    assert torch.all(skel == 0.0), "the skeleton of an empty map must be zero"


# ---------------------------------------------------------------------------
# CLDiceLoss
# ---------------------------------------------------------------------------


def test_cldice_loss_perfect_prediction_is_near_zero():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.losses import CLDiceLoss

    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 7:9, 4:12] = 1.0  # a 2-px-wide line
    pred = target.clone()  # perfect prediction, probabilities in [0, 1]

    loss = CLDiceLoss(from_logits=False)(pred, target)

    assert loss.item() < 0.05, "the loss of a perfect prediction should be near zero"


def test_cldice_loss_missing_line_is_near_one():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.losses import CLDiceLoss

    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 16, 2:30] = 1.0  # a long, 1-px-wide line
    pred = torch.zeros_like(target)  # the line is entirely missing

    loss = CLDiceLoss(from_logits=False)(pred, target)

    assert loss.item() > 0.9, "the loss when the line is entirely missing should be near one"


def test_cldice_loss_accepts_soft_targets_and_logits():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.losses import CLDiceLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.rand(2, 1, 16, 16)  # soft targets in [0, 1]

    loss = CLDiceLoss()(logits, target)

    assert torch.isfinite(loss), "the loss must be finite for soft targets and logits"
    assert 0.0 <= loss.item() <= 1.0, "the clDice loss should lie in [0, 1]"

    loss.backward()

    assert logits.grad is not None, "the gradient must flow back to the logits"
    assert torch.isfinite(logits.grad).all(), "the gradient must be finite everywhere"


# ---------------------------------------------------------------------------
# CompositeLoss.from_config with a clDice component
# ---------------------------------------------------------------------------


def test_composite_with_cldice_component_builds_and_runs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.segmentation.losses import CompositeLoss, LossConfig

    cfg = LossConfig(
        name="composite",
        composite_weights={"bce": 1.0, "dice": 1.0, "cldice": 0.5},
    )
    loss_fn = CompositeLoss.from_config(cfg)

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.rand(2, 8, 8)  # soft target, (B, H, W) shape (channel axis added)

    loss = loss_fn(logits, target)

    assert loss.shape == (), "a composite loss with a clDice component should return a scalar"
    assert loss.grad_fn is not None, (
        "a composite loss with a clDice component must be differentiable"
    )
    assert torch.isfinite(loss), "the composite loss must be finite"


# ---------------------------------------------------------------------------
# LossConfig — cldice_iterations field
# ---------------------------------------------------------------------------


def test_loss_config_rejects_cldice_iterations_below_one():
    from deepcontrail.segmentation.losses.config import LossConfig

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        LossConfig(cldice_iterations=0)
