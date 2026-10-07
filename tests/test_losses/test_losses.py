"""Tests for deepcontrail.losses.losses — DiceLoss, TverskyLoss, CompositeLoss."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# DiceLoss
# ---------------------------------------------------------------------------


def test_dice_loss_returns_scalar(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import DiceLoss

    logits, targets = logits_and_targets
    loss = DiceLoss()(logits, targets)

    assert loss.shape == (), f"expected scalar, got shape {loss.shape}"


def test_dice_loss_has_grad_fn(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import DiceLoss

    logits, targets = logits_and_targets
    loss = DiceLoss()(logits, targets)

    assert loss.grad_fn is not None, "loss must be differentiable"


def test_dice_loss_perfect_prediction_near_zero():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import DiceLoss

    targets = torch.ones(1, 1, 4, 4)
    logits = torch.full((1, 1, 4, 4), 10.0)  # high logits → sigmoid ≈ 1.0

    loss = DiceLoss(smooth=0.0)(logits, targets)

    assert loss.item() < 0.05, "loss for perfect prediction should be near zero"


def test_dice_loss_no_overlap_high():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import DiceLoss

    targets = torch.ones(1, 1, 4, 4)
    logits = torch.full((1, 1, 4, 4), -10.0)  # low logits → sigmoid ≈ 0.0

    loss = DiceLoss(smooth=0.0)(logits, targets)

    assert loss.item() > 0.9, "loss for no overlap should be high"


# ---------------------------------------------------------------------------
# TverskyLoss
# ---------------------------------------------------------------------------


def test_tversky_loss_returns_scalar(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import TverskyLoss

    logits, targets = logits_and_targets
    loss = TverskyLoss()(logits, targets)

    assert loss.shape == ()


def test_tversky_loss_has_grad_fn(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import TverskyLoss

    logits, targets = logits_and_targets
    loss = TverskyLoss()(logits, targets)

    assert loss.grad_fn is not None


def test_tversky_loss_respects_alpha_beta():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import TverskyLoss

    torch.manual_seed(0)
    logits = torch.randn(1, 1, 8, 8)
    targets = torch.randint(0, 2, (1, 1, 8, 8)).float()

    loss_fp_heavy = TverskyLoss(alpha=0.9, beta=0.1)(logits, targets)
    loss_fn_heavy = TverskyLoss(alpha=0.1, beta=0.9)(logits, targets)

    assert loss_fp_heavy.item() != pytest.approx(loss_fn_heavy.item(), abs=1e-4), (
        "different alpha/beta should produce different losses"
    )


# ---------------------------------------------------------------------------
# CompositeLoss
# ---------------------------------------------------------------------------


def test_composite_loss_returns_scalar(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import CompositeLoss, DiceLoss, TverskyLoss

    loss_fn = CompositeLoss(losses={"dice": (DiceLoss(), 0.5), "tversky": (TverskyLoss(), 0.5)})
    logits, targets = logits_and_targets
    loss = loss_fn(logits, targets)

    assert loss.shape == ()


def test_composite_loss_has_grad_fn(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import CompositeLoss, DiceLoss, TverskyLoss

    loss_fn = CompositeLoss(losses={"dice": (DiceLoss(), 0.5), "tversky": (TverskyLoss(), 0.5)})
    logits, targets = logits_and_targets
    loss = loss_fn(logits, targets)

    assert loss.grad_fn is not None


def test_composite_loss_is_weighted_sum(logits_and_targets):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import CompositeLoss, DiceLoss, TverskyLoss

    dice = DiceLoss()
    tversky = TverskyLoss()
    w_dice, w_tversky = 0.3, 0.7
    composite = CompositeLoss(losses={"dice": (dice, w_dice), "tversky": (tversky, w_tversky)})

    logits, targets = logits_and_targets
    expected = w_dice * dice(logits, targets) + w_tversky * tversky(logits, targets)
    actual = composite(logits, targets)

    assert torch.allclose(actual, expected, atol=1e-6), "composite should be weighted sum"


# ---------------------------------------------------------------------------
# CompositeLoss.from_config
# ---------------------------------------------------------------------------


def test_composite_loss_from_config(logits_and_targets):
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import CompositeLoss, LossConfig

    cfg = LossConfig(
        name="composite",
        composite_weights={"dice": 0.4, "tversky": 0.6},
    )
    loss_fn = CompositeLoss.from_config(cfg)
    logits, targets = logits_and_targets
    loss = loss_fn(logits, targets)

    assert loss.shape == ()
    assert loss.grad_fn is not None


# ---------------------------------------------------------------------------
# build_loss
# ---------------------------------------------------------------------------


def test_build_loss_dice():
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import DiceLoss, LossConfig, build_loss

    loss_fn = build_loss(LossConfig(name="dice"))

    assert isinstance(loss_fn, DiceLoss)


def test_build_loss_tversky():
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import LossConfig, TverskyLoss, build_loss

    loss_fn = build_loss(LossConfig(name="tversky"))

    assert isinstance(loss_fn, TverskyLoss)


def test_build_loss_composite():
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import CompositeLoss, LossConfig, build_loss

    loss_fn = build_loss(LossConfig(name="composite"))

    assert isinstance(loss_fn, CompositeLoss)


def test_build_loss_unknown_raises():
    from deepcontrail.losses import LossConfig, build_loss

    with pytest.raises(ValueError, match="Unknown loss name"):
        build_loss(LossConfig(name="unknown"))
