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


# ---------------------------------------------------------------------------
# BCELoss
# ---------------------------------------------------------------------------


def test_bce_loss_output_is_scalar():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import BCELoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4)).float()

    loss = BCELoss()(logits, target)

    assert loss.shape == (), f"expected scalar, got shape {loss.shape}"


def test_bce_loss_has_grad_fn():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import BCELoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4)).float()

    loss = BCELoss()(logits, target)

    assert loss.grad_fn is not None, "loss must be differentiable"


def test_bce_loss_with_pos_weight():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import BCELoss

    # Logits of zero → sigmoid = 0.5, maximises BCE sensitivity to pos_weight
    logits = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 4, 4)
    target[0, 0, 0] = 1.0  # single positive pixel (very imbalanced)

    loss_no_weight = BCELoss()(logits, target)
    loss_weighted = BCELoss(pos_weight=10.0)(logits, target)

    assert loss_weighted.item() > loss_no_weight.item(), (
        "pos_weight=10 should increase loss on imbalanced positive examples"
    )


def test_bce_loss_perfect_prediction_near_zero():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import BCELoss

    mask = torch.zeros(2, 4, 4)
    mask[:, :2, :] = 1.0
    logits = torch.where(mask.unsqueeze(1) == 1, torch.tensor(10.0), torch.tensor(-10.0))

    loss = BCELoss()(logits, mask)

    assert loss.item() < 0.01, "near-perfect prediction should have near-zero BCE loss"


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


def test_focal_loss_output_is_scalar():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import FocalLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4)).float()

    loss = FocalLoss()(logits, target)

    assert loss.shape == (), f"expected scalar, got shape {loss.shape}"


def test_focal_loss_has_grad_fn():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import FocalLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4)).float()

    loss = FocalLoss()(logits, target)

    assert loss.grad_fn is not None, "focal loss must be differentiable"


def test_focal_loss_with_gamma():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import FocalLoss

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4)
    target = torch.randint(0, 2, (2, 4, 4)).float()

    loss = FocalLoss(gamma=2.0)(logits, target)

    assert loss.shape == (), "FocalLoss(gamma=2.0) should produce a scalar"


def test_focal_loss_easy_examples_downweighted():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import FocalLoss

    # Confident correct predictions: large positive logit, target=1
    logits = torch.full((2, 1, 4, 4), 5.0)
    target = torch.ones(2, 4, 4)

    loss_no_focus = FocalLoss(gamma=0.0)(logits, target)  # gamma=0 ≡ BCE
    loss_focused = FocalLoss(gamma=2.0)(logits, target)

    assert loss_focused.item() < loss_no_focus.item(), (
        "focal gamma=2 should down-weight easy (confident) examples vs gamma=0"
    )


# ---------------------------------------------------------------------------
# CompositeLoss with BCE / build_loss dispatch
# ---------------------------------------------------------------------------


def test_composite_dice_bce_runs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import CompositeLoss, LossConfig

    cfg = LossConfig(
        name="composite",
        smooth=1.0,
        pos_weight=10.0,
        composite_weights={"dice": 0.5, "bce": 0.5},
    )
    loss_fn = CompositeLoss.from_config(cfg)

    torch.manual_seed(0)
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4)).float()

    loss = loss_fn(logits, target)

    assert loss.shape == (), "composite Dice+BCE should return a scalar"
    assert loss.grad_fn is not None, "composite Dice+BCE must be differentiable"


def test_build_loss_bce():
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import BCELoss, LossConfig, build_loss

    loss_fn = build_loss(LossConfig(name="bce"))

    assert isinstance(loss_fn, BCELoss)


def test_build_loss_focal():
    pytest.importorskip("segmentation_models_pytorch")
    from deepcontrail.losses import FocalLoss, LossConfig, build_loss

    loss_fn = build_loss(LossConfig(name="focal"))

    assert isinstance(loss_fn, FocalLoss)


# ---------------------------------------------------------------------------
# LossConfig — new fields
# ---------------------------------------------------------------------------


def test_loss_config_pos_weight_default_none():
    from deepcontrail.losses import LossConfig

    cfg = LossConfig()

    assert cfg.pos_weight is None


def test_loss_config_focal_gamma_default():
    from deepcontrail.losses import LossConfig

    cfg = LossConfig()

    assert cfg.focal_gamma == 2.0
