"""Tests for evaluation utilities (threshold search + global evaluation)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepcontrail.segmentation.training import find_optimal_threshold

# ---------------------------------------------------------------------------
# evaluate_ensemble
# ---------------------------------------------------------------------------


class _SqueezingBCE(torch.nn.Module):
    """BCE that accepts ``(B, 1, H, W)`` logits and a ``(B, H, W)`` mask."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), target)


class _RampConv(torch.nn.Module):
    """1x1 conv with seeded weights: output spatial size mirrors the input."""

    def __init__(self, seed: int, scale: float) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = torch.nn.Conv2d(3, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn_like(self.conv.weight) * scale)
            self.conv.bias.copy_(torch.randn(1) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x)


class _ValDataset(torch.utils.data.Dataset):
    """Tiny in-memory split: 64x64 images, 256x256 masks, 4 samples."""

    def __init__(self, n: int = 4) -> None:
        g = torch.Generator().manual_seed(7)
        self.images = [torch.randn(3, 64, 64, generator=g) for _ in range(n)]
        self.masks = [torch.randint(0, 2, (256, 256), generator=g).long() for _ in range(n)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[i], self.masks[i]


def _make_dummy_module(seed: int, scale: float) -> object:
    pytest.importorskip("torch")

    from deepcontrail.segmentation.training import SegmentationModule

    module = SegmentationModule(
        model=_RampConv(seed, scale),
        loss_fn=_SqueezingBCE(),
        lr=1e-4,
        weight_decay=0.0,
        scheduler="CosineAnnealingLR",
        epochs=1,
        native_size=256,
    )
    module.eval()
    return module  # type: ignore[return-value]


def _val_loader() -> object:
    torch = pytest.importorskip("torch")
    loader = torch.utils.data.DataLoader(_ValDataset(), batch_size=2, shuffle=False)
    return loader  # type: ignore[return-value]


def test_find_optimal_threshold_perfect_predictions_give_dice_one() -> None:
    targets = np.array([0, 0, 1, 1, 1, 0], dtype=np.uint8)
    probs = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    threshold, dice = find_optimal_threshold(probs, targets)

    assert dice == 1.0, f"Perfect predictions should yield Dice=1.0, got {dice}"
    assert 0.0 < threshold < 1.0


def test_find_optimal_threshold_all_zeros_gives_dice_one() -> None:
    targets = np.zeros(100, dtype=np.uint8)
    probs = np.zeros(100, dtype=np.float32)

    _threshold, dice = find_optimal_threshold(probs, targets)

    assert dice == 1.0, "All-zero targets with all-zero probs should give Dice=1.0"


def test_find_optimal_threshold_between_zero_and_one() -> None:
    rng = np.random.default_rng(42)
    targets = rng.integers(0, 2, size=200).astype(np.uint8)
    probs = rng.random(200).astype(np.float32)

    threshold, _dice = find_optimal_threshold(probs, targets)

    assert 0.0 < threshold < 1.0, f"Threshold should be in (0, 1), got {threshold}"


def test_find_optimal_threshold_custom_thresholds_respected() -> None:
    targets = np.array([0, 1, 1, 0], dtype=np.uint8)
    probs = np.array([0.1, 0.6, 0.7, 0.2], dtype=np.float32)
    custom = np.array([0.3, 0.5, 0.8])

    threshold, _dice = find_optimal_threshold(probs, targets, thresholds=custom)

    assert threshold in [0.3, 0.5, 0.8], (
        f"Threshold should be one of the custom values, got {threshold}"
    )


def test_find_optimal_threshold_returns_best_dice_value() -> None:
    targets = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
    probs = np.array([0.9, 0.8, 0.1, 0.2, 0.7], dtype=np.float32)

    threshold, dice = find_optimal_threshold(probs, targets)

    preds = (probs > threshold).astype(np.uint8)
    tp = float(np.sum(preds & targets))
    expected_dice = 2.0 * tp / (float(np.sum(preds)) + float(np.sum(targets)))
    assert abs(dice - expected_dice) < 1e-6, (
        f"Returned dice ({dice}) doesn't match recomputed ({expected_dice})"
    )


def test_evaluate_ensemble_averages_multiple_models(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")
    import pytorch_lightning as pl

    from deepcontrail.segmentation.losses import LossConfig
    from deepcontrail.segmentation.models import ModelConfig
    from deepcontrail.segmentation.training import evaluate_ensemble, evaluate_global

    device = torch.device("cpu")
    loader = _val_loader()

    # Train two dummy modules (different init) and checkpoint them.
    module_a = _make_dummy_module(1, 1.0)
    module_b = _make_dummy_module(2, 1.0)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
    )
    ckpt_a = str(tmp_path / "fold_a.ckpt")
    ckpt_b = str(tmp_path / "fold_b.ckpt")
    trainer.fit(module_a, loader)
    trainer.save_checkpoint(ckpt_a)
    trainer.fit(module_b, loader)
    trainer.save_checkpoint(ckpt_b)

    cfg = ModelConfig()
    loss_cfg = LossConfig()

    # Rebuild modules from the checkpoint; architecture/loss come from the
    # monkeypatched factories, mirroring how scripts/evaluate.py loads folds.
    monkeypatch.setattr(
        "deepcontrail.segmentation.training.evaluation.build_model",
        lambda c: _make_dummy_module(0, 0.0).model,
    )
    monkeypatch.setattr(
        "deepcontrail.segmentation.training.evaluation.build_loss", lambda c: _SqueezingBCE()
    )

    ens = evaluate_ensemble(
        [ckpt_a, ckpt_b], cfg, loss_cfg, _dummy_training_cfg(), loader, device, 2
    )
    a_only = evaluate_global(module_a, loader, device)

    # Same metrics layout as the single-model evaluator.
    assert set(ens) == set(a_only), f"keys mismatch: {set(ens)} vs {set(a_only)}"
    for key, value in ens.items():
        if key.endswith("threshold"):
            continue
        assert 0.0 <= value <= 1.0, f"{key}={value} out of [0, 1]"

    # Averaging two different models changes the result versus one model.
    assert ens != a_only, "ensemble should differ from any single fold"
    # Deterministic: identical checkpoints produce identical metrics.
    ens2 = evaluate_ensemble(
        [ckpt_a, ckpt_b], cfg, loss_cfg, _dummy_training_cfg(), loader, device, 2
    )
    assert ens == ens2


class _DummyTrainingConfig:
    """Minimal stand-in for TrainingConfig used only as a build-loss arg."""

    lr = 1e-4
    weight_decay = 0.0
    scheduler = "CosineAnnealingLR"
    epochs = 1
    warmup_epochs = 0
    scheduler_interval = "epoch"


def _dummy_training_cfg() -> object:
    return _DummyTrainingConfig()  # type: ignore[return-value]
