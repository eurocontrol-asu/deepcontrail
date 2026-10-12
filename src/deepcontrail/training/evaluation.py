"""Post-training evaluation utilities (global metrics + threshold search)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from deepcontrail.losses import LossConfig, build_loss
from deepcontrail.metrics import dice_coefficient, iou_score
from deepcontrail.models import ModelConfig, build_model
from deepcontrail.training.module import ContrailSegmentationModule

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from deepcontrail.training.config import TrainingConfig


def find_optimal_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep thresholds on sigmoid probabilities and return the best Dice.

    Parameters
    ----------
    probs
        Flattened sigmoid probabilities in [0, 1].
    targets
        Flattened binary ground-truth labels in {0, 1}.
    thresholds
        Threshold values to sweep. Defaults to 0.01-0.99 in 0.01 steps.

    Returns
    -------
    tuple[float, float]
        ``(best_threshold, best_dice)``.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_threshold = 0.5
    best_dice = 0.0

    for t in thresholds:
        preds_binary = (probs > t).astype(np.uint8)
        d = dice_coefficient(preds_binary, targets)
        if d > best_dice:
            best_dice = d
            best_threshold = float(t)

    return best_threshold, best_dice


def _threshold_metrics(
    probs_flat: np.ndarray,
    targets_flat: np.ndarray,
    prefix: str,
    threshold: float | None,
) -> dict[str, float]:
    """Compute the metrics dict from flattened probs/targets and a threshold.

    Shared by :func:`evaluate_global` and :func:`evaluate_ensemble` so both
    return an identical metrics layout.
    """
    # Dice at fixed 0.5 (for comparison)
    dice_at_05 = dice_coefficient((probs_flat > 0.5).astype(np.uint8), targets_flat)

    if threshold is None:
        best_threshold, best_dice = find_optimal_threshold(probs_flat, targets_flat)
    else:
        best_threshold = threshold
        best_dice = dice_coefficient((probs_flat > threshold).astype(np.uint8), targets_flat)

    # IoU at optimal threshold
    optimal_iou = iou_score((probs_flat > best_threshold).astype(np.uint8), targets_flat)

    return {
        f"final/{prefix}_dice_global": best_dice,
        f"final/{prefix}_iou_global": optimal_iou,
        f"final/{prefix}_threshold": best_threshold,
        f"final/{prefix}_dice_at_0.5": dice_at_05,
    }


@torch.no_grad()
def evaluate_global(
    module: ContrailSegmentationModule,
    data_loader: DataLoader[Any],
    device: torch.device,
    prefix: str = "val",
    threshold: float | None = None,
) -> dict[str, float]:
    """Run a full evaluation pass with optional threshold search.

    Accumulates sigmoid probabilities and targets on CPU. When ``threshold``
    is ``None``, sweeps thresholds to find the optimal Dice. When a threshold
    is given, applies it directly (used for test set with val-tuned threshold).

    Parameters
    ----------
    module
        Trained Lightning module.
    data_loader
        Data loader for the split to evaluate.
    device
        Device to run inference on.
    prefix
        Metric name prefix (e.g. ``"val"`` or ``"test"``).
    threshold
        Fixed threshold to apply. ``None`` triggers a threshold search.

    Returns
    -------
    dict[str, float]
        Metrics dict including Dice at optimal threshold, the threshold
        value, Dice at 0.5, and IoU at optimal threshold.
    """
    module.eval()
    module.to(device)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, masks in data_loader:
        images = images.to(device)
        logits = module(images)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)
        all_probs.append(probs.ravel())
        all_targets.append(masks.numpy().astype(np.uint8).ravel())

    probs_flat = np.concatenate(all_probs)
    targets_flat = np.concatenate(all_targets)

    return _threshold_metrics(probs_flat, targets_flat, prefix, threshold)


@torch.no_grad()
def evaluate_ensemble(
    checkpoints: list[str],
    model_config: ModelConfig,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    data_loader: DataLoader[Any],
    device: torch.device,
    steps_per_epoch: int,
    prefix: str = "val",
    threshold: float | None = None,
) -> dict[str, float]:
    """Ensemble predictions from several checkpoints for a single metric pass.

    Loads each checkpoint into a :class:`ContrailSegmentationModule`, runs
    inference on every batch with all models, and averages the sigmoid
    probabilities pixel-wise across models before the threshold search. This
    averaging of probabilities (rather than of binary predictions) is what
    yields the Dice gain of a CV ensemble.

    Parameters
    ----------
    checkpoints
        Paths to the per-fold ``.ckpt`` files to ensemble.
    model_config
        Model architecture config used to rebuild each module.
    loss_config
        Loss config used to rebuild each module.
    training_config
        Training hyper-parameters used to rebuild each module.
    data_loader
        Data loader for the split to evaluate.
    device
        Device to run inference on.
    steps_per_epoch
        Number of batches per epoch (mirrors the training module).
    prefix
        Metric name prefix (e.g. ``"val"`` or ``"test"``).
    threshold
        Fixed threshold to apply. ``None`` triggers a threshold search, so the
        ensemble auto-tunes on its averaged probabilities.

    Returns
    -------
    dict[str, float]
        Metrics dict in the same format as :func:`evaluate_global`.
    """
    modules: list[ContrailSegmentationModule] = [
        ContrailSegmentationModule.load_from_checkpoint(
            ckpt,
            model=build_model(model_config),
            loss_fn=build_loss(loss_config),
            lr=training_config.lr,
            weight_decay=training_config.weight_decay,
            scheduler=training_config.scheduler,
            epochs=training_config.epochs,
            warmup_epochs=training_config.warmup_epochs,
            scheduler_interval=training_config.scheduler_interval,
            steps_per_epoch=steps_per_epoch,
        )
        for ckpt in checkpoints
    ]
    for module in modules:
        module.to(device)
        module.eval()

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, masks in data_loader:
        images = images.to(device)

        # Average sigmoid probabilities pixel-wise across all models.
        stack = [torch.sigmoid(module(images)).squeeze(1) for module in modules]
        avg_probs = torch.stack(stack).mean(dim=0)

        all_probs.append(avg_probs.cpu().numpy().astype(np.float32).ravel())
        all_targets.append(masks.numpy().astype(np.uint8).ravel())

    probs_flat = np.concatenate(all_probs)
    targets_flat = np.concatenate(all_targets)

    return _threshold_metrics(probs_flat, targets_flat, prefix, threshold)
