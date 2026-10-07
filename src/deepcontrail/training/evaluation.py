"""Post-training evaluation utilities (global metrics + threshold search)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from deepcontrail.metrics import dice_coefficient, iou_score

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from deepcontrail.training.module import ContrailSegmentationModule


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

    # Dice at fixed 0.5 (for comparison)
    preds_at_05 = (probs_flat > 0.5).astype(np.uint8)
    dice_at_05 = dice_coefficient(preds_at_05, targets_flat)

    if threshold is None:
        best_threshold, best_dice = find_optimal_threshold(probs_flat, targets_flat)
    else:
        best_threshold = threshold
        preds_binary = (probs_flat > threshold).astype(np.uint8)
        best_dice = dice_coefficient(preds_binary, targets_flat)

    # IoU at optimal threshold
    preds_optimal = (probs_flat > best_threshold).astype(np.uint8)
    optimal_iou = iou_score(preds_optimal, targets_flat)

    return {
        f"final/{prefix}_dice_global": best_dice,
        f"final/{prefix}_iou_global": optimal_iou,
        f"final/{prefix}_threshold": best_threshold,
        f"final/{prefix}_dice_at_0.5": dice_at_05,
    }
