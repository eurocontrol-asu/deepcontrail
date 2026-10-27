"""Pure evaluation metrics for binary segmentation (no gradient, numpy-based)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _to_numpy(x: object) -> np.ndarray:
    """Convert input to a flat float32 numpy array, handling torch tensors safely.

    float32 is enough for binary {0, 1} mask metrics; it halves the host-side
    memory footprint when full val/test sets are concatenated.
    """
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32).ravel()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32).ravel()


def dice_coefficient(pred: ArrayLike, target: ArrayLike, smooth: float = 1e-6) -> float:
    """Compute the Dice coefficient between binary masks.

    Parameters
    ----------
    pred
        Predicted binary mask (values in {0, 1}). Any shape.
    target
        Ground-truth binary mask (values in {0, 1}). Same number of elements
        as ``pred``.
    smooth
        Smoothing constant to avoid division by zero.

    Returns
    -------
    float
        Dice coefficient in [0, 1]. Returns 1.0 when both masks are empty
        (all zeros).
    """
    pred_flat = _to_numpy(pred)
    target_flat = _to_numpy(target)

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return 1.0

    intersection = (pred_flat * target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(pred: ArrayLike, target: ArrayLike, smooth: float = 1e-6) -> float:
    """Compute the Intersection over Union (Jaccard index) between binary masks.

    Parameters
    ----------
    pred
        Predicted binary mask (values in {0, 1}).
    target
        Ground-truth binary mask (values in {0, 1}).
    smooth
        Smoothing constant.

    Returns
    -------
    float
        IoU score in [0, 1]. Returns 1.0 when both masks are empty.
    """
    pred_flat = _to_numpy(pred)
    target_flat = _to_numpy(target)

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return 1.0

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))
