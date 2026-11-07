"""Pure evaluation metrics for binary segmentation (no gradient, numpy-based)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _to_numpy(x: object) -> np.ndarray:
    """Convert input to a flat float32 numpy array, handling torch tensors safely.

    float32 is enough for binary {0, 1} mask metrics; it halves the host-side
    memory footprint when full val/test sets are concatenated.
    """
    # A tensor can only exist if torch is already imported: look it up instead of
    # importing it, so numpy-only callers never load torch (deepcontrail-core has no torch).
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        tensor: Any = x
        return tensor.detach().cpu().numpy().astype(np.float32).ravel()
    return np.asarray(x, dtype=np.float32).ravel()


def _check_binary_pair(pred: np.ndarray, target: np.ndarray) -> None:
    """Raise ``ValueError`` unless both masks have the same size and hold only 0 and 1."""
    if pred.size != target.size:
        msg = f"inputs must have the same number of elements; got {pred.size} and {target.size}"
        raise ValueError(msg)
    for name, values in (("pred", pred), ("target", target)):
        # Elementwise test: no sort, so cheap on whole validation sets.
        if not ((values == 0) | (values == 1)).all():
            bad = values[(values != 0) & (values != 1)]
            msg = f"{name} must contain only 0 and 1, got e.g. {bad[:5].tolist()}"
            raise ValueError(msg)


def dice_coefficient(pred: ArrayLike, target: ArrayLike, smooth: float = 1e-6) -> float:
    """Compute the Dice coefficient between binary masks.

    Parameters
    ----------
    pred
        Predicted binary mask (values in {0, 1}, boolean accepted). Any shape.
    target
        Ground-truth binary mask (values in {0, 1}, boolean accepted). Same number
        of elements as ``pred``.
    smooth
        Smoothing constant to avoid division by zero.

    Returns
    -------
    float
        Dice coefficient in [0, 1], pooled over all elements (one global score,
        not a per-image mean). Returns 1.0 when both masks are empty (all zeros).

    Raises
    ------
    ValueError
        If ``pred`` and ``target`` have different numbers of elements, or if either
        contains values outside {0, 1}.
    """
    pred_flat = _to_numpy(pred)
    target_flat = _to_numpy(target)

    _check_binary_pair(pred_flat, target_flat)

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return 1.0

    intersection = (pred_flat * target_flat).sum()
    return float((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))


def iou_score(pred: ArrayLike, target: ArrayLike, smooth: float = 1e-6) -> float:
    """Compute the Intersection over Union (Jaccard index) between binary masks.

    Parameters
    ----------
    pred
        Predicted binary mask (values in {0, 1}, boolean accepted).
    target
        Ground-truth binary mask (values in {0, 1}, boolean accepted).
    smooth
        Smoothing constant.

    Returns
    -------
    float
        IoU score in [0, 1], pooled over all elements (one global score, not a
        per-image mean). Returns 1.0 when both masks are empty (all zeros).

    Raises
    ------
    ValueError
        If ``pred`` and ``target`` have different numbers of elements, or if either
        contains values outside {0, 1}.
    """
    pred_flat = _to_numpy(pred)
    target_flat = _to_numpy(target)

    _check_binary_pair(pred_flat, target_flat)

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return 1.0

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return float((intersection + smooth) / (union + smooth))
