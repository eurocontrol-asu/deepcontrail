"""Pixel-set overlap and greedy duplicate removal of contrail candidates.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

Two regions overlap when they share pixels (contrails crossing). A candidate is a duplicate
of an already-kept one when their pixel-set IoU exceeds a threshold; the first found is kept.
"""

from __future__ import annotations

import numpy as np


def region_iou(
    index_a: np.ndarray,
    index_b: np.ndarray,
) -> float:
    """Pixel-set IoU of two regions given as arrays of unique flat pixel indices.

    Parameters
    ----------
    index_a
        Flat pixel indices (``row * width + col``) of the first region, unique.
    index_b
        Flat pixel indices of the second region, unique.

    Returns
    -------
    float
        Intersection-over-union in ``[0, 1]``; ``0.0`` when the union is empty.
    """
    shared = np.intersect1d(index_a, index_b).size
    if shared == 0:
        return 0.0
    union = int(index_a.size) + int(index_b.size) - int(shared)
    return shared / union


def _flat_indices(candidates: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
    """One array of unique flat pixel indices per candidate, under a shared image width."""
    if not candidates:
        return []
    width = max(int(cols.max()) for _, cols in candidates) + 1
    return [(rows * width + cols).astype(np.int64) for rows, cols in candidates]


def remove_duplicates(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    iou_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Drop candidates that overlap an already-kept one beyond ``iou_threshold``.

    Walks the candidates in order; a candidate is kept only if its IoU with every candidate
    kept so far is at most ``iou_threshold``. The surviving candidates preserve the input
    order and references.

    Parameters
    ----------
    candidates
        Candidate contrails as ``(rows, cols)`` coordinate arrays.
    iou_threshold
        IoU above which two candidates are considered duplicates.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The surviving candidates, in the original order.
    """
    indices = _flat_indices(candidates)
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    kept_indices: list[np.ndarray] = []
    for candidate, index in zip(candidates, indices, strict=True):
        duplicate = any(region_iou(index, other) > iou_threshold for other in kept_indices)
        if not duplicate:
            kept.append(candidate)
            kept_indices.append(index)
    return kept
