"""Refinement of detected instances using the semantic probability map.

Annotators draw contrails wider than a semantic mask thresholded at 0.5: the model is
trained on agreement-weighted soft labels, so contrail edges where only some annotators
agree fall below 0.5. Growing each instance into neighbouring pixels of moderate
probability brings its shape closer to an annotated contrail.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

_STRUCTURES = {
    4: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    8: np.ones((3, 3), dtype=bool),
}


def grow_instances(
    masks: list[np.ndarray],
    probs: np.ndarray,
    scores: list[float],
    threshold: float,
    steps: int,
    connectivity: int = 8,
) -> list[np.ndarray]:
    """Grow instances into neighbouring pixels whose probability exceeds ``threshold``.

    Each step adds, to every instance, its neighbours (4- or 8-connected) with
    ``probs > threshold`` that belong to no instance yet. A pixel reached by several
    instances in the same step goes to the highest-scoring one (the first one on ties),
    so every newly grown pixel belongs to exactly one instance. Pixels an instance
    already had are always kept, and pixels of other instances are never taken.

    Parameters
    ----------
    masks
        Boolean instance masks, each ``(H, W)``.
    probs
        Semantic probability map ``(H, W)`` in ``[0, 1]``.
    scores
        One score per instance, used to settle contested pixels.
    threshold
        Growth only enters pixels with ``probs > threshold``.
    steps
        Number of growth steps (0 returns copies of the input).
    connectivity
        4 or 8 (default, as in :class:`InstancePipelineConfig`).

    Returns
    -------
    list[np.ndarray]
        Grown boolean masks, aligned with ``masks``.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 4 or 8, ``steps`` is negative, or ``scores`` does
        not have one entry per mask.
    """
    if connectivity not in _STRUCTURES:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if len(scores) != len(masks):
        raise ValueError(f"need one of scores per mask, got {len(scores)} for {len(masks)}")
    grown = [m.astype(bool).copy() for m in masks]
    if not grown or steps == 0:
        return grown

    structure = _STRUCTURES[connectivity]
    allowed = probs > threshold
    owned = np.any(np.stack(grown), axis=0)
    order = sorted(range(len(grown)), key=lambda i: -scores[i])
    for _ in range(steps):
        claimed = np.zeros_like(owned)
        additions = []
        for i in order:
            dilated = np.asarray(binary_dilation(grown[i], structure), dtype=bool)
            new = dilated & allowed & ~owned & ~claimed
            claimed |= new
            additions.append((i, new))
        if not claimed.any():
            break
        for i, new in additions:
            grown[i] |= new
        owned |= claimed
    return grown


def probability_weighted_scores(
    masks: list[np.ndarray], probs: np.ndarray, scores: list[float]
) -> list[float]:
    """Multiply each instance score by the mean probability over the instance.

    The semantic probability reflects how many annotators would mark a pixel, so it
    ranks instances by expected agreement. Empty masks score 0.

    Parameters
    ----------
    masks
        Boolean instance masks, each ``(H, W)``.
    probs
        Semantic probability map ``(H, W)`` in ``[0, 1]``.
    scores
        Base score per instance (e.g. the Mannstein quality score).

    Returns
    -------
    list[float]
        Weighted scores, aligned with ``masks``.
    """
    weighted = []
    for mask, score in zip(masks, scores, strict=True):
        # A 0/1 int mask indexes ``probs`` by position rather than selecting its True pixels.
        bool_mask = np.asarray(mask, dtype=bool)
        weighted.append(float(score * probs[bool_mask].mean()) if bool_mask.any() else 0.0)
    return weighted
