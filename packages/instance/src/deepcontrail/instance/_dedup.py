"""Deduplication of overlapping contrail candidates.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

``dedup_greedy`` keeps the first-found of overlapping candidates (order dependent, no
scoring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepcontrail.instance._geometry import pairwise_iou

if TYPE_CHECKING:
    import numpy as np


def dedup_greedy(
    contrails: list[tuple[np.ndarray, np.ndarray]],
    iou_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Remove overlapping contrails using a greedy first-seen-wins strategy.

    Walks the candidates in order; a candidate is kept only if it does not
    exceed ``iou_threshold`` against every candidate kept so far.

    Parameters
    ----------
    contrails
        Candidate contrails as ``(rows, cols)`` coordinate arrays.
    iou_threshold
        IoU above which two candidates are considered overlapping.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        The surviving candidates, preserving the original references and order.
    """
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    for candidate in contrails:
        if any(pairwise_iou(candidate, other) > iou_threshold for other in kept):
            continue
        kept.append(candidate)
    return kept
