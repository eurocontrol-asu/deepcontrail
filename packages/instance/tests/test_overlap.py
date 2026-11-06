"""Tests for pixel-set overlap (IoU) and greedy duplicate removal."""

from __future__ import annotations

import numpy as np

from deepcontrail.instance._overlap import region_iou, remove_duplicates


def _region(cols: list[int], width: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-row region as ``(rows, cols)`` arrays from a list of column indices."""
    rows = np.zeros(len(cols), dtype=np.int64)
    cols_arr = np.array(cols, dtype=np.int64)
    return rows, cols_arr


def test_iou_of_identical_regions_is_one():
    index = np.array([0, 1, 2, 3], dtype=np.int64)

    assert region_iou(index, index.copy()) == 1.0


def test_iou_of_disjoint_regions_is_zero():
    a = np.array([0, 1], dtype=np.int64)
    b = np.array([2, 3], dtype=np.int64)

    assert region_iou(a, b) == 0.0


def test_iou_of_empty_is_zero():
    empty = np.array([], dtype=np.int64)

    assert region_iou(empty, empty) == 0.0


def test_iou_of_partially_overlapping_regions_is_the_shared_fraction():
    a = np.array([0, 1, 2, 3], dtype=np.int64)
    b = np.array([2, 3, 4, 5], dtype=np.int64)  # share 2 of 6 union pixels

    assert region_iou(a, b) == 2.0 / 6.0


def test_duplicate_removal_keeps_the_first_of_two_overlapping_regions():
    first = _region([0, 1, 2, 3])
    second = _region([2, 3, 4, 5])  # IoU 1/3 with first

    kept = remove_duplicates([first, second], iou_threshold=0.25)

    assert len(kept) == 1
    assert kept[0] is first  # the first overlapping candidate is kept, the second dropped


def test_duplicate_removal_keeps_all_non_overlapping_regions():
    a = _region([0, 1, 2, 3])
    b = _region([20, 21, 22, 23])  # far away, IoU 0

    kept = remove_duplicates([a, b], iou_threshold=0.25)

    assert len(kept) == 2
