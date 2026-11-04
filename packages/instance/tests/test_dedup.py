"""Tests for deepcontrail.instance._dedup — overlapping-candidate deduplication."""

from __future__ import annotations

import numpy as np

from deepcontrail.instance._dedup import dedup_greedy


def _vertical_line(row0: int, row1: int, col: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertical line covering rows [row0, row1) at the given column."""
    rows = np.arange(row0, row1)
    return rows, np.full(rows.shape, col, dtype=rows.dtype)


def _pixels(contrail: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[int, int], ...]:
    """Ordered pixel coordinates, so equality is order-independent."""
    rows, cols = contrail
    return tuple(sorted(zip(rows.tolist(), cols.tolist(), strict=True)))


def _pixel_set_list(
    contrails: list[tuple[np.ndarray, np.ndarray]],
) -> set[tuple[tuple[int, int], ...]]:
    return {_pixels(c) for c in contrails}


# --- dedup_greedy ---


def test_dedup_greedy_no_overlap_keeps_all():
    a = _vertical_line(0, 10, 5)
    b = _vertical_line(0, 10, 7)
    c = _vertical_line(0, 10, 9)

    result = dedup_greedy([a, b, c], iou_threshold=0.3)

    assert len(result) == 3
    assert result[0] is a
    assert result[1] is b
    assert result[2] is c


def test_dedup_greedy_identical_removes_duplicate():
    a = _vertical_line(0, 10, 5)
    b = _vertical_line(0, 10, 5)

    result = dedup_greedy([a, b], iou_threshold=0.3)

    assert len(result) == 1
    assert result[0] is a


def test_dedup_greedy_high_overlap_removes_second():
    a = _vertical_line(0, 10, 5)
    b = _vertical_line(3, 13, 5)

    result = dedup_greedy([a, b], iou_threshold=0.3)

    assert len(result) == 1
    assert result[0] is a


def test_dedup_greedy_preserves_order():
    a = _vertical_line(0, 10, 5)
    b = _vertical_line(0, 10, 7)
    c = _vertical_line(0, 10, 9)

    result = dedup_greedy([a, b, c], iou_threshold=0.3)

    assert [_pixels(x) for x in result] == [_pixels(a), _pixels(b), _pixels(c)]
