"""Tests for direction-following region growing and the local alignment test."""

from __future__ import annotations

import numpy as np

from deepcontrail.instance._growing import grow_regions, pixel_aligns

H = W = 64
LOOKBACK = 53
THRESHOLD = 0.5


def _crossing_lines() -> np.ndarray:
    """Draw a horizontal line at row 32 and a vertical line at column 32, crossing at (32, 32)."""
    mask = np.zeros((H, W), dtype=bool)
    mask[32, :] = True
    mask[:, 32] = True
    return mask


def _flat(rows: np.ndarray, cols: np.ndarray, width: int = W) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in zip(rows, cols, strict=True)}


def _grow(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return grow_regions(
        mask,
        min_pixels=5,
        min_length=3.0,
        min_r_score=0.0,
        lookback=LOOKBACK,
        alignment_threshold=THRESHOLD,
        avg_size=2,
    )


def test_region_growing_splits_two_crossing_lines_into_two_regions():
    regions = _grow(_crossing_lines())

    assert len(regions) == 2
    sizes = sorted(len(r) for r, _ in regions)
    assert sizes == [W, W]  # each full line is 64 pixels


def test_region_growing_lets_a_crossing_pixel_belong_to_both_regions():
    regions = _grow(_crossing_lines())

    assert all((32, 32) in _flat(r, c) for r, c in regions)


def test_region_growing_rejects_a_region_smaller_than_min_pixels():
    mask = np.zeros((H, W), dtype=bool)
    mask[30:32, 30:32] = True  # a 2x2 block, 4 pixels

    regions = grow_regions(
        mask,
        min_pixels=5,
        min_length=3.0,
        min_r_score=0.0,
        lookback=LOOKBACK,
        alignment_threshold=THRESHOLD,
        avg_size=2,
    )

    assert regions == []


def test_alignment_accepts_any_pixel_while_the_region_has_fewer_than_five():
    rows = [0, 1, 2]
    cols = [0, 1, 2]

    assert pixel_aligns(rows, cols, 99, 99, LOOKBACK, THRESHOLD)
    assert pixel_aligns(rows, cols, -50, 400, LOOKBACK, THRESHOLD)


def test_alignment_fit_matches_numpy_polyfit():
    cols = list(range(50))
    rows = [c / 2.0 for c in cols]  # rows = 0.5 * cols exactly
    slope, intercept = np.polyfit(cols, rows, 1)  # independent OLS oracle

    on_line_row = float(intercept + slope * 60.0)
    off_line_row = on_line_row + THRESHOLD + 0.5

    assert pixel_aligns(rows, cols, on_line_row, 60, LOOKBACK, THRESHOLD)
    assert not pixel_aligns(rows, cols, off_line_row, 60, LOOKBACK, THRESHOLD)
