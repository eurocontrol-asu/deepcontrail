"""Tests for flat-index region growing, the owner stamp and the running-sum alignment window."""

from __future__ import annotations

import numpy as np
import pytest

from deepcontrail.instance._growing import _Region, grow_regions

H = W = 64
LOOKBACK = 53
THRESHOLD = 0.5
WIDTH = 1000  # padded width used by the direct _Region tests, wider than any coordinate


def _flat(row: int, col: int, width: int = WIDTH) -> int:
    """Flat index of ``(row, col)`` in a mask padded by one pixel on every side."""
    return (row + 1) * width + (col + 1)


def _crossing_lines() -> np.ndarray:
    """Build a mask with a horizontal line at row 32 and a vertical line at column 32."""
    mask = np.zeros((H, W), dtype=bool)
    mask[32, :] = True
    mask[:, 32] = True
    return mask


def _flat_set(rows: np.ndarray, cols: np.ndarray) -> set[tuple[int, int]]:
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


# --- grow_regions behaviour ---


def test_region_growing_splits_two_crossing_lines_into_two_regions():
    regions = _grow(_crossing_lines())

    assert len(regions) == 2
    sizes = sorted(len(r) for r, _ in regions)
    assert sizes == [W, W]  # each full line is 64 pixels


def test_region_growing_lets_a_crossing_pixel_belong_to_both_regions():
    regions = _grow(_crossing_lines())

    assert all((32, 32) in _flat_set(r, c) for r, c in regions)


def test_region_growing_rejects_a_region_smaller_than_min_pixels():
    mask = np.zeros((H, W), dtype=bool)
    mask[30:32, 30:32] = True  # a 2x2 block, 4 pixels

    assert _grow(mask) == []


def test_region_growing_returns_nothing_for_an_empty_mask():
    assert (
        grow_regions(
            np.zeros((H, W), dtype=bool),
            min_pixels=5,
            min_length=3.0,
            min_r_score=0.0,
            lookback=LOOKBACK,
            alignment_threshold=THRESHOLD,
            avg_size=2,
        )
        == []
    )


def test_region_growing_stops_at_the_image_border_without_leaking_out():
    # A vertical line against the top edge: no neighbour may escape above row 0.
    mask = np.zeros((H, W), dtype=bool)
    mask[:, 10] = True

    regions = _grow(mask)

    assert len(regions) == 1
    rows, cols = regions[0]
    assert cols.min() == 10
    assert cols.max() == 10
    assert rows.min() == 0
    assert rows.max() == H - 1


def test_region_growing_keeps_a_straight_line_and_drops_a_bent_blob():
    # A horizontal run that turns sharply up: the turn is rejected by alignment, so the kept
    # region is the straight part only, and a round blob of the same size is dropped entirely.
    mask = np.zeros((H, W), dtype=bool)
    mask[30, 10:20] = True  # straight horizontal run, 10 pixels
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    br = np.round(6 * np.cos(theta) + 40).astype(int)
    bc = np.round(6 * np.sin(theta) + 40).astype(int)
    mask[np.clip(br, 0, H - 1), np.clip(bc, 0, W - 1)] = True  # a round blob, > min_pixels

    regions = _grow(mask)

    kept = {
        tuple(sorted((int(r), int(c)) for r, c in zip(rows, cols, strict=True)))
        for rows, cols in regions
    }
    assert any(len(region) == 10 for region in kept), "the straight run is kept"
    assert not any(len(region) > 10 for region in kept), "the round blob is dropped as not straight"


def test_region_growing_stops_growing_when_a_neighbour_fails_alignment():
    # A horizontal run then a vertical stub: the stub is off the run's direction and is rejected,
    # so the run stays on its row and never grows up the stub.
    mask = np.zeros((H, W), dtype=bool)
    mask[30, 10:20] = True  # horizontal run at row 30
    mask[20:31, 10] = True  # vertical stub up from (30, 10)

    regions = _grow(mask)

    assert len(regions) == 2
    run_rows, run_cols = next(rc for rc in regions if rc[0].min() == 30)
    assert (run_rows == 30).all(), "the run stays on row 30"
    assert set(run_cols.tolist()) == set(range(10, 20)), "the run spans its ten columns"


# --- _Region running-sum alignment window ---


def _line_region(cols, rows, lookback=LOOKBACK) -> _Region:
    # The seed (rows[0], cols[0]) is already in the region; add the rest.
    region = _Region(_flat(rows[0], cols[0]), int(rows[0]), int(cols[0]), lookback, WIDTH)
    for row, col in list(zip(rows, cols, strict=True))[1:]:
        region.add(_flat(int(row), int(col)), int(row), int(col))
    return region


def test_alignment_accepts_any_pixel_while_the_region_has_fewer_than_five():
    region = _Region(_flat(0, 0), 0, 0, LOOKBACK, WIDTH)
    for col in (1, 2, 3):  # three more pixels, still below the bootstrap threshold of five
        region.add(_flat(0, col), 0, col)

    assert region.accepts(99, 99, THRESHOLD) is True
    assert region.accepts(-50, 400, THRESHOLD) is True


def test_alignment_fit_matches_a_direct_least_squares_fit():
    cols = [4 * i for i in range(30)]  # columns vary four times more than rows
    rows = list(range(30))
    region = _line_region(cols, rows)
    region.refresh_fit()
    poly_slope, poly_intercept = np.polyfit(cols, np.asarray(rows, dtype=float), 1)

    assert region.branch == 0  # columns vary more -> rows regressed on columns
    assert region.slope == pytest.approx(poly_slope, abs=1e-9)
    assert region.intercept == pytest.approx(poly_intercept, abs=1e-9)


def test_alignment_window_fits_only_the_last_lookback_pixels():
    lookback = 20
    # Warm-up: a long horizontal run at row 0 that would dominate if the whole history counted.
    region = _Region(_flat(0, 0), 0, 0, lookback, WIDTH)
    for col in range(lookback + 5):
        region.add(_flat(0, col), 0, col)
    # Then a diagonal whose columns vary more; only its last ``lookback`` pixels set the direction.
    cols = [25 + 2 * i for i in range(lookback)]
    rows = list(range(lookback))
    for row, col in zip(rows, cols, strict=True):
        region.add(_flat(row, col), row, col)
    region.refresh_fit()
    poly_slope, poly_intercept = np.polyfit(cols, np.asarray(rows, dtype=float), 1)

    assert region.window_n == lookback
    assert region.branch == 0
    assert region.slope == pytest.approx(poly_slope, abs=1e-9)
    assert region.intercept == pytest.approx(poly_intercept, abs=1e-9)


def test_alignment_accepts_a_pixel_on_the_local_line_and_rejects_one_off_it():
    # Columns vary more (branch 0): a horizontal run at row 0.
    region = _line_region([2 * c for c in range(6)], [0] * 6)
    region.refresh_fit()
    assert region.branch == 0

    assert region.accepts(0, 100, THRESHOLD) is True  # on the line row == 0
    assert region.accepts(5, 100, THRESHOLD) is False  # five rows off


def test_alignment_fits_columns_on_rows_when_rows_vary_more():
    # Rows vary more (branch 1): a vertical run at column 0.
    region = _line_region([0] * 6, [2 * r for r in range(6)])
    region.refresh_fit()
    assert region.branch == 1

    assert region.accepts(100, 0, THRESHOLD) is True  # on the line col == 0
    assert region.accepts(100, 5, THRESHOLD) is False  # five columns off


def test_alignment_slope_is_zero_when_the_window_has_no_spread():
    # A degenerate window where every pixel coincides: both variances are zero, so the fit falls
    # to slope 0 with the regressand mean as intercept.
    region = _Region(_flat(5, 7), 5, 7, LOOKBACK, WIDTH)
    region.add(_flat(5, 7), 5, 7)
    region.add(_flat(5, 7), 5, 7)
    region.refresh_fit()

    assert region.branch == 1
    assert region.slope == 0.0
    assert region.intercept == pytest.approx(7.0)


def _exact_variances(region: _Region) -> tuple[int, int]:
    n = region.window_n
    return (
        n * region.sum_c2 - region.sum_c * region.sum_c,
        n * region.sum_r2 - region.sum_r * region.sum_r,
    )


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        # 45-degree diagonal: the float spreads are equal too, so rows are the regressor.
        (list(range(8)), list(range(8))),
        # Scattered window whose float spreads differ only in the last digit (columns larger).
        ([22, 6, 31, 32, 38], [33, 15, 2, 29, 25]),
    ],
)
def test_alignment_exact_variance_tie_follows_the_float_spread_comparison(rows, cols):
    region = _line_region(cols, rows)

    region.refresh_fit()

    var_c, var_r = _exact_variances(region)
    assert var_c == var_r != 0, "the window must be an exact, non-degenerate spread tie"
    expected = 0 if np.std(np.asarray(cols, float)) > np.std(np.asarray(rows, float)) else 1
    assert region.branch == expected, "exact ties keep the float std comparison of v0.1.0"


def test_alignment_unequal_spreads_choose_the_axis_with_the_larger_spread():
    region = _line_region(cols=[0, 3, 6, 9, 12, 15], rows=[0, 1, 2, 3, 4, 5])

    region.refresh_fit()

    var_c, var_r = _exact_variances(region)
    assert var_c > var_r, "columns spread more in this window"
    assert region.branch == 0, "the larger spread (columns) is the regressor"


def test_region_coords_returns_accepted_pixels_in_acceptance_order():
    cols = [4 * i for i in range(8)]
    rows = list(range(8))
    region = _line_region(cols, rows)

    out_rows, out_cols = region.coords()

    assert out_rows.tolist() == rows
    assert out_cols.tolist() == cols
