"""Tests for deepcontrail.mtg.raster — polygons to masks and block-mean downscaling."""

from __future__ import annotations

import warnings

import numpy as np


def _square(x0: float, y0: float, x1: float, y1: float) -> list:
    """Axis-aligned square with opposite corners ``(x0, y0)`` and ``(x1, y1)``."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_rasterize_fills_pixel_centres_inside_a_square():
    from deepcontrail.mtg.raster import rasterize

    mask = rasterize([_square(0.25, 0.25, 0.75, 0.75)], size=8)

    expected = np.zeros((8, 8), dtype=bool)
    expected[2:6, 2:6] = True
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, expected, err_msg="centres 2.5..5.5 lie inside [2, 6]")


def test_rasterize_excludes_pixels_whose_centre_is_outside():
    from deepcontrail.mtg.raster import rasterize

    # scaled to x in (0.7, 1.3) on an 8px grid: between the centres 0.5 and 1.5 -> empty
    mask = rasterize([_square(0.7 / 8, 0.3, 1.3 / 8, 0.7)], size=8)

    assert not mask.any(), "a sliver containing no pixel centre rasterises to nothing"


def test_rasterize_offset_shifts_the_mask_by_whole_pixels():
    from deepcontrail.mtg.raster import rasterize

    poly = [_square(0.25, 0.25, 0.75, 0.75)]
    base = rasterize(poly, size=8)

    shifted = rasterize(poly, size=8, offset=(1, 0))

    np.testing.assert_array_equal(
        shifted, np.roll(base, shift=1, axis=1), err_msg="offset (dx, dy) shifts columns then rows"
    )


def test_rasterize_clips_polygons_outside_the_grid():
    from deepcontrail.mtg.raster import rasterize

    # x from 0.5 to 1.5 -> scaled [4, 12]; the part beyond the grid must not raise.
    mask = rasterize([_square(0.5, 0.25, 1.5, 0.75)], size=8)

    expected = np.zeros((8, 8), dtype=bool)
    expected[2:6, 4:8] = True
    np.testing.assert_array_equal(mask, expected, err_msg="only the in-grid part is set")


def test_rasterize_ignores_degenerate_polygons():
    from deepcontrail.mtg.raster import rasterize

    mask = rasterize([[[0.0, 0.0], [1.0, 1.0]]], size=8)  # two vertices

    assert not mask.any(), "a polygon with fewer than three vertices is ignored"


def test_rasterize_triangle_matches_the_even_odd_rule():
    from deepcontrail.mtg.raster import rasterize

    # no vertex lands on a pixel centre and no edge crosses one exactly
    triangle = [[0.2, 0.16], [0.8, 0.4], [0.46, 0.84]]
    size = 10

    mask = rasterize([triangle], size=size)

    # the same polygon in pixel space, exactly as rasterize scales it
    triangle_px = [[x * size, y * size] for x, y in triangle]
    xs = np.arange(size) + 0.5
    ys = np.arange(size) + 0.5
    xx, yy = np.meshgrid(xs, ys)  # (row, col) of pixel-centre x and y
    reference = np.vectorize(lambda cx, cy: _point_in_poly(cx, cy, triangle_px))(xx, yy)
    np.testing.assert_array_equal(mask, reference, err_msg="matches a brute-force even-odd test")


def _point_in_poly(x: float, y: float, polygon: list) -> bool:
    """Even-odd point-in-polygon: the brute-force reference the triangle test compares against."""
    inside = False
    last = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[last]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        last = i
    return inside


def test_block_mean_averages_blocks_ignoring_nan():
    from deepcontrail.mtg.raster import block_mean

    image = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan]],
            [[5.0, 6.0], [np.nan, 8.0]],
        ],
        dtype=np.float32,
    )

    out = block_mean(image, 2)

    assert out.shape == (1, 1, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [3.0, 16.0 / 3.0], atol=1e-6, err_msg="NaN excluded")


def test_block_mean_all_nan_block_stays_nan_without_warning():
    from deepcontrail.mtg.raster import block_mean

    image = np.full((2, 2, 2), np.nan, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = block_mean(image, 2)

    assert np.isnan(out).all(), "an all-NaN block stays NaN"
    assert out.dtype == np.float32


def test_block_mean_rejects_indivisible_sizes():
    import pytest

    from deepcontrail.mtg.raster import block_mean

    with pytest.raises(ValueError, match="divisible"):
        block_mean(np.zeros((3, 4, 2), np.float32), 2)  # height 3 is not divisible by 2
