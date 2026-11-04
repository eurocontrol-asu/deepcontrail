"""Tests for deepcontrail.instance._segmentation — alignment-aware BFS."""

from __future__ import annotations

import numpy as np

from deepcontrail.instance._segmentation import segment_contrails

SEGMENT_KWARGS = {
    "min_pixels": 5,
    "min_length": 3.0,
    "min_r_score": 0.1,
    "lookback": 30,
    "alignment_threshold": 5.0,
    "avg_size": 3,
}


def test_segment_contrails_empty_mask_returns_empty_list():
    mask = np.zeros((32, 32), dtype=float)

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    assert segments == []


def test_segment_contrails_horizontal_line_returns_one_segment():
    mask = np.zeros((32, 32), dtype=float)
    mask[16, 2:29] = 1.0

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    assert len(segments) == 1
    rows = segments[0][0]
    assert len(rows) > 20


def test_segment_contrails_two_parallel_lines_returns_two_segments():
    mask = np.zeros((32, 32), dtype=float)
    mask[8, 2:29] = 1.0
    mask[24, 2:29] = 1.0

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    assert len(segments) == 2


def test_segment_contrails_single_pixel_filtered_by_min_pixels():
    mask = np.zeros((32, 32), dtype=float)
    mask[16, 16] = 1.0

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    assert segments == []


def test_segment_contrails_small_cluster_filtered_by_min_pixels():
    mask = np.zeros((32, 32), dtype=float)
    mask[16:18, 16:18] = 1.0

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    assert segments == []


def test_segment_contrails_round_blob_filtered_by_linearity():
    mask = np.zeros((64, 64), dtype=float)
    ys, xs = np.ogrid[0:64, 0:64]
    circle = (ys - 32.0) ** 2 + (xs - 32.0) ** 2 <= 8.0**2
    mask[circle] = 1.0

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    assert segments == []


def test_segment_contrails_output_format_is_coordinate_pairs():
    mask = np.zeros((32, 32), dtype=float)
    mask[16, 2:29] = 1.0

    segments = segment_contrails(mask, **SEGMENT_KWARGS)

    rows, cols = segments[0]
    assert isinstance(rows, np.ndarray)
    assert isinstance(cols, np.ndarray)
    assert rows.dtype.kind == "i"
    assert cols.dtype.kind == "i"
    assert len(rows) == len(cols)
    assert 0 <= int(rows.min()) < 32
    assert 0 <= int(rows.max()) < 32
    assert 0 <= int(cols.min()) < 32
    assert 0 <= int(cols.max()) < 32


def test_segment_contrails_all_foreground_visited_after_run():
    mask = np.zeros((32, 32), dtype=float)
    mask[16, 2:29] = 1.0
    mask[4:7, 4:7] = 1.0

    segments_a = segment_contrails(mask, **SEGMENT_KWARGS)
    segments_b = segment_contrails(mask, **SEGMENT_KWARGS)

    assert len(segments_a) == len(segments_b) == 1
    rows_a, cols_a = segments_a[0]
    rows_b, cols_b = segments_b[0]
    assert np.array_equal(rows_a, rows_b)
    assert np.array_equal(cols_a, cols_b)

    rows, cols = segments_a[0]
    blob_pixels = set(zip(range(4, 7), range(4, 7), strict=True))
    segment_pixels = set(zip(rows.tolist(), cols.tolist(), strict=True))
    assert not (segment_pixels & blob_pixels)
