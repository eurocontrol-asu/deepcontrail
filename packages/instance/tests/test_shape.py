"""Tests for the least-squares fit, straightness and the length/quality shape tests."""

from __future__ import annotations

import numpy as np
import pytest

from deepcontrail.instance._shape import (
    candidate_score,
    is_long_and_straight,
    linear_fit,
    straightness,
)


def _horizontal(rows: float, cols0: int, cols1: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(cols0, cols1)
    return np.full_like(cols, float(rows)), cols.astype(float)


def test_linear_fit_matches_numpy_polyfit_on_a_noisy_line():
    rng = np.random.default_rng(0)
    xs = np.arange(40, dtype=float)
    ys = 2.0 * xs + 3.0 + rng.normal(0.0, 0.01, size=xs.shape)

    r_squared, slope, intercept = linear_fit(xs, ys)
    poly_slope, poly_intercept = np.polyfit(xs, ys, 1)

    assert r_squared == pytest.approx(1.0, abs=1e-6)
    assert slope == pytest.approx(poly_slope, abs=1e-9)
    assert intercept == pytest.approx(poly_intercept, abs=1e-9)


def test_straightness_of_a_line_is_one_and_a_blob_is_low():
    rows, cols = _horizontal(0.0, 0, 40)
    blob_r = np.array([30.0, 31.0, 30.0, 31.0])
    blob_c = np.array([30.0, 30.0, 31.0, 31.0])

    assert straightness(rows, cols) == pytest.approx(1.0)
    assert straightness(blob_r, blob_c) < 0.5


def test_length_test_accepts_a_long_straight_segment():
    rows, cols = _horizontal(5.0, 0, 10)  # 10 pixels, length 9

    assert is_long_and_straight(rows, cols, min_r_score=0.0, min_length=5.0, avg_size=2)


def test_length_test_rejects_a_short_straight_segment():
    rows, cols = _horizontal(5.0, 0, 10)  # 10 pixels, length 9

    assert not is_long_and_straight(rows, cols, min_r_score=0.0, min_length=20.0, avg_size=2)


def test_length_test_rejects_a_long_but_bent_segment():
    straight_r = np.arange(10, dtype=float)
    straight_c = 3.0 * straight_r  # steep but perfectly straight
    # a round blob is long enough but not straight, so it must be rejected
    ys, xs = np.mgrid[-6:7, -6:7]
    blob_r = ys[(ys**2 + xs**2) <= 36].astype(float)
    blob_c = xs[(ys**2 + xs**2) <= 36].astype(float)

    assert straightness(blob_r, blob_c) < 0.3
    assert is_long_and_straight(straight_r, straight_c, min_r_score=0.0, min_length=5.0, avg_size=2)
    assert not is_long_and_straight(blob_r, blob_c, min_r_score=0.1, min_length=5.0, avg_size=2)


def test_candidate_score_is_zero_for_a_tiny_blob():
    rows = np.array([0.0, 1.0])
    cols = np.array([0.0, 1.0])

    assert candidate_score(rows, cols, saturation=10.0, avg_size=2) == 0.0


def test_candidate_score_is_straightness_times_tanh_of_size():
    rows, cols = _horizontal(0.0, 0, 30)
    saturation = 10.0

    expected = float(straightness(rows, cols) * np.tanh(30.0 / saturation))

    assert candidate_score(rows, cols, saturation=saturation, avg_size=2) == pytest.approx(expected)
