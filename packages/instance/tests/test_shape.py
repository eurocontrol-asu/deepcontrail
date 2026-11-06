"""Tests for the least-squares fit, straightness and the length/quality shape tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from deepcontrail.instance._shape import (
    _dominant_fit,
    _ols,
    candidate_score,
    is_long_and_straight,
    linear_fit,
    straightness,
)

# scikit-learn is the reference implementation of the line fit; installed with the [scores] extra.
requires_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="needs scikit-learn as reference"
)


def _strip(length: int, width: int, vertical: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of a straight ``length`` x ``width`` pixel strip along an image axis."""
    along, across = np.meshgrid(np.arange(length), np.arange(width), indexing="ij")
    along, across = along.ravel().astype(float), across.ravel().astype(float)
    return (along, across) if vertical else (across, along)


def _horizontal(rows: float, cols0: int, cols1: int) -> tuple[np.ndarray, np.ndarray]:
    cols = np.arange(cols0, cols1)
    return np.full_like(cols, float(rows)), cols.astype(float)


# --- _ols ---


def test_ols_degenerate_cases_are_exact():
    # Zero variance in the independent variable: slope 0, intercept the mean of y, r2 0 (y varies).
    r_squared, slope, intercept = _ols([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    assert (slope, intercept) == (0.0, 2.5)
    assert r_squared == 0.0
    # Constant y: total sum of squares zero, so r2 is 1, slope 0 and intercept the mean of y.
    r_squared, slope, intercept = _ols([0.0, 1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0, 2.0])
    assert (slope, intercept) == (0.0, 2.0)
    assert r_squared == 1.0


@requires_sklearn
def test_ols_matches_sklearn_on_random_points():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(1234)
    for _ in range(200):
        n = int(rng.integers(2, 61))
        xs = rng.normal(size=n)
        ys = rng.normal(size=n)

        r2, slope, intercept = _ols(xs, ys)
        model = LinearRegression().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
        ref_slope = float(np.asarray(model.coef_).reshape(-1)[0])
        ref_intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        ref_r2 = float(model.score(xs.reshape(-1, 1), ys.reshape(-1, 1)))

        assert slope == pytest.approx(ref_slope, abs=1e-9)
        assert intercept == pytest.approx(ref_intercept, abs=1e-9)
        assert r2 == pytest.approx(ref_r2, abs=1e-9)


@requires_sklearn
def test_ols_matches_sklearn_on_degenerate_inputs():
    from sklearn.linear_model import LinearRegression

    for xs, ys in (
        ([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]),  # zero-variance x
        ([0.0, 1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0, 2.0]),  # constant y
    ):
        r2, slope, intercept = _ols(np.asarray(xs), np.asarray(ys))
        model = LinearRegression().fit(np.asarray(xs).reshape(-1, 1), np.asarray(ys).reshape(-1, 1))
        ref_slope = float(np.asarray(model.coef_).reshape(-1)[0])
        ref_intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
        ref_r2 = float(model.score(np.asarray(xs).reshape(-1, 1), np.asarray(ys).reshape(-1, 1)))

        assert slope == pytest.approx(ref_slope)
        assert intercept == pytest.approx(ref_intercept)
        assert r2 == pytest.approx(ref_r2)


# --- linear_fit ---


def test_linear_fit_matches_numpy_polyfit_on_a_noisy_line():
    rng = np.random.default_rng(0)
    xs = np.arange(40, dtype=float)
    ys = 2.0 * xs + 3.0 + rng.normal(0.0, 0.01, size=xs.shape)

    r_squared, slope, intercept = linear_fit(xs, ys)
    poly_slope, poly_intercept = np.polyfit(xs, ys, 1)

    assert r_squared == pytest.approx(1.0, abs=1e-6)
    assert slope == pytest.approx(poly_slope, abs=1e-9)
    assert intercept == pytest.approx(poly_intercept, abs=1e-9)


def test_linear_fit_perfect_line_r2_is_one():
    xs = np.array([0, 1, 2, 3, 4], dtype=float)
    ys = np.array([1, 3, 5, 7, 9], dtype=float)

    r2, slope, intercept = linear_fit(xs, ys)

    assert r2 == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


# --- _dominant_fit ---


def test_dominant_fit_recovers_a_perfect_line_from_integer_sums():
    cols = list(range(0, 40, 2))  # 0, 2, ..., 38
    rows = [c // 2 for c in cols]  # exactly 0.5 * cols
    n = len(cols)
    sum_rc = sum(r * c for r, c in zip(rows, cols, strict=True))

    slope, intercept = _dominant_fit(sum_rc, sum(rows), sum(cols), sum(c * c for c in cols), n)

    assert slope == pytest.approx(0.5)
    assert intercept == pytest.approx(0.0)


def test_dominant_fit_zero_spread_independent_gives_zero_slope_and_mean_intercept():
    slope, intercept = _dominant_fit(sum_cross=0, sum_dep=15, sum_indep=15, sum_indep_sq=75, n=3)

    assert slope == 0.0
    assert intercept == pytest.approx(5.0)  # mean of the dependent variable


# --- straightness ---


@pytest.mark.parametrize(
    ("rows", "cols"),
    [
        (np.zeros(20), np.arange(20.0)),
        (np.arange(20.0), np.zeros(20)),
        (np.arange(20.0), np.arange(20.0)),
        (np.arange(20.0), -np.arange(20.0)),
    ],
    ids=["horizontal", "vertical", "diagonal", "anti-diagonal"],
)
def test_straightness_of_a_one_pixel_line_is_one_at_any_angle(rows, cols):
    assert straightness(rows, cols) == pytest.approx(1.0)


def test_straightness_is_the_same_for_a_wide_strip_along_either_axis():
    horizontal = straightness(*_strip(40, 3))
    vertical = straightness(*_strip(40, 3, vertical=True))

    assert horizontal == pytest.approx(vertical)
    assert horizontal > 0.95, "a 40 x 3 strip is very straight"


def test_straightness_equals_the_least_squares_r2_at_45_degrees():
    rng = np.random.default_rng(0)
    t = np.arange(30.0)
    offset = rng.normal(scale=1.5, size=30)
    rows, cols = t + offset, t - offset  # noisy points along the 45-degree diagonal

    r2, _, _ = linear_fit(xs=cols, ys=rows)

    assert straightness(rows, cols) == pytest.approx(r2, rel=0.02)


def test_straightness_of_a_round_blob_is_near_zero():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)

    assert straightness(10.0 * np.cos(theta), 10.0 * np.sin(theta)) == pytest.approx(0.0, abs=1e-9)


def test_straightness_of_a_single_point_is_zero():
    assert straightness(np.array([3.0]), np.array([4.0])) == 0.0


# --- is_long_and_straight ---


def test_length_test_accepts_a_long_straight_segment():
    rows, cols = _horizontal(5.0, 0, 10)  # 10 pixels, length 9

    assert is_long_and_straight(rows, cols, min_r_score=0.0, min_length=5.0, avg_size=2)


def test_length_test_rejects_a_short_straight_segment():
    rows, cols = _horizontal(5.0, 0, 10)  # 10 pixels, length 9

    assert not is_long_and_straight(rows, cols, min_r_score=0.0, min_length=20.0, avg_size=2)


def test_length_test_rejects_a_region_with_too_few_points():
    ys = np.arange(5, dtype=float)
    xs = np.arange(5, dtype=float)

    assert not is_long_and_straight(ys, xs, min_r_score=0.1, min_length=5.0, avg_size=3)


def test_length_test_rejects_a_long_but_bent_segment():
    straight_r = np.arange(10, dtype=float)
    straight_c = 3.0 * straight_r  # steep but perfectly straight
    ys, xs = np.mgrid[-6:7, -6:7]
    blob_r = ys[(ys**2 + xs**2) <= 36].astype(float)
    blob_c = xs[(ys**2 + xs**2) <= 36].astype(float)

    assert straightness(blob_r, blob_c) < 0.3
    assert is_long_and_straight(straight_r, straight_c, min_r_score=0.0, min_length=5.0, avg_size=2)
    assert not is_long_and_straight(blob_r, blob_c, min_r_score=0.1, min_length=5.0, avg_size=2)


def test_length_test_accepts_a_straight_diagonal():
    ys = np.arange(20, dtype=float)
    xs = np.arange(20, dtype=float)

    assert is_long_and_straight(ys, xs, min_r_score=0.1, min_length=5.0, avg_size=3)


@pytest.mark.parametrize("vertical", [False, True], ids=["horizontal", "vertical"])
def test_length_test_accepts_a_wide_contrail_along_an_axis(vertical):
    rows, cols = _strip(40, 3, vertical=vertical)

    assert is_long_and_straight(rows, cols, min_r_score=0.1, min_length=5.0, avg_size=3)


# --- candidate_score ---


def test_candidate_score_is_zero_for_a_tiny_blob():
    rows = np.array([0.0, 1.0])
    cols = np.array([0.0, 1.0])

    assert candidate_score(rows, cols, saturation=10.0, avg_size=2) == 0.0


def test_candidate_score_is_straightness_times_tanh_of_size():
    rows, cols = _horizontal(0.0, 0, 30)
    saturation = 10.0

    expected = float(straightness(rows, cols) * np.tanh(30.0 / saturation))

    assert candidate_score(rows, cols, saturation=saturation, avg_size=2) == pytest.approx(expected)


def test_candidate_score_perfect_line_high_score():
    rows = np.arange(20, dtype=float)
    cols = np.arange(20, dtype=float)

    assert candidate_score(rows, cols, saturation=1.0, avg_size=3) == pytest.approx(1.0, abs=1e-6)


def test_candidate_score_round_blob_is_zero():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rows = 10.0 * np.cos(theta)
    cols = 10.0 * np.sin(theta)

    assert candidate_score(rows, cols, saturation=40.0, avg_size=3) == pytest.approx(0.0, abs=0.01)


def test_candidate_score_increases_with_size():
    score_small = candidate_score(
        np.arange(10, dtype=float), np.arange(10, dtype=float), saturation=40.0, avg_size=3
    )
    score_large = candidate_score(
        np.arange(30, dtype=float), np.arange(30, dtype=float), saturation=40.0, avg_size=3
    )

    assert score_large > score_small


def test_candidate_score_does_not_depend_on_the_direction():
    horizontal = candidate_score(*_strip(40, 1), saturation=40.0, avg_size=3)
    vertical = candidate_score(*_strip(40, 1, vertical=True), saturation=40.0, avg_size=3)
    diagonal = candidate_score(np.arange(40.0), np.arange(40.0), saturation=40.0, avg_size=3)

    assert vertical > 0.0, "a vertical contrail is a contrail"
    assert horizontal == pytest.approx(vertical)
    assert horizontal == pytest.approx(diagonal)


def test_candidate_score_few_points_returns_zero():
    rows = np.array([0, 1, 2], dtype=float)
    cols = np.array([0, 1, 2], dtype=float)

    assert candidate_score(rows, cols, saturation=1.0, avg_size=3) == 0.0
