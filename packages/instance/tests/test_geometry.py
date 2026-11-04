"""Tests for deepcontrail.instance._geometry — linearity, alignment, scoring."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from deepcontrail.instance._geometry import (
    _ols,
    candidate_score,
    is_linear_and_long,
    is_pixel_aligned,
    linear_fit,
    pairwise_iou,
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


# --- linear_fit ---


def test_linear_fit_perfect_line_r2_is_one():
    xs = np.array([0, 1, 2, 3, 4], dtype=float)
    ys = np.array([1, 3, 5, 7, 9], dtype=float)

    r2, slope, intercept = linear_fit(xs, ys)

    assert r2 == pytest.approx(1.0)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)


def test_linear_fit_noisy_points_r2_below_one():
    rng = np.random.default_rng(42)
    xs = np.arange(10, dtype=float)
    ys = xs + rng.normal(scale=0.1, size=10)

    r2, slope, intercept = linear_fit(xs, ys)

    assert r2 < 1.0
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.5)


# --- _ols ---


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
def test_ols_vertical_input_zero_variance_matches_sklearn():
    from sklearn.linear_model import LinearRegression

    xs = np.array([5.0, 5.0, 5.0, 5.0])
    ys = np.array([1.0, 2.0, 3.0, 4.0])

    r2, slope, intercept = _ols(xs, ys)

    model = LinearRegression().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    ref_slope = float(np.asarray(model.coef_).reshape(-1)[0])
    ref_intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    ref_r2 = float(model.score(xs.reshape(-1, 1), ys.reshape(-1, 1)))

    assert slope == pytest.approx(ref_slope)
    assert intercept == pytest.approx(ref_intercept)
    assert r2 == pytest.approx(ref_r2)


@requires_sklearn
def test_ols_constant_y_matches_sklearn():
    from sklearn.linear_model import LinearRegression

    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    r2, slope, intercept = _ols(xs, ys)

    model = LinearRegression().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    ref_slope = float(np.asarray(model.coef_).reshape(-1)[0])
    ref_intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    ref_r2 = float(model.score(xs.reshape(-1, 1), ys.reshape(-1, 1)))

    assert slope == pytest.approx(ref_slope)
    assert intercept == pytest.approx(ref_intercept)
    assert r2 == pytest.approx(ref_r2)


@requires_sklearn
def test_is_pixel_aligned_matches_sklearn_reference():
    from sklearn.linear_model import LinearRegression

    def _reference(
        current_ys: list[int],
        current_xs: list[int],
        new_y: int,
        new_x: int,
        lookback: int,
        threshold: float,
    ) -> bool:
        if len(current_ys) < 5:
            return True
        local_ys = np.asarray(current_ys[-lookback:], dtype=float).reshape(-1, 1)
        local_xs = np.asarray(current_xs[-lookback:], dtype=float).reshape(-1, 1)
        model = LinearRegression()
        if np.std(local_xs) > np.std(local_ys):
            model.fit(local_xs, local_ys)
            predicted_y = float(model.predict(np.array([[new_x]], dtype=float))[0, 0])
            return abs(predicted_y - new_y) < threshold
        model.fit(local_ys, local_xs)
        predicted_x = float(model.predict(np.array([[new_y]], dtype=float))[0, 0])
        return abs(predicted_x - new_x) < threshold

    rng = np.random.default_rng(9876)
    for _ in range(500):
        n = int(rng.integers(5, 60))
        current_ys = rng.integers(0, 100, size=n).tolist()
        current_xs = rng.integers(0, 100, size=n).tolist()
        new_y = int(rng.integers(0, 100))
        new_x = int(rng.integers(0, 100))
        lookback = int(rng.integers(3, 20))
        threshold = float(rng.uniform(0.5, 5.0))

        expected = _reference(current_ys, current_xs, new_y, new_x, lookback, threshold)
        actual = is_pixel_aligned(current_ys, current_xs, new_y, new_x, lookback, threshold)

        assert actual is expected


# --- is_linear_and_long ---


def test_is_linear_and_long_straight_diagonal_passes():
    ys = np.arange(20, dtype=float)
    xs = np.arange(20, dtype=float)

    result = is_linear_and_long(ys, xs, min_r_score=0.196, min_length=5.0, avg_size=3)

    assert result is True


def test_is_linear_and_long_short_segment_fails():
    ys = np.array([0, 1], dtype=float)
    xs = np.array([0, 1], dtype=float)

    result = is_linear_and_long(ys, xs, min_r_score=0.196, min_length=5.0, avg_size=3)

    assert result is False


def test_is_linear_and_long_too_few_points_returns_false():
    ys = np.arange(5, dtype=float)
    xs = np.arange(5, dtype=float)

    result = is_linear_and_long(ys, xs, min_r_score=0.196, min_length=5.0, avg_size=3)

    assert result is False


def test_is_linear_and_long_round_blob_fails():
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ys = np.round(8 * np.cos(theta) + 20).astype(float)
    xs = np.round(8 * np.sin(theta) + 20).astype(float)

    result = is_linear_and_long(ys, xs, min_r_score=0.196, min_length=5.0, avg_size=3)

    assert result is False


@pytest.mark.parametrize("vertical", [False, True], ids=["horizontal", "vertical"])
def test_is_linear_and_long_wide_contrail_along_an_axis_passes(vertical):
    rows, cols = _strip(40, 3, vertical=vertical)

    result = is_linear_and_long(rows, cols, min_r_score=0.196, min_length=5.0, avg_size=3)

    assert result is True, "a straight 3 px wide contrail must pass whatever its direction"


# --- is_pixel_aligned ---


def test_is_pixel_aligned_few_pixels_always_true():
    current_ys = [0, 1, 2]
    current_xs = [0, 1, 2]

    result = is_pixel_aligned(
        current_ys, current_xs, new_y=50, new_x=50, lookback=10, threshold=1.0
    )

    assert result is True


def test_is_pixel_aligned_on_line_returns_true():
    current_ys = [2 * x for x in range(10)]
    current_xs = list(range(10))

    result = is_pixel_aligned(
        current_ys, current_xs, new_y=22, new_x=11, lookback=10, threshold=2.0
    )

    assert result is True


def test_is_pixel_aligned_far_off_line_returns_false():
    current_ys = [2 * x for x in range(10)]
    current_xs = list(range(10))

    result = is_pixel_aligned(current_ys, current_xs, new_y=0, new_x=11, lookback=10, threshold=2.0)

    assert result is False


# --- pairwise_iou ---


def test_pairwise_iou_identical_is_one():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])

    iou = pairwise_iou((rows, cols), (rows, cols))

    assert iou == pytest.approx(1.0)


def test_pairwise_iou_disjoint_is_zero():
    rows1 = np.array([0, 1, 2])
    cols1 = np.array([0, 1, 2])
    rows2 = np.array([10, 11, 12])
    cols2 = np.array([10, 11, 12])

    iou = pairwise_iou((rows1, cols1), (rows2, cols2))

    assert iou == pytest.approx(0.0)


def test_pairwise_iou_partial_overlap_correct_value():
    rows1 = np.array([0, 1, 2, 3, 4])
    cols1 = np.array([0, 1, 2, 3, 4])
    rows2 = np.array([0, 1, 2, 5, 6])
    cols2 = np.array([0, 1, 2, 5, 6])

    iou = pairwise_iou((rows1, cols1), (rows2, cols2))

    assert iou == pytest.approx(3 / 7)


def test_pairwise_iou_symmetric():
    rows1 = np.array([0, 1, 2, 3])
    cols1 = np.array([0, 1, 2, 3])
    rows2 = np.array([2, 3, 4, 5])
    cols2 = np.array([2, 3, 4, 5])

    iou_ab = pairwise_iou((rows1, cols1), (rows2, cols2))
    iou_ba = pairwise_iou((rows2, cols2), (rows1, cols1))

    assert iou_ab == pytest.approx(iou_ba)


# --- candidate_score ---


def test_candidate_score_perfect_line_high_score():
    rows = np.arange(20, dtype=float)
    cols = np.arange(20, dtype=float)

    score = candidate_score(rows, cols, saturation=1.0, avg_size=3)

    assert score == pytest.approx(1.0, abs=1e-6)


def test_candidate_score_round_blob_is_zero():
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rows = 10.0 * np.cos(theta)
    cols = 10.0 * np.sin(theta)

    score = candidate_score(rows, cols, saturation=40.0, avg_size=3)

    assert score == pytest.approx(0.0, abs=0.01)


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

    score = candidate_score(rows, cols, saturation=1.0, avg_size=3)

    assert score == 0.0
