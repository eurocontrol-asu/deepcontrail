"""Tests for the oriented line-detection kernels and the line filter."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import oaconvolve

from deepcontrail.instance._lines import (
    _balance,
    build_kernels,
    detect_line_pixels,
    line_orientations,
)

H = W = 64


def _horizontal_line(y: int = 32, col0: int = 5, col1: int = 56) -> np.ndarray:
    """All-zero 64x64 float mask with a solid horizontal line at row ``y``."""
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y, col0:col1] = 1.0
    return mask


def _vertical_line(x: int = 32, row0: int = 5, row1: int = 56) -> np.ndarray:
    """All-zero 64x64 float mask with a solid vertical line at column ``x``."""
    mask = np.zeros((H, W), dtype=np.float32)
    mask[row0:row1, x] = 1.0
    return mask


def _isolated_pixel() -> np.ndarray:
    """All-zero 64x64 float mask with a single on pixel."""
    mask = np.zeros((H, W), dtype=np.float32)
    mask[32, 32] = 1.0
    return mask


def test_line_orientations_are_evenly_spaced_below_180():
    orientations = line_orientations(8)

    assert orientations.shape == (8,)
    assert orientations[0] == 0.0
    assert orientations[-1] == pytest.approx(157.5)
    assert np.all(np.diff(orientations) == pytest.approx(22.5))


def test_build_kernels_shape_matches_configuration():
    kernels = build_kernels(num_kernels=6, kernel_size=15, sigma1=1.6, sigma_delta=0.7)

    assert kernels.shape == (6, 15, 15)


def test_line_kernel_entries_balance_to_plus_and_minus_one():
    kernels = build_kernels(num_kernels=8, kernel_size=15, sigma1=1.6, sigma_delta=0.7)

    for kernel in kernels:
        pos = float(np.sum(np.where(kernel > 0.0, kernel, 0.0)))
        neg = float(np.sum(np.where(kernel <= 0.0, kernel, 0.0)))
        assert pos == pytest.approx(1.0)
        assert neg == pytest.approx(-1.0)


def test_line_kernel_responds_most_along_its_orientation():
    num_kernels = 4  # orientations 0, 45, 90, 135 degrees
    kernels = build_kernels(num_kernels=num_kernels, kernel_size=15, sigma1=1.6, sigma_delta=0.7)
    # orientation 90 deg (index 2) prefers a horizontal line; 0 deg (index 0) a vertical one.
    horiz, vert = 2, 0
    center = (H // 2, W // 2)

    responses_h = np.array(
        [
            oaconvolve(_horizontal_line(), kernels[k], mode="same")[center]
            for k in range(num_kernels)
        ]
    )
    responses_v = np.array(
        [oaconvolve(_vertical_line(), kernels[k], mode="same")[center] for k in range(num_kernels)]
    )

    assert responses_h.argmax() == horiz
    assert responses_v.argmax() == vert


def test_line_detection_keeps_a_straight_line_and_drops_isolated_pixels():
    # The oriented DoG responds ~15x more strongly to a line (interior response ~1.0) than to an
    # isolated point (~0.07, its single positive-lobe cell), so a threshold in between keeps the
    # line and drops the point.
    threshold = 0.2
    detected_line = detect_line_pixels(
        _horizontal_line(),
        num_kernels=24,
        kernel_size=15,
        sigma1=1.6,
        sigma_delta=0.7,
        threshold=threshold,
    )
    detected_pixel = detect_line_pixels(
        _isolated_pixel(),
        num_kernels=24,
        kernel_size=15,
        sigma1=1.6,
        sigma_delta=0.7,
        threshold=threshold,
    )

    assert detected_line[_horizontal_line() > 0].all()
    assert not detected_pixel.any()


def test_line_detection_returns_empty_for_an_empty_mask():
    detected = detect_line_pixels(
        np.zeros((0, 0), dtype=np.float32),
        num_kernels=8,
        kernel_size=15,
        sigma1=1.6,
        sigma_delta=0.7,
        threshold=1e-4,
    )

    assert detected.shape == (0, 0)
    assert detected.dtype == bool


def test_balance_scales_each_lobe_to_plus_and_minus_one():
    kernel = build_kernels(num_kernels=6, kernel_size=11, sigma1=1.4, sigma_delta=0.5)[0]
    balanced = _balance(kernel)

    assert float(np.sum(np.where(balanced > 0.0, balanced, 0.0))) == pytest.approx(1.0)
    assert float(np.sum(np.where(balanced <= 0.0, balanced, 0.0))) == pytest.approx(-1.0)


def test_balance_divides_positives_and_negatives_by_their_own_lobe_sum():
    diff = np.array([[0.5, -0.5], [0.5, -0.5]])  # positive total 1.0, negative total 1.0

    balanced = _balance(diff)

    assert balanced.tolist() == [[0.5, -0.5], [0.5, -0.5]]


def test_balance_returns_a_kernel_unchanged_when_it_has_no_negative_weights():
    diff = np.array([[1.0, 2.0], [3.0, 4.0]])  # no negatives -> negative total 0

    assert np.array_equal(_balance(diff), diff)


def test_balance_returns_a_kernel_unchanged_when_it_has_no_positive_weights():
    diff = np.array([[-1.0, -2.0], [-3.0, -4.0]])  # no positives -> positive total 0

    assert np.array_equal(_balance(diff), diff)
