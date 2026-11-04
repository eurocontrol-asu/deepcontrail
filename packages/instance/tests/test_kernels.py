"""Tests for deepcontrail.instance._kernels — DoG line-kernel synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from deepcontrail.instance._kernels import (
    detect_lines,
    get_angles,
    synthesize_line_kernel,
    zero_sum_normalize,
)

# --- zero_sum_normalize ---


def test_zero_sum_normalize_positive_weights_sum_to_one():
    kernel = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [-3.0, 0.0, -1.0]])

    normalized = zero_sum_normalize(kernel)

    pos_sum = normalized[normalized > 0].sum()
    assert pos_sum == pytest.approx(1.0)


def test_zero_sum_normalize_negative_weights_sum_to_minus_one():
    kernel = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [-3.0, 0.0, -1.0]])

    normalized = zero_sum_normalize(kernel)

    neg_sum = normalized[normalized <= 0].sum()
    assert neg_sum == pytest.approx(-1.0)


def test_zero_sum_normalize_all_positive_returns_unchanged():
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])

    normalized = zero_sum_normalize(kernel)

    assert np.array_equal(normalized, kernel)


# --- synthesize_line_kernel ---


def test_synthesize_line_kernel_shape_matches_size():
    for size in (5, 7, 11):
        kernel = synthesize_line_kernel(0.0, size, sigma1=1.5, sigma_delta=1.0)

        assert kernel.shape == (size, size)


def test_synthesize_line_kernel_is_zero_sum_normalized():
    kernel = synthesize_line_kernel(30.0, 7, sigma1=1.5, sigma_delta=1.0)

    pos_sum = kernel[kernel > 0].sum()
    neg_sum = -kernel[kernel <= 0].sum()

    assert pos_sum == pytest.approx(1.0)
    assert neg_sum == pytest.approx(1.0)


def test_synthesize_line_kernel_0_degrees_is_symmetric_horizontally():
    kernel = synthesize_line_kernel(0.0, 7, sigma1=1.5, sigma_delta=1.0)

    assert np.allclose(kernel, np.flip(kernel, axis=1))


def test_synthesize_line_kernel_90_degrees_is_symmetric_vertically():
    kernel = synthesize_line_kernel(90.0, 7, sigma1=1.5, sigma_delta=1.0)

    assert np.allclose(kernel, np.flip(kernel, axis=0))


# --- get_angles ---


def test_get_angles_count_matches_num_kernels():
    angles = get_angles(25)

    assert len(angles) == 25


def test_get_angles_range_is_0_to_180_exclusive():
    angles = get_angles(25)

    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] < 180.0


def test_get_angles_evenly_spaced():
    angles = get_angles(9)

    diffs = np.diff(angles)
    assert np.allclose(diffs, diffs[0])


# --- detect_lines ---


def test_detect_lines_empty_mask_returns_all_false():
    mask = np.zeros((32, 32), dtype=np.float32)

    detected = detect_lines(
        mask, num_kernels=25, kernel_size=7, sigma1=1.884, sigma_delta=1.443, threshold=0.00052
    )

    assert detected.shape == (32, 32)
    assert not detected.any()


def test_detect_lines_horizontal_line_detected():
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[16, 5:28] = 1.0

    detected = detect_lines(
        mask, num_kernels=25, kernel_size=7, sigma1=1.884, sigma_delta=1.443, threshold=0.00052
    )

    assert detected[16, 8:24].all()


def test_detect_lines_output_shape_matches_input():
    rng = np.random.default_rng(0)
    mask = rng.random((40, 48)).astype(np.float32)

    detected = detect_lines(
        mask, num_kernels=25, kernel_size=7, sigma1=1.884, sigma_delta=1.443, threshold=0.00052
    )

    assert detected.shape == mask.shape


def test_detect_lines_high_response_threshold_drops_thin_line():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[32, 8:56] = 1.0

    kept = detect_lines(
        mask, num_kernels=25, kernel_size=7, sigma1=1.884, sigma_delta=1.443, threshold=0.1
    )
    dropped = detect_lines(
        mask, num_kernels=25, kernel_size=7, sigma1=1.884, sigma_delta=1.443, threshold=0.9
    )

    assert kept.any(), "a 1 px line at value 1.0 must survive a response threshold below its peak"
    assert not dropped.any(), "above the line's peak filter response it must be dropped"
