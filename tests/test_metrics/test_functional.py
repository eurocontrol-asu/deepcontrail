"""Tests for deepcontrail.metrics.functional — dice_coefficient, iou_score."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# dice_coefficient
# ---------------------------------------------------------------------------


def test_dice_coefficient_perfect_prediction_is_one(
    perfect_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    pred, target = perfect_pair

    assert dice_coefficient(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_dice_coefficient_no_overlap_is_near_zero(
    no_overlap_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    pred, target = no_overlap_pair

    assert dice_coefficient(pred, target) < 1e-4


def test_dice_coefficient_empty_masks_is_one(
    empty_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    pred, target = empty_pair

    assert dice_coefficient(pred, target) == 1.0


def test_dice_coefficient_partial_overlap_expected_value(
    partial_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    pred, target = partial_pair
    expected = 2 * 2 / (3 + 2)  # 0.8

    assert dice_coefficient(pred, target) == pytest.approx(expected, abs=1e-5)


def test_dice_coefficient_symmetric() -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    a = np.array([1, 1, 1, 0])
    b = np.array([1, 0, 0, 0])

    assert dice_coefficient(a, b) == pytest.approx(dice_coefficient(b, a), abs=1e-10), (
        "Dice must be symmetric"
    )


def test_dice_coefficient_2d_input() -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    pred = np.array([[1, 0], [1, 0]])
    target = np.array([[1, 0], [0, 0]])

    expected = 2 * 1 / (2 + 1)  # 2/3
    assert dice_coefficient(pred, target) == pytest.approx(expected, abs=1e-5)


def test_dice_coefficient_batched_input() -> None:
    from deepcontrail.metrics.functional import dice_coefficient

    pred = np.zeros((2, 4, 4))
    target = np.zeros((2, 4, 4))
    pred[0, 0, 0] = 1
    target[0, 0, 0] = 1

    assert dice_coefficient(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_dice_coefficient_handles_torch_tensor() -> None:
    torch = pytest.importorskip("torch")
    from deepcontrail.metrics.functional import dice_coefficient

    pred = torch.tensor([1, 1, 0, 0])
    target = torch.tensor([1, 1, 0, 0])

    assert dice_coefficient(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_to_numpy_reduces_torch_tensor_to_float32() -> None:
    torch = pytest.importorskip("torch")
    from deepcontrail.metrics.functional import _to_numpy

    tensor = torch.tensor([1.0, 0.0, 1.0, 1.0])

    arr = _to_numpy(tensor)

    assert arr.dtype == np.float32, "binary-mask metrics must not carry float64 tensors to host"


def test_to_numpy_downcasts_python_float64_array_to_float32() -> None:
    from deepcontrail.metrics.functional import _to_numpy

    arr = _to_numpy(np.array([1.0, 0.0], dtype=np.float64))

    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# iou_score
# ---------------------------------------------------------------------------


def test_iou_score_perfect_prediction_is_one(
    perfect_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import iou_score

    pred, target = perfect_pair

    assert iou_score(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_iou_score_no_overlap_is_near_zero(
    no_overlap_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import iou_score

    pred, target = no_overlap_pair

    assert iou_score(pred, target) < 1e-4


def test_iou_score_empty_masks_is_one(
    empty_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import iou_score

    pred, target = empty_pair

    assert iou_score(pred, target) == 1.0


def test_iou_score_partial_overlap_expected_value(
    partial_pair: tuple[np.ndarray, np.ndarray],
) -> None:
    from deepcontrail.metrics.functional import iou_score

    pred, target = partial_pair
    expected = 2 / 3  # intersection=2, union=3

    assert iou_score(pred, target) == pytest.approx(expected, abs=1e-5)


def test_iou_score_symmetric() -> None:
    from deepcontrail.metrics.functional import iou_score

    a = np.array([1, 1, 1, 0])
    b = np.array([1, 0, 0, 0])

    assert iou_score(a, b) == pytest.approx(iou_score(b, a), abs=1e-10), "IoU must be symmetric"
