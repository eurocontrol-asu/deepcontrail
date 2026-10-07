"""Tests for evaluation utilities (threshold search + global evaluation)."""

from __future__ import annotations

import numpy as np

from deepcontrail.training import find_optimal_threshold


def test_find_optimal_threshold_perfect_predictions_give_dice_one() -> None:
    targets = np.array([0, 0, 1, 1, 1, 0], dtype=np.uint8)
    probs = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    threshold, dice = find_optimal_threshold(probs, targets)

    assert dice == 1.0, f"Perfect predictions should yield Dice=1.0, got {dice}"
    assert 0.0 < threshold < 1.0


def test_find_optimal_threshold_all_zeros_gives_dice_one() -> None:
    targets = np.zeros(100, dtype=np.uint8)
    probs = np.zeros(100, dtype=np.float32)

    _threshold, dice = find_optimal_threshold(probs, targets)

    assert dice == 1.0, "All-zero targets with all-zero probs should give Dice=1.0"


def test_find_optimal_threshold_between_zero_and_one() -> None:
    rng = np.random.default_rng(42)
    targets = rng.integers(0, 2, size=200).astype(np.uint8)
    probs = rng.random(200).astype(np.float32)

    threshold, _dice = find_optimal_threshold(probs, targets)

    assert 0.0 < threshold < 1.0, f"Threshold should be in (0, 1), got {threshold}"


def test_find_optimal_threshold_custom_thresholds_respected() -> None:
    targets = np.array([0, 1, 1, 0], dtype=np.uint8)
    probs = np.array([0.1, 0.6, 0.7, 0.2], dtype=np.float32)
    custom = np.array([0.3, 0.5, 0.8])

    threshold, _dice = find_optimal_threshold(probs, targets, thresholds=custom)

    assert threshold in [0.3, 0.5, 0.8], (
        f"Threshold should be one of the custom values, got {threshold}"
    )


def test_find_optimal_threshold_returns_best_dice_value() -> None:
    targets = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
    probs = np.array([0.9, 0.8, 0.1, 0.2, 0.7], dtype=np.float32)

    threshold, dice = find_optimal_threshold(probs, targets)

    preds = (probs > threshold).astype(np.uint8)
    tp = float(np.sum(preds & targets))
    expected_dice = 2.0 * tp / (float(np.sum(preds)) + float(np.sum(targets)))
    assert abs(dice - expected_dice) < 1e-6, (
        f"Returned dice ({dice}) doesn't match recomputed ({expected_dice})"
    )
