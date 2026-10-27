"""Shared fixtures for metric tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def perfect_pair() -> tuple[np.ndarray, np.ndarray]:
    """Identical pred and target — Dice should be 1.0."""
    mask = np.array([1, 1, 0, 0], dtype=np.float64)
    return mask, mask.copy()


@pytest.fixture
def no_overlap_pair() -> tuple[np.ndarray, np.ndarray]:
    """Completely disjoint pred and target — Dice should be ~0.0."""
    pred = np.array([1, 1, 0, 0], dtype=np.float64)
    target = np.array([0, 0, 1, 1], dtype=np.float64)
    return pred, target


@pytest.fixture
def empty_pair() -> tuple[np.ndarray, np.ndarray]:
    """Both masks all-zero — Dice should be 1.0 (vacuous truth)."""
    return np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)


@pytest.fixture
def partial_pair() -> tuple[np.ndarray, np.ndarray]:
    """Partial overlap: intersection=2, pred_sum=3, target_sum=2."""
    pred = np.array([1, 1, 1, 0], dtype=np.float64)
    target = np.array([1, 1, 0, 0], dtype=np.float64)
    return pred, target
