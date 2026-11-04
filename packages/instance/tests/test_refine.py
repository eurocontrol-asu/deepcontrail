"""Tests for deepcontrail.instance.refine — growth and probability-weighted scores."""

from __future__ import annotations

import numpy as np
import pytest

from deepcontrail.instance.refine import grow_instances, probability_weighted_scores

H = W = 9


def _mask(*pixels: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for y, x in pixels:
        mask[y, x] = True
    return mask


def test_grow_instances_zero_steps_returns_copies_unchanged():
    masks = [_mask((4, 4))]
    probs = np.ones((H, W), dtype=np.float32)

    grown = grow_instances(masks, probs, [1.0], threshold=0.5, steps=0)

    assert np.array_equal(grown[0], masks[0])
    assert grown[0] is not masks[0], "growth must not alias the input masks"


def test_grow_instances_four_connectivity_adds_only_edge_neighbours():
    probs = np.ones((H, W), dtype=np.float32)

    grown = grow_instances([_mask((4, 4))], probs, [1.0], threshold=0.5, steps=1, connectivity=4)

    assert grown[0].sum() == 5
    for y, x in ((3, 4), (5, 4), (4, 3), (4, 5)):
        assert grown[0][y, x], f"edge neighbour {(y, x)} must be added"
    assert not grown[0][3, 3], "a diagonal neighbour is not 4-connected"


def test_grow_instances_eight_connectivity_adds_diagonals():
    probs = np.ones((H, W), dtype=np.float32)

    grown = grow_instances([_mask((4, 4))], probs, [1.0], threshold=0.5, steps=1, connectivity=8)

    assert grown[0].sum() == 9


def test_grow_instances_respects_probability_threshold():
    probs = np.zeros((H, W), dtype=np.float32)
    probs[4, 5] = 0.6
    probs[4, 3] = 0.4

    grown = grow_instances([_mask((4, 4))], probs, [1.0], threshold=0.5, steps=1, connectivity=4)

    assert grown[0][4, 5], "a neighbour above the threshold is added"
    assert not grown[0][4, 3], "a neighbour at or below the threshold is not added"


def test_grow_instances_keeps_original_pixels_below_threshold():
    probs = np.zeros((H, W), dtype=np.float32)

    grown = grow_instances([_mask((4, 4))], probs, [1.0], threshold=0.5, steps=2)

    assert grown[0][4, 4], "growth never removes an instance's own pixels"


def test_grow_instances_multiple_steps_grow_further():
    probs = np.ones((H, W), dtype=np.float32)

    grown = grow_instances([_mask((4, 4))], probs, [1.0], threshold=0.5, steps=2, connectivity=4)

    assert grown[0][4, 6]
    assert grown[0][2, 4]
    assert grown[0].sum() == 13


def test_grow_instances_contested_pixel_goes_to_higher_score():
    probs = np.ones((H, W), dtype=np.float32)
    left, right = _mask((4, 3)), _mask((4, 5))

    grown = grow_instances([left, right], probs, [0.2, 0.9], threshold=0.5, steps=1, connectivity=4)

    assert grown[1][4, 4], "the contested pixel goes to the higher-scoring instance"
    assert not grown[0][4, 4], "each newly grown pixel belongs to one instance only"


def test_grow_instances_does_not_steal_other_instance_pixels():
    probs = np.ones((H, W), dtype=np.float32)
    left, right = _mask((4, 3)), _mask((4, 4))

    grown = grow_instances([left, right], probs, [0.9, 0.2], threshold=0.5, steps=1, connectivity=4)

    assert grown[1][4, 4], "an instance keeps its own pixels"
    assert not grown[0][4, 4], "growth does not take pixels that already belong to an instance"


def test_grow_instances_rejects_bad_arguments():
    probs = np.ones((H, W), dtype=np.float32)
    masks = [_mask((4, 4))]

    with pytest.raises(ValueError, match="connectivity"):
        grow_instances(masks, probs, [1.0], threshold=0.5, steps=1, connectivity=6)
    with pytest.raises(ValueError, match="scores"):
        grow_instances(masks, probs, [1.0, 0.5], threshold=0.5, steps=1)
    with pytest.raises(ValueError, match="steps"):
        grow_instances(masks, probs, [1.0], threshold=0.5, steps=-1)


def test_grow_instances_empty_list_returns_empty_list():
    assert grow_instances([], np.ones((H, W), np.float32), [], threshold=0.5, steps=1) == []


def test_probability_weighted_scores_multiplies_by_mean_probability():
    probs = np.zeros((H, W), dtype=np.float32)
    probs[4, 4], probs[4, 5] = 0.8, 0.4
    mask = _mask((4, 4), (4, 5))

    scores = probability_weighted_scores([mask], probs, [0.5])

    assert scores == pytest.approx([0.5 * 0.6])


def test_probability_weighted_scores_empty_mask_scores_zero():
    probs = np.ones((H, W), dtype=np.float32)

    assert probability_weighted_scores([np.zeros((H, W), bool)], probs, [0.7]) == [0.0]
