"""Integration tests for deepcontrail.instance.mannstein — the public pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from deepcontrail.instance import (
    detect_instances,
)
from deepcontrail.instance.config import MannsteinConfig

H = W = 64


def _horizontal_line(y: int, col0: int = 5, col1: int = 56) -> np.ndarray:
    """All-zero 64x64 mask with a solid horizontal line at row ``y``."""
    mask = np.zeros((H, W), dtype=np.float32)
    mask[y, col0:col1] = 1.0
    return mask


# --- detect_instances ---


def test_detect_instances_empty_mask_returns_empty_list():
    mask = np.zeros((H, W), dtype=np.float32)

    instances, _ = detect_instances(mask)

    assert instances == []


def test_detect_instances_single_horizontal_line_returns_one_instance():
    mask = _horizontal_line(32)

    instances, _ = detect_instances(mask)

    assert len(instances) == 1
    instance = instances[0]
    assert instance.dtype == bool
    assert instance.shape == (H, W)
    assert int(instance[32, 5:56].sum()) >= 10


def test_detect_instances_two_separated_lines_returns_two_instances():
    mask = np.zeros((H, W), dtype=np.float32)
    mask[16, 5:56] = 1.0
    mask[48, 5:56] = 1.0

    instances, _ = detect_instances(mask)

    assert len(instances) == 2


def test_detect_instances_output_masks_are_boolean():
    mask = _horizontal_line(32)

    instances, _ = detect_instances(mask)

    assert len(instances) == 1
    assert instances[0].dtype == bool


def test_detect_instances_output_shape_matches_input():
    mask = np.zeros((H, W), dtype=np.float32)
    mask[20, 5:56] = 1.0
    mask[44, 5:56] = 1.0

    instances, _ = detect_instances(mask)

    assert len(instances) == 2
    assert all(m.shape == (H, W) for m in instances)


def test_detect_instances_default_config_when_none():
    mask = _horizontal_line(32)

    instances, _ = detect_instances(mask)  # config defaults to MannsteinConfig()

    assert isinstance(instances, list)


@pytest.mark.parametrize("vertical", [False, True], ids=["horizontal", "vertical"])
def test_detect_instances_finds_a_wide_contrail_along_either_axis(vertical):
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[31:34, 5:56] = 1
    mask = mask.T if vertical else mask

    instances, scores = detect_instances(mask)

    assert len(instances) == 1, "a straight 3 px wide contrail along an axis is one contrail"
    assert scores[0] > 0.5


def test_detect_instances_score_does_not_depend_on_the_direction():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[32, 5:56] = 1

    _, horizontal = detect_instances(mask)
    _, vertical = detect_instances(mask.T)

    assert horizontal == pytest.approx(vertical)


def test_detect_instances_line_filter_drops_pixels_with_a_weak_response():
    mask = _horizontal_line(32)
    filtered = MannsteinConfig(line_filter=True, response_threshold=0.99)

    with_filter, _ = detect_instances(mask, filtered)
    without_filter, _ = detect_instances(mask, filtered.model_copy(update={"line_filter": False}))

    assert with_filter == [], "a 1 px line has a weak response: the filter removes it"
    assert len(without_filter) == 1, "without the filter every mask pixel is kept"


def test_detect_instances_line_filter_is_off_by_default():
    assert MannsteinConfig().line_filter is False


def test_detect_instances_deterministic():
    mask = np.zeros((H, W), dtype=np.float32)
    mask[16, 5:56] = 1.0
    mask[48, 5:56] = 1.0

    n_first = len(detect_instances(mask)[0])
    n_second = len(detect_instances(mask)[0])

    assert n_first == n_second
    assert n_first == 2


# --- scores ---


def test_detect_instances_scores_aligned_lengths():
    mask = np.zeros((H, W), dtype=np.float32)
    mask[16, 5:56] = 1.0
    mask[48, 5:56] = 1.0

    masks, scores = detect_instances(mask)

    assert len(masks) == len(scores) == 2


def test_detect_instances_scores_in_unit_interval():
    mask = np.zeros((H, W), dtype=np.float32)
    mask[16, 5:56] = 1.0
    mask[48, 5:56] = 1.0

    _masks, scores = detect_instances(mask)

    assert scores
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_detect_instances_scores_empty_mask_returns_empty_lists():
    mask = np.zeros((H, W), dtype=np.float32)

    masks, scores = detect_instances(mask)

    assert masks == []
    assert scores == []


# --- filter_avg_size (separate end-averaging for the length filter) ---


def _short_line_config(**kw: int | None) -> MannsteinConfig:
    """Lenient settings so only the end-averaging size decides whether a short line survives."""
    return MannsteinConfig(min_pixels=3, min_length=2.0, min_straightness=0.0, **kw)


def test_filter_avg_size_none_reuses_fit_avg_size():
    mask = _horizontal_line(32, col0=20, col1=28)

    instances, _ = detect_instances(mask, _short_line_config(fit_avg_size=5, filter_avg_size=None))

    assert instances == [], "8 pixels < 2 * fit_avg_size: the length/linearity filter rejects it"


def test_filter_avg_size_overrides_fit_avg_size_for_the_filter_only():
    mask = _horizontal_line(32, col0=20, col1=28)

    instances, scores = detect_instances(
        mask, _short_line_config(fit_avg_size=5, filter_avg_size=3)
    )

    assert len(instances) == 1, "8 pixels >= 2 * filter_avg_size: the filter keeps it"
    assert scores == [0.0], "scoring still uses fit_avg_size (8 < 10 points gives score 0)"


# --- input validation ---


def test_detect_instances_accepts_bool_integer_and_float_binary_masks():
    mask = _horizontal_line(10)

    results = [detect_instances(mask.astype(t))[0] for t in (bool, np.uint8, np.float32)]

    assert [len(r) for r in results] == [1, 1, 1]


def test_detect_instances_rejects_a_probability_map():
    probs = _horizontal_line(10).astype(np.float32) * 0.4

    with pytest.raises(ValueError, match="detect_instances_from_probabilities"):
        detect_instances(probs)


@pytest.mark.parametrize("shape", [(4, 64, 64), (64,)])
def test_detect_instances_rejects_non_2d_input(shape):
    with pytest.raises(ValueError, match="2-D"):
        detect_instances(np.zeros(shape, dtype=np.uint8))


def test_detect_instances_rejects_non_finite_values():
    mask = _horizontal_line(10).astype(np.float32)
    mask[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        detect_instances(mask)


def test_public_api_has_one_plain_mannstein_call():
    import deepcontrail.instance as instance

    assert "detect_instances_labeled" not in instance.__all__
    assert "detect_instances_with_scores" not in instance.__all__
