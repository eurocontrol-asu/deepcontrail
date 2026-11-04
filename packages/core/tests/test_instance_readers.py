"""Tests for the instance-mask reader in deepcontrail.core.readers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _seed(
    n_instances: int,
    height: int,
    width: int,
    n_annotators: int,
    value: int = -1,
) -> np.ndarray:
    """Return an all-``value`` int8 instance array to be modified in place."""
    return np.full((n_instances, height, width, n_annotators), value, dtype=np.int8)


# --- load_instance_masks ---


def test_load_instance_masks_returns_correct_shape(tmp_path: Path):
    from deepcontrail.core.readers import load_instance_masks

    raw = np.random.RandomState(0).randint(-1, 2, size=(3, 16, 16, 2)).astype(np.int8)
    path = tmp_path / "inst.npy"
    np.save(path, raw)

    result = load_instance_masks(path)

    assert result.shape == (3, 16, 16, 2)
    assert result.dtype == np.int8
    np.testing.assert_array_equal(result, raw)


# --- extract_annotator_instances ---


def test_extract_annotator_instances_returns_correct_count():
    from deepcontrail.core.readers import extract_annotator_instances

    raw = _seed(5, 16, 16, 2)
    raw[0, 2:5, 2:5, 0] = 1
    raw[1, 6:9, 6:9, 0] = 1
    raw[2, 10:13, 10:13, 0] = 1  # annotator 1 stays all -1 -> dropped for annotator 1

    masks = extract_annotator_instances(raw, annotator=0)

    assert len(masks) == 3


def test_extract_annotator_instances_filters_empty_instances():
    from deepcontrail.core.readers import extract_annotator_instances

    raw = _seed(3, 16, 16, 2)
    raw[0, 3:6, 3:6, 0] = 1  # active foreground -> kept
    raw[1] = 0  # annotator 0 sees the instance but marks no foreground -> dropped
    # instance 2 is all -1 for annotator 0 -> dropped as absent

    masks = extract_annotator_instances(raw, annotator=0)

    assert len(masks) == 1
    # The kept instance's full (H, W) slice must equal its foreground.
    np.testing.assert_array_equal(masks[0], raw[0, :, :, 0] == 1)


def test_extract_annotator_instances_returns_boolean_masks():
    from deepcontrail.core.readers import extract_annotator_instances

    raw = _seed(2, 16, 16, 2)
    raw[0, 2:6, 2:6, 0] = 1
    raw[1, 8:12, 8:12, 0] = 1

    masks = extract_annotator_instances(raw, annotator=0)

    assert len(masks) == 2
    for mask in masks:
        assert mask.dtype == bool
        assert mask.shape == (16, 16)


def test_extract_annotator_instances_masks_match_input():
    from deepcontrail.core.readers import extract_annotator_instances

    raw = _seed(2, 16, 16, 2)
    raw[0, 2:5, 2:5, 0] = 1

    masks = extract_annotator_instances(raw, annotator=0)

    assert len(masks) == 1
    np.testing.assert_array_equal(masks[0], raw[0, :, :, 0] == 1)


def test_extract_annotator_instances_empty_array():
    from deepcontrail.core.readers import extract_annotator_instances

    raw = _seed(0, 16, 16, 2)

    masks = extract_annotator_instances(raw, annotator=0)

    assert masks == []


def test_extract_annotator_instances_invalid_annotator_raises():
    from deepcontrail.core.readers import extract_annotator_instances

    raw = _seed(3, 16, 16, 4)

    with pytest.raises((IndexError, ValueError)):
        extract_annotator_instances(raw, annotator=5)


def test_load_and_extract_roundtrip(tmp_path: Path):
    from deepcontrail.core.readers import (
        extract_annotator_instances,
        load_instance_masks,
    )

    raw = _seed(3, 16, 16, 2)
    raw[0, 1:4, 1:4, 0] = 1
    raw[1, 5:9, 5:9, 0] = 1
    raw[2, 8:11, 8:11, 1] = 1  # active only for annotator 1

    path = tmp_path / "inst.npy"
    np.save(path, raw)

    loaded = load_instance_masks(path)
    masks = extract_annotator_instances(loaded, annotator=0)

    expected = {
        tuple((raw[i, :, :, 0] == 1).ravel().tolist())
        for i in range(loaded.shape[0])
        if np.any(raw[i, :, :, 0] == 1)
    }
    actual = {tuple(m.ravel().tolist()) for m in masks}

    assert expected == actual


# --- extract_all_annotator_instances ---


def test_extract_all_annotator_instances_returns_list_of_lists():
    from deepcontrail.core.readers import extract_all_annotator_instances

    raw = _seed(3, 16, 16, 2)
    raw[0, 2:5, 2:5, 0] = 1
    raw[1, 6:9, 6:9, 0] = 1
    raw[0, 1:4, 1:4, 1] = 1  # annotator 1 has one active instance

    result = extract_all_annotator_instances(raw)

    assert len(result) == 2
    for annotator_masks in result:
        assert isinstance(annotator_masks, list)
        for mask in annotator_masks:
            assert mask.dtype == bool
            assert mask.shape == (16, 16)
    assert len(result[0]) == 2
    assert len(result[1]) == 1


# --- format checks on load ---


def test_load_instance_masks_rejects_wrong_number_of_dimensions(tmp_path: Path):
    from deepcontrail.core.readers import load_instance_masks

    path = tmp_path / "inst.npy"
    np.save(path, np.zeros((16, 16, 2), dtype=np.int8))

    with pytest.raises(ValueError, match="instances, H, W, annotators"):
        load_instance_masks(path)


def test_load_instance_masks_rejects_values_outside_minus_one_to_one(tmp_path: Path):
    from deepcontrail.core.readers import load_instance_masks

    raw = _seed(1, 8, 8, 1, value=0)
    raw[0, 2, 2, 0] = 2
    path = tmp_path / "inst.npy"
    np.save(path, raw)

    with pytest.raises(ValueError, match="-1, 0 or 1"):
        load_instance_masks(path)


def test_load_instance_masks_accepts_a_scene_without_contrails(tmp_path: Path):
    from deepcontrail.core.readers import extract_all_annotator_instances, load_instance_masks

    path = tmp_path / "inst.npy"
    np.save(path, np.zeros((0, 8, 8, 3), dtype=np.int8))

    raw = load_instance_masks(path)

    assert raw.shape == (0, 8, 8, 3), "a scene without contrails has zero instances"
    assert extract_all_annotator_instances(raw) == [[], [], []]
