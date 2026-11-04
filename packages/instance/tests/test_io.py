"""Tests for deepcontrail.instance.io (building instance masks, saving results)."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from deepcontrail.core.readers import extract_all_annotator_instances


def _line(row: int, size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[row, 2:14] = True
    return mask


def test_instance_masks_from_label_image_round_trips_through_the_reader():
    from deepcontrail.instance.io import instance_masks_from_label_image

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[3, 2:14] = 1
    labels[9, 2:14] = 7  # labels need not be consecutive

    raw = instance_masks_from_label_image(labels)

    assert raw.shape == (2, 16, 16, 1)
    assert raw.dtype == np.int8
    contrails = extract_all_annotator_instances(raw)[0]
    assert [np.flatnonzero(m).tolist() for m in contrails] == [
        np.flatnonzero(_line(3)).tolist(),
        np.flatnonzero(_line(9)).tolist(),
    ]


def test_instance_masks_from_label_image_without_contrails_is_empty():
    from deepcontrail.instance.io import instance_masks_from_label_image

    raw = instance_masks_from_label_image(np.zeros((8, 8), dtype=np.uint8))

    assert raw.shape == (0, 8, 8, 1)


def test_instance_masks_from_annotators_marks_missing_contrails_as_not_drawn():
    from deepcontrail.instance.io import instance_masks_from_annotators

    raw = instance_masks_from_annotators([[_line(3), _line(9)], [_line(5)], []])

    assert raw.shape == (2, 16, 16, 3)
    per_annotator = extract_all_annotator_instances(raw)
    assert [len(c) for c in per_annotator] == [2, 1, 0]
    assert (raw[1, :, :, 1] == -1).all(), "annotator 1 drew no second contrail"
    assert (raw[:, :, :, 2] == -1).all(), "annotator 2 drew nothing"


def test_instance_masks_from_annotators_rejects_mismatched_shapes():
    from deepcontrail.instance.io import instance_masks_from_annotators

    with pytest.raises(ValueError, match="shape"):
        instance_masks_from_annotators([[_line(3)], [_line(3, size=8)]])


def test_instance_masks_from_annotators_needs_a_size_when_nobody_drew():
    from deepcontrail.instance.io import instance_masks_from_annotators

    assert instance_masks_from_annotators([[], []], shape=(8, 8)).shape == (0, 8, 8, 2)
    with pytest.raises(ValueError, match="shape"):
        instance_masks_from_annotators([[], []])


def test_to_coco_results_keeps_overlapping_contrails_and_is_json_ready():
    mask_utils = pytest.importorskip("pycocotools.mask")
    from deepcontrail.instance.io import to_coco_results

    crossing = np.zeros((16, 16), dtype=bool)
    crossing[:, 7] = True  # shares one pixel with _line(3)
    masks, scores = [_line(3), crossing], [0.9, 0.4]

    results = to_coco_results(masks, scores, image_id="scene_1")

    json.dumps(results)
    assert [r["score"] for r in results] == [0.9, 0.4]
    assert all(r["image_id"] == "scene_1" and r["category_id"] == 1 for r in results)
    decoded = [
        mask_utils.decode({**r["segmentation"], "counts": r["segmentation"]["counts"].encode()})
        for r in results
    ]
    assert all(np.array_equal(d.astype(bool), m) for d, m in zip(decoded, masks, strict=True))
    assert decoded[0][3, 7], "the shared pixel stays in the first contrail"
    assert decoded[1][3, 7], "the shared pixel stays in the crossing contrail"


def test_to_coco_results_rejects_misaligned_scores():
    pytest.importorskip("pycocotools.mask")
    from deepcontrail.instance.io import to_coco_results

    with pytest.raises(ValueError, match="score"):
        to_coco_results([_line(3)], [0.5, 0.6], image_id=1)


def test_instance_masks_from_label_image_rejects_a_stack():
    from deepcontrail.instance.io import instance_masks_from_label_image

    with pytest.raises(ValueError, match="2-D"):
        instance_masks_from_label_image(np.zeros((2, 8, 8), dtype=np.int32))


def test_to_coco_results_without_pycocotools_names_the_extra(monkeypatch):
    from deepcontrail.instance.io import to_coco_results

    monkeypatch.setitem(sys.modules, "pycocotools", None)
    monkeypatch.setitem(sys.modules, "pycocotools.mask", None)

    with pytest.raises(ImportError, match=r"deepcontrail-instance\[map\]"):
        to_coco_results([_line(3)], [0.5], image_id=1)
