"""Tests for deepcontrail.instance.labels (consensus contrails and derived labels)."""

from __future__ import annotations

import numpy as np
import pytest

# --- filter / partition of GT instances by the GT semantic mask ---


def _box(y0: int, y1: int, x0: int, x1: int, size: int = 8) -> np.ndarray:
    """Boolean ``size`` x ``size`` mask that is True on ``[y0:y1, x0:x1]``."""
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_filter_keeps_instance_mostly_inside_semantic_mask():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    instance = _box(2, 3, 0, 8)
    semantic = _box(2, 3, 2, 8)

    out = partition_gt_instances_by_semantic([[[instance]]], [semantic])[0]

    assert len(out[0][0]) == 1, "6 of 8 pixels inside (75%) >= 50%: kept"
    assert np.array_equal(out[0][0][0], instance), "kept instances are kept whole, not clipped"


def test_filter_fraction_boundary_is_inclusive():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    instance = _box(2, 3, 0, 8)
    semantic = _box(2, 3, 4, 8)

    at = partition_gt_instances_by_semantic(
        [[[instance]]], [semantic], min_overlap_fraction=0.5, tolerance_px=0
    )[0]
    above = partition_gt_instances_by_semantic(
        [[[instance]]], [semantic], min_overlap_fraction=0.51, tolerance_px=0
    )[0]

    assert len(at[0][0]) == 1, "exactly half inside is kept at 0.5"
    assert above[0][0] == [], "below the fraction is dropped"


def test_filter_drops_crossing_instance_mostly_outside_mask():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    consensus_contrail = _box(0, 8, 4, 5)
    crossing = _box(3, 4, 0, 8)

    out = partition_gt_instances_by_semantic([[[crossing]]], [consensus_contrail])[0]

    assert out[0][0] == [], "1 of 8 pixels inside (a crossing point) is not enough to keep it"


def test_filter_drops_instance_fully_outside_mask():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    out = partition_gt_instances_by_semantic([[[_box(0, 1, 0, 8)]]], [_box(6, 8, 0, 8)])[0]

    assert out[0][0] == []


def test_filter_empty_semantic_mask_drops_all_in_scene():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    scene = [[_box(0, 1, 0, 8), _box(3, 4, 0, 8)], [_box(5, 6, 0, 8)]]

    out = partition_gt_instances_by_semantic([scene], [np.zeros((8, 8), dtype=bool)])[0]

    assert out == [[[], []]], "slots are kept, every instance in them is dropped"


def test_filter_preserves_slot_and_scene_structure_and_does_not_mutate():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    inside, outside = _box(2, 3, 0, 8), _box(6, 7, 0, 8)
    semantic = _box(1, 4, 0, 8)
    all_gt = [[[inside, outside], [outside], []], [[outside, inside]]]
    snapshot = [[[m.copy() for m in slot] for slot in scene] for scene in all_gt]

    out = partition_gt_instances_by_semantic(all_gt, [semantic, semantic])[0]

    assert [len(scene) for scene in out] == [3, 1], "scene and slot counts are preserved"
    assert [[len(slot) for slot in scene] for scene in out] == [[1, 0, 0], [1]]
    assert all(
        np.array_equal(a, b)
        for scene, ref in zip(all_gt, snapshot, strict=True)
        for slot, ref_slot in zip(scene, ref, strict=True)
        for a, b in zip(slot, ref_slot, strict=True)
    ), "the input masks are not modified"
    assert [len(slot) for slot in all_gt[0]] == [2, 1, 0], "the input lists are not modified"


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_filter_rejects_bad_fraction(fraction):
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    with pytest.raises(ValueError, match="min_overlap_fraction"):
        partition_gt_instances_by_semantic(
            [[[_box(0, 1, 0, 8)]]], [_box(0, 1, 0, 8)], min_overlap_fraction=fraction
        )[0]


def test_filter_rejects_mismatched_scene_counts():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    with pytest.raises(ValueError, match="scene"):
        partition_gt_instances_by_semantic([[[_box(0, 1, 0, 8)]]], [])[0]


def test_partition_returns_kept_and_dropped_that_together_are_the_input():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    inside, outside = _box(2, 3, 0, 8), _box(6, 7, 0, 8)
    all_gt = [[[inside, outside], [outside]]]
    semantic = [_box(1, 4, 0, 8)]

    kept, dropped = partition_gt_instances_by_semantic(all_gt, semantic)

    assert [[len(s) for s in sc] for sc in kept] == [[1, 0]]
    assert [[len(s) for s in sc] for sc in dropped] == [[1, 1]]
    assert np.array_equal(kept[0][0][0], inside)
    assert np.array_equal(dropped[0][0][0], outside)


def test_filter_tolerance_keeps_contrail_drawn_one_pixel_off():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    consensus = _box(3, 4, 0, 8)
    shifted = _box(4, 5, 0, 8)

    strict = partition_gt_instances_by_semantic([[[shifted]]], [consensus], tolerance_px=0)[0]
    tolerant = partition_gt_instances_by_semantic([[[shifted]]], [consensus], tolerance_px=1)[0]

    assert strict[0][0] == [], "0% of its pixels are inside the consensus"
    assert len(tolerant[0][0]) == 1, "100% within 1 px: the same contrail drawn 1 px off"


def test_filter_tolerance_still_drops_crossing_contrail():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    consensus_contrail = _box(0, 8, 4, 5)
    crossing = _box(3, 4, 0, 8)

    out = partition_gt_instances_by_semantic([[[crossing]]], [consensus_contrail], tolerance_px=1)[
        0
    ]

    assert out[0][0] == [], "3 of 8 pixels within 1 px of the consensus: still set aside"


def test_filter_tolerance_is_four_connected():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    consensus = _box(3, 4, 3, 4)
    diagonal = _box(4, 5, 4, 5)

    out = partition_gt_instances_by_semantic([[[diagonal]]], [consensus], tolerance_px=1)[0]

    assert out[0][0] == [], "a diagonal neighbour is 2 steps away in 4-connectivity"


def test_filter_rejects_negative_tolerance():
    from deepcontrail.instance.labels import partition_gt_instances_by_semantic

    with pytest.raises(ValueError, match="tolerance_px"):
        partition_gt_instances_by_semantic(
            [[[_box(0, 1, 0, 8)]]], [_box(0, 1, 0, 8)], tolerance_px=-1
        )[0]


# --- build_consensus_instances ---


def _hline(row: int, x0: int = 2, x1: int = 14, size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[row, x0:x1] = True
    return mask


def _vline(col: int, y0: int = 2, y1: int = 14, size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, col] = True
    return mask


def test_consensus_merges_annotator_copies_into_one_instance_shaped_by_consensus():
    from deepcontrail.instance.labels import build_consensus_instances

    consensus = _hline(5)
    per_slot = [[_hline(5)], [_hline(5) | _hline(6)], [_hline(4) | _hline(5)], [_hline(5)]]

    out = build_consensus_instances(per_slot, consensus)

    assert len(out) == 1, "four annotators drew the same contrail: one consensus instance"
    assert np.array_equal(out[0], consensus), "its shape is the consensus mask, not any annotator's"


def test_consensus_keeps_contrail_drawn_by_majority_only():
    from deepcontrail.instance.labels import build_consensus_instances

    consensus = _hline(5) | _hline(10)
    per_slot = [[_hline(5), _hline(10)], [_hline(5), _hline(10)], [_hline(5)], [_hline(5)]]

    out = build_consensus_instances(per_slot, consensus)

    assert len(out) == 1, "row 10 is drawn by 2 of 4 annotators (not more than half): excluded"
    assert out[0][5].any()
    assert not out[0][10].any()


def test_consensus_three_of_four_is_a_majority():
    from deepcontrail.instance.labels import build_consensus_instances

    consensus = _hline(5)
    per_slot = [[_hline(5)], [_hline(5)], [_hline(5)], []]

    assert len(build_consensus_instances(per_slot, consensus)) == 1


def test_consensus_crossing_contrails_stay_separate_and_share_pixels():
    from deepcontrail.instance.labels import build_consensus_instances

    h, v = _hline(8), _vline(8)
    consensus = h | v
    per_slot = [[h, v], [h, v], [h, v], [v, h]]

    out = build_consensus_instances(per_slot, consensus)

    assert len(out) == 2, "two crossing contrails agreed by all annotators stay two instances"
    assert all(m[8, 8] for m in out), "both keep the crossing pixel, as in the annotations"


def test_consensus_ignores_contrails_outside_the_consensus():
    from deepcontrail.instance.labels import build_consensus_instances

    consensus = _hline(5)
    per_slot = [[_hline(5), _hline(12)], [_hline(5), _hline(12)], [_hline(5), _hline(12)]]

    out = build_consensus_instances(per_slot, consensus)

    assert len(out) == 1, "row 12 is outside the consensus mask: never a consensus instance"


def test_consensus_empty_inputs_and_no_mutation():
    from deepcontrail.instance.labels import build_consensus_instances

    per_slot = [[_hline(5)], [_hline(5)]]
    snapshot = [[m.copy() for m in slot] for slot in per_slot]

    assert build_consensus_instances([], np.zeros((16, 16), bool)) == []
    assert build_consensus_instances(per_slot, np.zeros((16, 16), bool)) == []
    out = build_consensus_instances(per_slot, _hline(5))

    assert len(out) == 1
    assert out[0].dtype == bool
    assert all(
        np.array_equal(a, b)
        for s, r in zip(per_slot, snapshot, strict=True)
        for a, b in zip(s, r, strict=True)
    )


# --- labels derived from the instance masks alone ---


def _raw(rows_per_annotator: list[list[int]], size: int = 12) -> np.ndarray:
    """(N, H, W, A) instance masks: annotator a drew one horizontal line per row in its list."""
    n = max((len(r) for r in rows_per_annotator), default=0)
    raw = np.full((n, size, size, len(rows_per_annotator)), -1, dtype=np.int8)
    for a, rows in enumerate(rows_per_annotator):
        for k, row in enumerate(rows):
            raw[k, :, :, a] = 0
            raw[k, row, 2:10, a] = 1
    return raw


def test_annotator_masks_from_instances_is_the_union_per_annotator():
    from deepcontrail.instance.labels import annotator_masks_from_instances

    raw = _raw([[2, 6], [], [6]])

    masks = annotator_masks_from_instances(raw)

    assert masks.shape == (12, 12, 3)
    assert masks.dtype == np.uint8
    assert masks[2, 2:10, 0].all(), "first contrail of annotator 0"
    assert masks[6, 2:10, 0].all(), "second contrail of annotator 0"
    assert not masks[..., 1].any(), "an annotator who drew nothing has an empty mask"
    assert masks[..., 2].sum() == 8


def test_agreement_and_consensus_mask_from_instances():
    from deepcontrail.instance.labels import agreement_from_instances, consensus_mask_from_instances

    raw = _raw([[2, 6], [], [6]])

    agreement = agreement_from_instances(raw)
    consensus = consensus_mask_from_instances(raw)

    assert agreement.dtype == np.float32
    assert agreement[6, 5] == pytest.approx(2 / 3)
    assert agreement[2, 5] == pytest.approx(1 / 3)
    assert consensus[6, 2:10].all(), "drawn by 2 of 3 annotators: more than half"
    assert not consensus[2].any(), "drawn by 1 of 3: not the consensus"


def test_derived_labels_of_a_scene_without_contrails_are_empty():
    from deepcontrail.instance.labels import (
        agreement_from_instances,
        annotator_masks_from_instances,
        consensus_instances,
    )

    raw = np.zeros((0, 12, 12, 4), dtype=np.int8)

    assert annotator_masks_from_instances(raw).shape == (12, 12, 4)
    assert not agreement_from_instances(raw).any()
    assert consensus_instances(raw) == []


def test_single_annotator_consensus_is_every_contrail_they_drew():
    from deepcontrail.instance.labels import consensus_instances

    raw = _raw([[2, 6]])

    consensus = consensus_instances(raw)

    assert len(consensus) == 2, "with one annotator, every drawn contrail is the consensus"


def test_consensus_instances_keep_the_majority_contrails():
    from deepcontrail.instance.labels import consensus_instances

    raw = _raw([[2, 6], [6], [6]])

    consensus = consensus_instances(raw)

    assert len(consensus) == 1
    assert consensus[0][6, 2:10].all()
