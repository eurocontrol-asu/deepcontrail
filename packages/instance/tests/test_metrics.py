"""Tests for deepcontrail.instance.metrics (per-annotator COCO mAP, precision, recall, F1)."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from deepcontrail.instance.metrics import match_instances

H = W = 16


# The COCO mAP needs the [map] extra (pycocotools); matching does not.
requires_map = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None, reason="needs deepcontrail-instance[map]"
)


def _hline(h: int, w: int, row: int, x0: int = 2, x1: int = 14) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[row, x0:x1] = True
    return mask


# --- match_instances ---


def test_match_instances_perfect_match():
    preds = [_hline(H, W, 3, 2, 8), _hline(H, W, 10, 4, 12)]
    gt = [_hline(H, W, 3, 2, 8), _hline(H, W, 10, 4, 12)]

    tp, fp, fn = match_instances(preds, gt)

    assert (tp, fp, fn) == (2, 0, 0)


def test_match_instances_disjoint():
    preds = [_hline(H, W, 3, 2, 8), _hline(H, W, 10, 4, 12)]
    gt = [_hline(H, W, 14, 2, 8), _hline(H, W, 1, 2, 8)]

    tp, fp, fn = match_instances(preds, gt)

    assert (tp, fp, fn) == (0, 2, 2)


def test_match_instances_one_to_one():
    gt = [_hline(H, W, 3, 2, 8)]
    preds = [_hline(H, W, 3, 2, 8), _hline(H, W, 3, 2, 8)]

    tp, fp, fn = match_instances(preds, gt)

    assert (tp, fp, fn) == (1, 1, 0)


def test_match_instances_below_threshold_is_unmatched():
    preds = [_hline(H, W, 3, 2, 8)]
    gt = [_hline(H, W, 3, 2, 4)]

    tp, fp, fn = match_instances(preds, gt, iou_threshold=0.5)

    assert (tp, fp, fn) == (0, 1, 1)


def test_match_instances_empty_inputs():
    gt = [_hline(H, W, 3, 2, 8), _hline(H, W, 10, 4, 12)]
    preds = [_hline(H, W, 3, 2, 8)]

    assert match_instances([], gt) == (0, 0, 2)
    assert match_instances(preds, []) == (0, 1, 0)
    assert match_instances([], []) == (0, 0, 0)


# --- evaluate_annotator_samples (pooled per-annotator mAP) ---


@requires_map
def test_evaluate_annotator_samples_perfect_predictions():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gts = [[_hline(H, W, 3, 2, 10), _hline(H, W, 9, 2, 10)], [_hline(H, W, 5, 2, 12)]]
    preds = [list(gts[0]), list(gts[1])]

    r = evaluate_annotator_samples(preds, gts, [[0.9, 0.8], [0.7]])

    assert r["map"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in r["ap_per_threshold"].values())
    assert r["f1"] == pytest.approx(1.0)
    assert r["n_samples"] == 2


@requires_map
def test_evaluate_annotator_samples_skips_samples_without_gt():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    line = _hline(H, W, 3, 2, 10)

    r = evaluate_annotator_samples([[line], [_hline(H, W, 8, 2, 10)]], [[line], []])

    assert r["n_samples"] == 1, "an annotator who drew nothing is not a sample"
    assert r["n_skipped_empty"] == 1
    assert r["map"] == pytest.approx(1.0)
    assert r["precision"] == pytest.approx(1.0), "predictions on a skipped sample are not scored"


@requires_map
def test_evaluate_annotator_samples_missed_instance_lowers_map_and_recall():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gts = [[_hline(H, W, 3, 2, 10), _hline(H, W, 9, 2, 10)]]

    r = evaluate_annotator_samples([[_hline(H, W, 3, 2, 10)]], gts, [[0.9]])

    assert r["map"] < 1.0
    assert r["recall"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(1.0)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 0, 1)


@requires_map
def test_evaluate_annotator_samples_map_is_mean_of_per_threshold_ap():
    from deepcontrail.instance.metrics import (
        DEFAULT_IOU_THRESHOLDS,
        evaluate_annotator_samples,
    )

    gts = [[_hline(H, W, 3, 2, 10), _hline(H, W, 9, 2, 10)], [_hline(H, W, 5, 2, 12)]]
    preds = [
        [_hline(H, W, 3, 2, 8), _hline(H, W, 9, 2, 10)],
        [_hline(H, W, 5, 2, 7), _hline(H, W, 12)],
    ]
    scores = [[0.9, 0.4], [0.6, 0.8]]

    r = evaluate_annotator_samples(preds, gts, scores)

    assert list(r["ap_per_threshold"]) == list(DEFAULT_IOU_THRESHOLDS)
    assert r["map"] == pytest.approx(np.mean(list(r["ap_per_threshold"].values())), abs=1e-6)


@requires_map
def test_evaluate_annotator_samples_scores_rank_predictions():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gts = [[_hline(H, W, 3, 2, 10)]]
    preds = [[_hline(H, W, 3, 2, 10), _hline(H, W, 9, 2, 10)]]

    good = evaluate_annotator_samples(preds, gts, [[0.9, 0.1]])
    bad = evaluate_annotator_samples(preds, gts, [[0.1, 0.9]])

    assert good["map"] > bad["map"], "a false positive ranked first lowers AP"


@requires_map
def test_evaluate_annotator_samples_without_per_threshold_and_empty_input():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    fast = evaluate_annotator_samples([[_hline(H, W, 3)]], [[_hline(H, W, 3)]], per_threshold=False)
    none = evaluate_annotator_samples([[]], [[]])

    assert "ap_per_threshold" not in fast
    assert fast["map"] == pytest.approx(1.0)
    assert none["map"] == 0.0
    assert none["n_samples"] == 0


def test_evaluate_annotator_samples_rejects_mismatched_lengths():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    with pytest.raises(ValueError, match="one entry per sample"):
        evaluate_annotator_samples([[]], [[], []])


# --- empty samples kept (real conditions) ---


def _mixed_samples() -> tuple[list, list, list]:
    """Hits, a shifted (partial) hit, a false positive, a miss; scores all different."""
    preds = [
        [_hline(16, 16, 3), _hline(16, 16, 9)],
        [_hline(16, 16, 5, 4, 14)],
        [_hline(16, 16, 2), _hline(16, 16, 12)],
        [],
    ]
    gts = [
        [_hline(16, 16, 3), _hline(16, 16, 9)],
        [_hline(16, 16, 5)],
        [_hline(16, 16, 2)],
        [_hline(16, 16, 6)],
    ]
    scores = [[0.9, 0.35], [0.6], [0.8, 0.7], []]
    return preds, gts, scores


@requires_map
def test_include_empty_without_empty_samples_equals_the_official_metric():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    preds, gts, scores = _mixed_samples()

    official = evaluate_annotator_samples(preds, gts, scores)
    kept = evaluate_annotator_samples(preds, gts, scores, include_empty=True)

    assert kept["map"] == pytest.approx(official["map"], abs=1e-9), (
        "registering every sample changes nothing when no sample is empty"
    )
    for t, ap in official["ap_per_threshold"].items():
        assert kept["ap_per_threshold"][t] == pytest.approx(ap, abs=1e-9)
    assert kept["f1"] == official["f1"]


@requires_map
def test_include_empty_counts_false_detections_on_empty_samples():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    preds, gts, scores = _mixed_samples()
    preds.append([_hline(16, 16, 7)])
    gts.append([])
    scores.append([0.95])

    skipped = evaluate_annotator_samples(preds, gts, scores)
    kept = evaluate_annotator_samples(preds, gts, scores, include_empty=True)

    assert skipped["n_skipped_empty"] == 1
    assert kept["n_samples"] == 5
    assert kept["n_skipped_empty"] == 0
    assert kept["fp"] == skipped["fp"] + 1, "the detection on the empty sample is a false one"
    assert kept["map"] < skipped["map"], "a confident false detection lowers the mAP"


@requires_map
def test_include_empty_sample_without_detection_changes_nothing():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    preds, gts, scores = _mixed_samples()
    preds.append([])
    gts.append([])
    scores.append([])

    skipped = evaluate_annotator_samples(preds, gts, scores)
    kept = evaluate_annotator_samples(preds, gts, scores, include_empty=True)

    assert kept["map"] == pytest.approx(skipped["map"], abs=1e-9)
    assert kept["fp"] == skipped["fp"]


# --- torch-free metrics ---


@requires_map
def test_metrics_run_without_torch():
    import subprocess
    import sys

    code = (
        "import sys, numpy as np\n"
        "from deepcontrail.instance.metrics import evaluate_annotator_samples\n"
        "m = np.zeros((8, 8), bool); m[2, 1:7] = True\n"
        "r = evaluate_annotator_samples([[m]], [[m]], [[0.9]])\n"
        "assert abs(r['map'] - 1.0) < 1e-9, r\n"
        "print(sys.modules.get('torch') is not None)\n"
    )

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert out.stdout.strip() == "False", "the metrics need pycocotools only, never torch"


@requires_map
def test_pooled_map_is_the_standard_101_point_coco_ap():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gt = [_hline(16, 16, 3), _hline(16, 16, 9)]
    preds = [[gt[0], _hline(16, 16, 13), gt[1]]]
    scores = [[0.9, 0.8, 0.7]]

    r = evaluate_annotator_samples(preds, [gt], scores)

    # Ranked: hit, false, hit. Recall 0.5 at precision 1, recall 1 at precision 2/3. COCO reads
    # the best precision at or beyond each of its 101 recall points: 51 points (0 to 0.5) get 1,
    # the other 50 get 2/3. Same at every IoU threshold, since the hits are exact.
    expected = (51 * 1.0 + 50 * 2 / 3) / 101
    assert r["map"] == pytest.approx(expected, abs=1e-12)
    assert all(ap == pytest.approx(expected, abs=1e-12) for ap in r["ap_per_threshold"].values())


def _dots(n: int, size: int = 32) -> list[np.ndarray]:
    """``n`` one-pixel masks at distinct positions."""
    masks = []
    for k in range(n):
        m = np.zeros((size, size), dtype=bool)
        m[divmod(k, size)] = True
        masks.append(m)
    return masks


@requires_map
def test_evaluate_annotator_samples_warns_when_a_sample_exceeds_100_detections():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gt = [_dots(1)]

    with pytest.warns(UserWarning, match="100 detections"):
        evaluate_annotator_samples([_dots(101)], gt, per_threshold=False)


@requires_map
def test_evaluate_annotator_samples_is_silent_at_100_detections(recwarn):
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    evaluate_annotator_samples([_dots(100)], [_dots(1)], per_threshold=False)

    assert not [w for w in recwarn if "100 detections" in str(w.message)]


def test_evaluate_annotator_samples_without_pycocotools_names_the_extra(monkeypatch):
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    for name in ("pycocotools", "pycocotools.mask", "pycocotools.coco", "pycocotools.cocoeval"):
        monkeypatch.setitem(sys.modules, name, None)
    m = _hline(16, 16, 3)

    with pytest.raises(ImportError, match=r"deepcontrail-instance\[map\]"):
        evaluate_annotator_samples([[m]], [[m]], [[0.9]])


# --- large frames: fast matching, detection cap, ignored ground truth ---


def _pixel(row: int, col: int, size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[row, col] = True
    return mask


def _brute_force_match(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float
) -> tuple[int, int, int]:
    """Reproduce the historical double loop over full masks, greedy by IoU (the reference)."""
    n_pred, n_gt = len(pred_masks), len(gt_masks)
    pairs: list[tuple[float, int, int]] = []
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = float(intersection / union) if union else 0.0
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    matched_pred = [False] * n_pred
    matched_gt = [False] * n_gt
    tp = 0
    for _iou, i, j in pairs:
        if not matched_pred[i] and not matched_gt[j]:
            matched_pred[i] = True
            matched_gt[j] = True
            tp += 1
    return tp, n_pred - tp, n_gt - tp


def test_match_instances_fast_path_equals_brute_force():
    rng = np.random.default_rng(0)
    h = w = 64

    def rect() -> np.ndarray:
        mask = np.zeros((h, w), dtype=bool)
        mask[
            rng.integers(0, 40) : rng.integers(0, 40) + rng.integers(6, 30),
            rng.integers(0, 40) : rng.integers(0, 40) + rng.integers(6, 30),
        ] = True
        return mask

    preds = [rect() for _ in range(15)]
    gts = [rect() for _ in range(12)]

    for threshold in (0.0, 0.5, 0.75):
        assert match_instances(preds, gts, threshold) == _brute_force_match(
            preds, gts, threshold
        ), f"the fast bbox matching must return the exact brute-force (tp, fp, fn) at {threshold}"


def test_match_instances_disjoint_boxes_never_match():
    pred = _hline(16, 16, 2, 2, 6)  # row 2, cols 2-5
    gt = _hline(16, 16, 13, 10, 14)  # row 13, cols 10-13: no box overlap at all

    tp, fp, fn = match_instances([pred], [gt], iou_threshold=0.5)

    assert (tp, fp, fn) == (0, 1, 1), "masks in non-overlapping boxes cannot share a pixel"


def test_match_instances_empty_masks_never_match():
    empty = np.zeros((16, 16), dtype=bool)
    line = _hline(16, 16, 3, 2, 8)

    tp, fp, fn = match_instances([empty, line], [empty, line], iou_threshold=0.5)

    assert (tp, fp, fn) == (1, 1, 1), "an empty mask has area 0 and never matches anything"


@requires_map
def test_evaluate_max_detections_keeps_more_than_100():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gt = _dots(150)  # 150 disjoint one-pixel contrails, all detected perfectly

    with pytest.warns(UserWarning, match="more than 100"):
        default = evaluate_annotator_samples([list(gt)], [gt], per_threshold=False)
    assert default["map"] < 1.0, "the 100-detection cap cannot cover 150 contrails"

    raised = evaluate_annotator_samples([list(gt)], [gt], max_detections=200, per_threshold=False)
    assert raised["map"] == pytest.approx(1.0), "raising the cap keeps every perfect detection"


def test_evaluate_max_detections_below_one_raises():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    with pytest.raises(ValueError, match="at least 1"):
        evaluate_annotator_samples([[_hline(16, 16, 3)]], [[_hline(16, 16, 3)]], max_detections=0)


@requires_map
def test_evaluate_gt_ignore_detection_on_ignored_contrail_is_not_a_false_positive():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    a, b = _pixel(2, 2), _pixel(10, 10)
    preds, scores = [[a, b]], [[0.8, 0.9]]  # the detection on b is ranked first

    flagged = evaluate_annotator_samples(preds, [[a, b]], scores, gt_ignore=[[False, True]])
    absent = evaluate_annotator_samples(preds, [[a]], scores)

    assert flagged["map"] == pytest.approx(1.0), (
        "a detection on an ignored contrail is neither a hit nor a false positive"
    )
    assert absent["map"] < 1.0, "the same detection is a false positive once that GT is gone"


@requires_map
def test_evaluate_sample_with_only_ignored_gt_counts_as_empty():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    line = _hline(16, 16, 3, 2, 8)
    preds = gts = [[line], [line]]
    scores = [[0.9], [0.9]]

    without = evaluate_annotator_samples(preds, gts, scores)
    r = evaluate_annotator_samples(preds, gts, scores, gt_ignore=[[], [True]])

    assert r["n_samples"] == 1, "a sample whose only GT is ignored is treated as empty"
    assert r["n_skipped_empty"] == 1
    assert without["n_samples"] == 2, "the same sample is kept once nothing is ignored"


@requires_map
def test_evaluate_without_new_options_is_unchanged():
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    gts = [[_hline(H, W, 3, 2, 10), _hline(H, W, 9, 2, 10)], [_hline(H, W, 5, 2, 12)]]
    preds = [list(gts[0]), list(gts[1])]
    scores = [[0.9, 0.8], [0.7]]

    r = evaluate_annotator_samples(preds, gts, scores)

    assert r["map"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in r["ap_per_threshold"].values())
    assert r["precision"] == pytest.approx(1.0)
    assert r["recall"] == pytest.approx(1.0)
    assert r["f1"] == pytest.approx(1.0)
    assert (r["tp"], r["fp"], r["fn"]) == (3, 0, 0)
    assert r["n_samples"] == 2
    assert r["n_skipped_empty"] == 0


# --- coco_map_from_rles (the RLE entry point the streaming script reuses) ---


@requires_map
def test_coco_map_from_rles_equals_dense_evaluation():
    """Encoding the masks and calling the RLE function matches the dense per-annotator metric."""
    import pycocotools.mask as mask_utils

    from deepcontrail.instance.metrics import (
        DEFAULT_IOU_THRESHOLDS,
        coco_map_from_rles,
        evaluate_annotator_samples,
    )

    rng = np.random.default_rng(0)
    h = w = 24

    def dots() -> list[np.ndarray]:
        masks = []
        for _ in range(int(rng.integers(0, 4))):
            m = np.zeros((h, w), dtype=bool)
            m[rng.integers(0, h), rng.integers(0, w)] = True
            masks.append(m)
        return masks

    preds, gts, scores = [], [], []
    for _ in range(8):
        p, g = dots(), dots()
        preds.append(p)
        gts.append(g)
        scores.append([float(rng.random()) for _ in p])

    # include_empty keeps every sample, so the dense metric pools exactly the samples we pass.
    dense = evaluate_annotator_samples(preds, gts, scores, include_empty=True)

    def enc(m: np.ndarray) -> dict:
        return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))

    rle_map, rle_ap = coco_map_from_rles(
        [[enc(m) for m in p] for p in preds],
        [[enc(m) for m in g] for g in gts],
        scores,
        [(h, w)] * len(preds),
    )

    assert rle_map == pytest.approx(dense["map"])
    expected_ap = np.array([dense["ap_per_threshold"][t] for t in DEFAULT_IOU_THRESHOLDS])
    np.testing.assert_allclose(rle_ap, expected_ap, atol=1e-12)


@requires_map
def test_coco_map_from_rles_respects_max_detections():
    """The per-sample detection cap truncates recall, exactly as in the dense metric."""
    import pycocotools.mask as mask_utils

    from deepcontrail.instance.metrics import coco_map_from_rles

    def enc(m: np.ndarray) -> dict:
        return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))

    dots = _dots(150)  # 150 disjoint one-pixel contrails, all detected perfectly
    gt = [enc(m) for m in dots]
    pred = [enc(m) for m in dots]
    scores = [[1.0] * 150]

    capped = coco_map_from_rles([pred], [gt], scores, [(32, 32)])
    assert capped[0] < 1.0, "the 100-detection cap cannot cover 150 contrails"

    raised = coco_map_from_rles([pred], [gt], scores, [(32, 32)], max_detections=200)
    assert raised[0] == pytest.approx(1.0), "raising the cap keeps every perfect detection"
