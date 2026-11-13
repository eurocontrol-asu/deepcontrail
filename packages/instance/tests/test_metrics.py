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
