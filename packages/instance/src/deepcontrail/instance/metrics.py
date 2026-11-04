"""Instance metrics across independent annotators: COCO mAP, precision, recall and F1.

The mAP needs the ``[map]`` extra (pycocotools); torch is never needed.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np

DEFAULT_IOU_THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
COCO_MAX_DETECTIONS = 100
"""Detections per sample kept by the COCO mAP (pycocotools default; extra ones are dropped)."""


def match_instances(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy one-to-one IoU matching of predicted and ground-truth masks.

    Parameters
    ----------
    pred_masks
        Predicted boolean instance masks, each ``(H, W)``.
    gt_masks
        Ground-truth boolean instance masks, each ``(H, W)``.
    iou_threshold
        Minimum IoU for a pred/GT pair to count as a match.

    Returns
    -------
    tuple[int, int, int]
        ``(tp, fp, fn)``. Every pred and GT is matched at most once, always
        taking the highest-IoU available pair first. Unmatched preds are false
        positives and unmatched GT are false negatives. Empty inputs are valid.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0

    matched_pred = [False] * n_pred
    matched_gt = [False] * n_gt
    pairs: list[tuple[float, int, int]] = []
    for i, pred in enumerate(pred_masks):
        for j, gt in enumerate(gt_masks):
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = float(intersection / union) if union else 0.0
            if iou >= iou_threshold:
                pairs.append((iou, i, j))

    pairs.sort(key=lambda pair: pair[0], reverse=True)
    tp = 0
    for _iou, i, j in pairs:
        if not matched_pred[i] and not matched_gt[j]:
            matched_pred[i] = True
            matched_gt[j] = True
            tp += 1

    return tp, n_pred - tp, n_gt - tp


def evaluate_annotator_samples(
    pred_masks: list[list[np.ndarray]],
    gt_masks: list[list[np.ndarray]],
    pred_scores: list[list[float]] | None = None,
    iou_thresholds: tuple[float, ...] | list[float] = DEFAULT_IOU_THRESHOLDS,
    match_iou: float = 0.5,
    per_threshold: bool = True,
    include_empty: bool = False,
) -> dict[str, Any]:
    """Score instance predictions per annotator sample (pooled COCO segmentation mAP).

    One sample is one annotator on one scene. Samples whose ground truth is empty are
    skipped (the official protocol) unless ``include_empty``. All remaining samples are
    pooled into a single COCO segmentation mAP computation (one image per sample, single
    class), averaged over ``iou_thresholds``. Precision, recall and F1 from one-to-one
    matching at ``match_iou``, pooled over the same samples, are returned as a reference.

    With ``include_empty``, every sample is kept and every detection on an empty sample is a
    false detection, as in real use where most scenes have no contrail.

    The computation is the standard COCO segmentation AP of pycocotools (101 recall points, at
    most 100 detections per sample, area "all", with a warning when a sample has more); torch is
    not needed. Every sample is registered
    with COCO, so ``include_empty`` counts the detections on empty samples (torchmetrics'
    ``MeanAveragePrecision`` drops them).

    Parameters
    ----------
    pred_masks
        ``pred_masks[i]`` is the list of predicted boolean ``(H, W)`` masks for sample ``i``.
    gt_masks
        ``gt_masks[i]`` is the list of GT boolean masks of sample ``i`` (one annotator).
    pred_scores
        Confidence per predicted mask, aligned with ``pred_masks``; ``None`` = 1.0 for all.
    iou_thresholds
        IoU thresholds averaged by the mAP. Defaults to 0.25, 0.35, ..., 0.75.
    match_iou
        IoU threshold of the reference one-to-one matching.
    per_threshold
        Also return the AP at each threshold.
    include_empty
        Keep samples with an empty ground truth and count their detections as false.

    Returns
    -------
    dict[str, Any]
        ``map``, ``ap_per_threshold`` (when requested), ``precision``, ``recall``, ``f1``,
        ``tp``, ``fp``, ``fn``, ``n_samples`` and ``n_skipped_empty``.

    Raises
    ------
    ValueError
        If ``pred_masks``, ``gt_masks`` (and ``pred_scores``) have different lengths.
    """
    if len(pred_masks) != len(gt_masks) or (
        pred_scores is not None and len(pred_scores) != len(pred_masks)
    ):
        raise ValueError("pred_masks, gt_masks and pred_scores must have one entry per samples")
    keep = [i for i, gt in enumerate(gt_masks) if gt or include_empty]
    preds = [pred_masks[i] for i in keep]
    gts = [gt_masks[i] for i in keep]
    scores = None if pred_scores is None else [pred_scores[i] for i in keep]

    tp = fp = fn = 0
    for p, g in zip(preds, gts, strict=True):
        a, b, c = match_instances(p, g, match_iou)
        tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    crowded = sum(len(p) > COCO_MAX_DETECTIONS for p in preds)
    if crowded:
        warnings.warn(
            f"{crowded} sample(s) have more than {COCO_MAX_DETECTIONS} detections; the COCO mAP "
            f"keeps only the {COCO_MAX_DETECTIONS} detections with the highest scores per sample",
            UserWarning,
            stacklevel=2,
        )
    pooled, ap = _coco_ap_all_samples(preds, gts, scores, list(iou_thresholds))
    result: dict[str, Any] = {
        "map": pooled,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_samples": len(keep),
        "n_skipped_empty": len(gt_masks) - len(keep),
    }
    if per_threshold:
        result["ap_per_threshold"] = dict(zip(iou_thresholds, ap.tolist(), strict=True))
    return result


def _coco_ap_all_samples(
    preds: list[list[np.ndarray]],
    gts: list[list[np.ndarray]],
    scores: list[list[float]] | None,
    thresholds: list[float],
) -> tuple[float, np.ndarray]:
    """Pooled mAP and AP per threshold, every sample registered in COCO (empty GT included).

    Standard COCO segmentation AP (pycocotools defaults: 101 recall points, 100 detections,
    area "all"), with ``thresholds`` as IoU thresholds. Zeros when there is nothing to score.
    """
    import contextlib
    import io

    try:
        import pycocotools.mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as err:
        raise ImportError(
            "the COCO mAP needs pycocotools: pip install 'deepcontrail-instance[map]'"
        ) from err

    images: list[dict[str, Any]] = []
    gt_anns: list[dict[str, Any]] = []
    dt_anns: list[dict[str, Any]] = []
    for image_id, (p, g) in enumerate(zip(preds, gts, strict=True)):
        h, w = (g or p)[0].shape if (g or p) else (1, 1)
        images.append({"id": image_id, "height": h, "width": w})
        for m in g:
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            gt_anns.append(
                {"id": len(gt_anns) + 1, "image_id": image_id, "category_id": 0,
                 "segmentation": rle, "area": float(mask_utils.area(rle)), "iscrowd": 0}
            )  # fmt: skip
        for k, m in enumerate(p):
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            score = 1.0 if scores is None else float(scores[image_id][k])
            dt_anns.append(
                {"id": len(dt_anns) + 1, "image_id": image_id, "category_id": 0,
                 "segmentation": rle, "area": float(mask_utils.area(rle)), "score": score}
            )  # fmt: skip
    if not gt_anns:
        return 0.0, np.zeros(len(thresholds))
    categories = [{"id": 0, "name": "0"}]
    coco_gt, coco_dt = COCO(), COCO()
    # In-memory datasets (the pycocotools stubs mark ``dataset`` read-only).
    cast("Any", coco_gt).dataset = {
        "images": images,
        "annotations": gt_anns,
        "categories": categories,
    }
    cast("Any", coco_dt).dataset = {
        "images": images,
        "annotations": dt_anns,
        "categories": categories,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt.createIndex()
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.params.iouThrs = np.array(thresholds, dtype=np.float64)
        coco_eval.evaluate()
        coco_eval.accumulate()
    precision = coco_eval.eval["precision"][:, :, 0, 0, -1]

    def summary(values: np.ndarray) -> float:
        valid = values[values > -1]
        return max(float(np.mean(valid)), 0.0) if valid.size else 0.0

    return summary(precision), np.array([summary(row) for row in precision])
