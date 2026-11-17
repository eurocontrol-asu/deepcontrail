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


def _mask_bbox_area(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Bounding box ``(r0, r1, c0, c1)`` and pixel area of a mask; ``(0,0,0,0,0)`` when empty."""
    rows = np.nonzero(mask)[0]
    if rows.size == 0:
        return 0, 0, 0, 0, 0
    cols = np.nonzero(mask)[1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max()), int(rows.size)


def _sorted_iou_pairs(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float
) -> list[tuple[float, int, int]]:
    """Pred/GT pairs with IoU >= ``iou_threshold``, sorted by IoU descending (stable on ties).

    The pair order before the sort is row-major over ``(i, j)``, exactly as the brute-force
    double loop, so the stable sort keeps the same tie behaviour. Every mask's area and
    bounding box are computed once; for an overlapping pair the intersection is measured only
    on the overlap of the two boxes (all set pixels lie inside the box, so nothing is missed),
    with ``union = area_p + area_g - intersection``. Empty masks (area 0) never match. When
    ``iou_threshold <= 0`` the fast path would drop IoU-0 pairs that then count, so the full
    mask computation is used instead, matching the historical behaviour exactly.
    """
    pairs: list[tuple[float, int, int]] = []
    if iou_threshold <= 0:
        for i, pred in enumerate(pred_masks):
            for j, gt in enumerate(gt_masks):
                intersection = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                iou = float(intersection / union) if union else 0.0
                if iou >= iou_threshold:
                    pairs.append((iou, i, j))
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        return pairs
    pred_stats = [_mask_bbox_area(m) for m in pred_masks]
    gt_stats = [_mask_bbox_area(m) for m in gt_masks]
    for i, (pr0, pr1, pc0, pc1, parea) in enumerate(pred_stats):
        if parea == 0:
            continue
        for j, (gr0, gr1, gc0, gc1, garea) in enumerate(gt_stats):
            if garea == 0:
                continue
            r0 = pr0 if pr0 > gr0 else gr0
            r1 = pr1 if pr1 < gr1 else gr1
            c0 = pc0 if pc0 > gc0 else gc0
            c1 = pc1 if pc1 < gc1 else gc1
            if r0 > r1 or c0 > c1:
                continue
            intersection = int(
                np.logical_and(
                    pred_masks[i][r0 : r1 + 1, c0 : c1 + 1],
                    gt_masks[j][r0 : r1 + 1, c0 : c1 + 1],
                ).sum()
            )
            union = parea + garea - intersection
            iou = intersection / union if union else 0.0
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return pairs


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
    tp = 0
    for _iou, i, j in _sorted_iou_pairs(pred_masks, gt_masks, iou_threshold):
        if not matched_pred[i] and not matched_gt[j]:
            matched_pred[i] = True
            matched_gt[j] = True
            tp += 1

    return tp, n_pred - tp, n_gt - tp


def _split_ignore(
    masks: list[np.ndarray], ignore: list[bool] | None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split a mask list into (kept, ignored) by an optional per-mask ignore flag."""
    kept: list[np.ndarray] = []
    ignored: list[np.ndarray] = []
    for k, m in enumerate(masks):
        if ignore is not None and k < len(ignore) and ignore[k]:
            ignored.append(m)
        else:
            kept.append(m)
    return kept, ignored


def _preds_matched_to(
    pred_masks: list[np.ndarray], gt_masks: list[np.ndarray], iou_threshold: float
) -> set[int]:
    """Return the indices of preds greedily matched one-to-one to a GT list at IoU >= threshold.

    Same greedy order as :func:`match_instances` (highest IoU first, stable on ties), used to
    drop the predictions that hit an ignored ground-truth instance before the real matching.
    """
    remove: set[int] = set()
    matched_gt = [False] * len(gt_masks)
    for _iou, i, j in _sorted_iou_pairs(pred_masks, gt_masks, iou_threshold):
        if i in remove or matched_gt[j]:
            continue
        remove.add(i)
        matched_gt[j] = True
    return remove


def evaluate_annotator_samples(
    pred_masks: list[list[np.ndarray]],
    gt_masks: list[list[np.ndarray]],
    pred_scores: list[list[float]] | None = None,
    iou_thresholds: tuple[float, ...] | list[float] = DEFAULT_IOU_THRESHOLDS,
    match_iou: float = 0.5,
    per_threshold: bool = True,
    include_empty: bool = False,
    max_detections: int = COCO_MAX_DETECTIONS,
    gt_ignore: list[list[bool]] | None = None,
) -> dict[str, Any]:
    """Score instance predictions per annotator sample (pooled COCO segmentation mAP).

    One sample is one annotator on one scene. Samples whose ground truth is empty are
    skipped (the official protocol) unless ``include_empty``. All remaining samples are
    pooled into a single COCO segmentation mAP computation (one image per sample, single
    class), averaged over ``iou_thresholds``. Precision, recall and F1 from one-to-one
    matching at ``match_iou``, pooled over the same samples, are returned as a reference.

    With ``include_empty``, every sample is kept and every detection on an empty sample is a
    false detection, as in real use where most scenes have no contrail.

    The computation is the standard COCO segmentation AP of pycocotools (101 recall points,
    area "all"), keeping at most ``max_detections`` detections per sample (default
    ``COCO_MAX_DETECTIONS`` = 100, the pycocotools default); torch is not needed. Every sample
    is registered with COCO, so ``include_empty`` counts the detections on empty samples
    (torchmetrics' ``MeanAveragePrecision`` drops them).

    On large, crowded frames (MTG is 1024 x 1024 with up to ~200 contrails) raise
    ``max_detections`` above 100, or the cap truncates recall. ``gt_ignore`` marks ground-truth
    instances an annotator did not want scored: an ignored instance does not count as ground
    truth (a sample whose only GT is ignored is treated as empty), and a detection on one is
    neither a hit nor a false positive — it is dropped from the reference counts and registered
    as a pycocotools "crowd" region in the mAP, which does not penalise detections inside it.

    Parameters
    ----------
    pred_masks
        ``pred_masks[i]`` is the list of predicted boolean ``(H, W)`` masks for sample ``i``.
    gt_masks
        ``gt_masks[i]`` is the list of GT boolean masks of sample ``i`` (one annotator).
    pred_scores
        Confidence per predicted mask, aligned with ``pred_masks``; ``None`` = 1.0 for all.
        With equal scores the ranking hides false detections, so the mAP then mostly
        measures recall: pass the pipeline's scores.
    iou_thresholds
        IoU thresholds averaged by the mAP. Defaults to 0.25, 0.35, ..., 0.75.
    match_iou
        IoU threshold of the reference one-to-one matching.
    per_threshold
        Also return the AP at each threshold.
    include_empty
        Keep samples with an empty ground truth and count their detections as false.
    max_detections
        Maximum detections per sample kept by the COCO mAP (pycocotools default 100). Raise it
        on crowded frames; must be at least 1.
    gt_ignore
        Optional per-GT ignore flags, nested like ``gt_masks`` (``gt_ignore[i][j]`` ignores
        ``gt_masks[i][j]``). Ignored instances are excluded from the reference counts and from
        the "has ground truth" test, and registered as crowd regions in the mAP.

    Returns
    -------
    dict[str, Any]
        ``map``, ``ap_per_threshold`` (when requested), ``precision``, ``recall``, ``f1``,
        ``tp``, ``fp``, ``fn``, ``n_samples`` and ``n_skipped_empty``.

    Raises
    ------
    ValueError
        If ``pred_masks``, ``gt_masks`` (and ``pred_scores``) have different lengths, if
        ``max_detections`` < 1, or if ``gt_ignore`` does not match ``gt_masks`` length.
    """
    if len(pred_masks) != len(gt_masks) or (
        pred_scores is not None and len(pred_scores) != len(pred_masks)
    ):
        raise ValueError("pred_masks, gt_masks and pred_scores must have one entry per sample")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")
    if gt_ignore is not None and len(gt_ignore) != len(gt_masks):
        raise ValueError("gt_ignore must have one per-sample entry, like gt_masks")

    def effective_gt(i: int) -> list[np.ndarray]:
        # GT of sample i minus its ignored instances; empty when every GT is ignored.
        kept, _ = _split_ignore(gt_masks[i], gt_ignore[i] if gt_ignore is not None else None)
        return kept

    keep = [i for i in range(len(gt_masks)) if effective_gt(i) or include_empty]
    preds = [pred_masks[i] for i in keep]
    gts = [gt_masks[i] for i in keep]
    gt_ignores = [gt_ignore[i] for i in keep] if gt_ignore is not None else None
    scores = None if pred_scores is None else [pred_scores[i] for i in keep]

    tp = fp = fn = 0
    for idx, (p, g) in enumerate(zip(preds, gts, strict=True)):
        ignore = gt_ignores[idx] if gt_ignores is not None else None
        kept_g, ignored = _split_ignore(g, ignore)
        # a prediction on an ignored contrail is neither a hit nor a false positive: drop it
        # (greedy, before the real matching) and drop the ignored GT from the reference counts.
        dropped = _preds_matched_to(p, ignored, match_iou)
        kept_p = [m for k, m in enumerate(p) if k not in dropped]
        a, b, c = match_instances(kept_p, kept_g, match_iou)
        tp, fp, fn = tp + a, fp + b, fn + c
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    crowded = sum(len(p) > max_detections for p in preds)
    if crowded:
        warnings.warn(
            f"{crowded} sample(s) have more than {max_detections} detections; the COCO mAP "
            f"keeps only the {max_detections} highest-scoring detections per sample",
            UserWarning,
            stacklevel=2,
        )
    pooled, ap = _coco_ap_all_samples(
        preds, gts, scores, list(iou_thresholds), max_detections, gt_ignores
    )
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
    max_detections: int = COCO_MAX_DETECTIONS,
    gt_ignore: list[list[bool]] | None = None,
) -> tuple[float, np.ndarray]:
    """Pooled mAP and AP per threshold, every sample registered in COCO (empty GT included).

    Standard COCO segmentation AP (pycocotools: 101 recall points, area "all"), keeping at most
    ``max_detections`` detections per sample, with ``thresholds`` as IoU thresholds. Zeros when
    there is nothing to score. Ignored GTs (``gt_ignore[i][k]`` for sample ``i``) are registered
    as ``iscrowd`` regions: pycocotools computes their IoU as the intersection over the
    detection's area and does not penalise detections that fall inside them.
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
        ignore = gt_ignore[image_id] if gt_ignore is not None else None
        for k, m in enumerate(g):
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            is_crowd = bool(ignore[k]) if ignore is not None and k < len(ignore) else False
            gt_anns.append(
                {"id": len(gt_anns) + 1, "image_id": image_id, "category_id": 0,
                 "segmentation": rle, "area": float(mask_utils.area(rle)), "iscrowd": int(is_crowd)}
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
        # The summary reads the last maxDets index, so the cap is max_detections.
        coco_eval.params.maxDets = [1, 10, max_detections]
        coco_eval.evaluate()
        coco_eval.accumulate()
    precision = coco_eval.eval["precision"][:, :, 0, 0, -1]

    def summary(values: np.ndarray) -> float:
        valid = values[values > -1]
        return max(float(np.mean(valid)), 0.0) if valid.size else 0.0

    return summary(precision), np.array([summary(row) for row in precision])
