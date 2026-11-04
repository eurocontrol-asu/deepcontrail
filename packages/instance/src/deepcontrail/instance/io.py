"""Instance masks from your own annotations, and detected contrails as COCO results.

The instance tools read only ``instance_masks/<split>/<id>.npy``: an ``(N, H, W, A)`` int8 array
where 1 marks a pixel of contrail n drawn by annotator a, 0 a pixel it does not cover, and -1
that annotator a drew no contrail n (``A = 1`` for a single annotator). The two builders below
write that array from a label image or from per-annotator masks. COCO polygons can be turned
into masks first with ``pycocotools`` (``COCO.annToMask``).

``to_coco_results`` saves detections as COCO result entries with RLE masks: overlapping
contrails stay separate, and the format is read by standard COCO tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def instance_masks_from_label_image(labels: np.ndarray) -> np.ndarray:
    """Build single-annotator instance masks from a label image.

    Parameters
    ----------
    labels
        Integer ``(H, W)`` image: 0 is background, every other value one contrail (values need
        not be consecutive). Contrails cannot overlap in a label image.

    Returns
    -------
    np.ndarray
        ``(N, H, W, 1)`` int8, contrails in increasing label order.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2-D (H, W) image, got shape {labels.shape}")
    ids = [v for v in np.unique(labels) if v != 0]
    raw = np.zeros((len(ids), *labels.shape, 1), dtype=np.int8)
    for n, value in enumerate(ids):
        raw[n, :, :, 0] = labels == value
    return raw


def instance_masks_from_annotators(
    contrails: Sequence[Sequence[np.ndarray]], shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Build instance masks from the contrails each annotator drew.

    Parameters
    ----------
    contrails
        ``contrails[a]`` holds the boolean ``(H, W)`` masks of annotator ``a``'s contrails;
        an annotator may have drawn none. Contrail n of one annotator and contrail n of another
        are unrelated.
    shape
        Image size, required only when nobody drew anything.

    Returns
    -------
    np.ndarray
        ``(N, H, W, A)`` int8 with N the most contrails any annotator drew; -1 where annotator
        a drew fewer than n + 1 contrails.

    Raises
    ------
    ValueError
        If masks have different shapes, or no mask and no ``shape`` is given.
    """
    masks = [np.asarray(m, dtype=bool) for drawn in contrails for m in drawn]
    shapes = {m.shape for m in masks} | ({tuple(shape)} if shape is not None else set())
    if len(shapes) != 1:
        raise ValueError(
            f"all masks must have one shape (pass shape= when nobody drew), got {sorted(shapes)}"
        )
    (h, w) = shapes.pop()
    n = max((len(drawn) for drawn in contrails), default=0)
    raw = np.full((n, h, w, len(contrails)), -1, dtype=np.int8)
    for a, drawn in enumerate(contrails):
        for k, mask in enumerate(drawn):
            raw[k, :, :, a] = np.asarray(mask, dtype=bool)
    return raw


def to_coco_results(
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    image_id: int | str,
    category_id: int = 1,
) -> list[dict[str, Any]]:
    """Turn one scene's detections into COCO result entries with RLE masks.

    Parameters
    ----------
    masks
        Boolean ``(H, W)`` contrail masks, e.g. from ``detect_instances_from_probabilities``.
    scores
        One score per mask.
    image_id
        Identifier of the scene, copied into every entry.
    category_id
        COCO category of a contrail.

    Returns
    -------
    list[dict]
        ``{"image_id", "category_id", "segmentation": {"size", "counts"}, "score"}`` per
        contrail, ready for ``json.dump`` (RLE counts as text).

    Raises
    ------
    ValueError
        If ``scores`` does not have one entry per mask.
    """
    try:
        import pycocotools.mask as mask_utils
    except ImportError as err:
        raise ImportError(
            "to_coco_results needs pycocotools: pip install 'deepcontrail-instance[map]'"
        ) from err

    if len(scores) != len(masks):
        raise ValueError(f"need one score per mask, got {len(scores)} for {len(masks)}")
    results = []
    for mask, score in zip(masks, scores, strict=True):
        rle = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
        counts = rle["counts"]
        counts = counts.decode() if isinstance(counts, bytes) else counts
        results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": {"size": list(rle["size"]), "counts": counts},
                "score": float(score),
            }
        )
    return results
