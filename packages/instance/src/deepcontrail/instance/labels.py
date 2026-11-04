"""Labels derived from instance masks: each annotator's mask, agreement, consensus contrails.

The instance task needs only ``instance_masks``: each annotator's mask, the agreement map and
the consensus mask are derived from it (``*_from_instances``), for any number of annotators,
one included. On OpenContrails they equal the stored ``annotator_masks``, ``probability_masks``
and ``semantic_masks`` exactly. The consensus contrails (:func:`build_consensus_instances`) are
the instance-level counterpart of the consensus mask.
"""

from __future__ import annotations

import numpy as np

from deepcontrail.core.readers import extract_all_annotator_instances


def partition_gt_instances_by_semantic(
    all_gt: list[list[list[np.ndarray]]],
    semantic_masks: list[np.ndarray],
    min_overlap_fraction: float = 0.5,
    tolerance_px: int = 1,
) -> tuple[list[list[list[np.ndarray]]], list[list[list[np.ndarray]]]]:
    """Split ground-truth instances into those lying mostly on the semantic mask and the rest.

    The first step of :func:`build_consensus_instances`. Always pass the ground-truth
    semantic mask, never a prediction.

    Parameters
    ----------
    all_gt
        ``all_gt[scene][slot]`` is the list of boolean ``(H, W)`` instance masks of one
        annotator slot. Slots are split independently.
    semantic_masks
        One boolean ``(H, W)`` ground-truth semantic mask per scene, in the order of ``all_gt``.
    min_overlap_fraction
        An instance is kept when at least this fraction of its pixels lies inside the mask
        (within ``tolerance_px``); kept instances stay whole. With 0.5, a contrail that only
        crosses or touches a consensus contrail is set aside.
    tolerance_px
        Pixels within this many 4-connected steps of the mask count as inside, so the same
        contrail drawn 1 px off by an annotator is kept. 0 means strictly inside.

    Returns
    -------
    tuple[list[list[list[np.ndarray]]], list[list[list[np.ndarray]]]]
        ``(kept, dropped)``, both with the scene and slot structure of ``all_gt``.

    Raises
    ------
    ValueError
        If the number of scenes differs between the two inputs, or
        ``min_overlap_fraction`` is not in ``(0, 1]``, or ``tolerance_px`` is negative.
    """
    if len(all_gt) != len(semantic_masks):
        raise ValueError(
            f"need one semantic mask per scene, got {len(semantic_masks)} for {len(all_gt)} scenes"
        )
    if not 0.0 < min_overlap_fraction <= 1.0:
        raise ValueError(f"min_overlap_fraction must be in (0, 1], got {min_overlap_fraction}")
    if tolerance_px < 0:
        raise ValueError(f"tolerance_px must be >= 0, got {tolerance_px}")
    kept: list[list[list[np.ndarray]]] = []
    dropped: list[list[list[np.ndarray]]] = []
    for scene, semantic in zip(all_gt, semantic_masks, strict=True):
        inside = _grow_4connected(np.asarray(semantic, dtype=bool), tolerance_px)
        keep_scene: list[list[np.ndarray]] = []
        drop_scene: list[list[np.ndarray]] = []
        for slot in scene:
            is_kept = [
                bool(m.any())
                and np.count_nonzero(m & inside) >= min_overlap_fraction * np.count_nonzero(m)
                for m in slot
            ]
            keep_scene.append([m for m, k in zip(slot, is_kept, strict=True) if k])
            drop_scene.append([m for m, k in zip(slot, is_kept, strict=True) if not k])
        kept.append(keep_scene)
        dropped.append(drop_scene)
    return kept, dropped


def _grow_4connected(mask: np.ndarray, steps: int) -> np.ndarray:
    """Dilate a boolean mask by ``steps`` 4-connected steps (numpy only)."""
    grown = mask.copy()
    for _ in range(steps):
        step = grown.copy()
        step[1:, :] |= grown[:-1, :]
        step[:-1, :] |= grown[1:, :]
        step[:, 1:] |= grown[:, :-1]
        step[:, :-1] |= grown[:, 1:]
        grown = step
    return grown


def build_consensus_instances(
    gt_per_slot: list[list[np.ndarray]],
    semantic_mask: np.ndarray,
    min_overlap_fraction: float = 0.5,
    tolerance_px: int = 1,
    link_iou: float = 0.5,
) -> list[np.ndarray]:
    """Build one set of consensus contrail instances for a scene from all annotators.

    The instance-level counterpart of the consensus semantic mask:

    1. keep each annotator's instances with at least ``min_overlap_fraction`` of their
       pixels within ``tolerance_px`` of the consensus mask, and clip them to the mask;
    2. link instances of different annotators whose clipped masks reach ``link_iou``;
    3. keep the linked groups drawn by more than half of the scene's annotators;
    4. each kept instance is the union of its group's clipped masks.

    Instances may share pixels where contrails cross. The result no longer depends on
    which annotator drew what, only on what a majority agreed on.

    Parameters
    ----------
    gt_per_slot
        ``gt_per_slot[slot]`` is one annotator's list of boolean ``(H, W)`` instance masks.
    semantic_mask
        Boolean ``(H, W)`` consensus semantic mask of the scene.
    min_overlap_fraction
        See :func:`partition_gt_instances_by_semantic`.
    tolerance_px
        See :func:`partition_gt_instances_by_semantic`.
    link_iou
        IoU between two clipped instances of different annotators to treat them as the
        same contrail.

    Returns
    -------
    list[np.ndarray]
        Boolean ``(H, W)`` consensus instance masks (empty when the scene has no consensus
        contrail or no annotator).
    """
    n_slots = len(gt_per_slot)
    semantic = np.asarray(semantic_mask, dtype=bool)
    if n_slots == 0 or not semantic.any():
        return []
    kept, _ = partition_gt_instances_by_semantic(
        [gt_per_slot], [semantic], min_overlap_fraction, tolerance_px
    )
    nodes = [(slot, m & semantic) for slot, masks in enumerate(kept[0]) for m in masks]
    nodes = [(slot, clipped) for slot, clipped in nodes if clipped.any()]

    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, (slot_i, mask_i) in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            slot_j, mask_j = nodes[j]
            if slot_i == slot_j:
                continue
            union = np.count_nonzero(mask_i | mask_j)
            if union and np.count_nonzero(mask_i & mask_j) / union >= link_iou:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)

    instances: list[np.ndarray] = []
    for members in groups.values():
        if len({nodes[i][0] for i in members}) * 2 > n_slots:
            instances.append(np.logical_or.reduce([nodes[i][1] for i in members]))
    return instances


def annotator_masks_from_instances(raw: np.ndarray) -> np.ndarray:
    """Return each annotator's mask: the union of the contrails they drew.

    Parameters
    ----------
    raw
        Instance mask array ``(N, H, W, A)`` (1 = pixel of instance n for annotator a).

    Returns
    -------
    np.ndarray
        ``(H, W, A)`` uint8 in ``{0, 1}``; all zeros for a scene without contrails.
    """
    if raw.shape[0] == 0:
        return np.zeros(raw.shape[1:], dtype=np.uint8)
    return np.any(raw == 1, axis=0).astype(np.uint8)


def agreement_from_instances(raw: np.ndarray) -> np.ndarray:
    """Return the share of annotators who marked each pixel, ``(H, W)`` float32 in ``[0, 1]``."""
    return annotator_masks_from_instances(raw).mean(axis=-1, dtype=np.float32)


def consensus_mask_from_instances(raw: np.ndarray) -> np.ndarray:
    """Return the pixels marked by more than half of the annotators, ``(H, W)`` bool."""
    return agreement_from_instances(raw) > 0.5


def consensus_instances(raw: np.ndarray) -> list[np.ndarray]:
    """Return the scene's consensus contrails, from its instance masks alone.

    See :func:`build_consensus_instances`; the consensus mask is
    :func:`consensus_mask_from_instances`. With a single annotator, every contrail they drew
    is kept.
    """
    if raw.shape[0] == 0:
        return []
    return build_consensus_instances(
        extract_all_annotator_instances(raw), consensus_mask_from_instances(raw)
    )
