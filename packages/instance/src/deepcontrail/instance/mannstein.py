"""Top-level Mannstein instance-segmentation pipeline.

Public entry point that threads the private line-filter, segmentation, scoring and
deduplication modules together into ``detect_instances`` (binary mask in; masks and scores
out). Probability maps go through ``detect_instances_from_probabilities``.
"""

from __future__ import annotations

import numpy as np

from deepcontrail.instance._growing import grow_regions
from deepcontrail.instance._lines import detect_line_pixels
from deepcontrail.instance._overlap import remove_duplicates
from deepcontrail.instance._shape import candidate_score
from deepcontrail.instance.config import MannsteinConfig


def _run_pipeline(
    semantic_mask: np.ndarray,
    config: MannsteinConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float]]:
    """Optionally line-filter, then segment, score and deduplicate into ``(contrails, scores)``."""
    if config.line_filter:
        mask = detect_line_pixels(
            semantic_mask.astype(np.float32),
            config.num_kernels,
            config.kernel_size,
            config.sigma1,
            config.sigma_delta,
            config.response_threshold,
        )
    else:
        mask = semantic_mask > 0

    candidates = grow_regions(
        mask,
        config.min_pixels,
        config.min_length,
        config.min_r_score,
        config.lookback,
        config.alignment_threshold,
        config.fit_avg_size if config.filter_avg_size is None else config.filter_avg_size,
    )
    survivors = remove_duplicates(candidates, config.iou_threshold)
    return survivors, [
        candidate_score(rows, cols, config.score_saturation, config.fit_avg_size)
        for rows, cols in survivors
    ]


def _masks_from_contrails(
    contrails: list[tuple[np.ndarray, np.ndarray]],
    h: int,
    w: int,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for rows, cols in contrails:
        mask = np.zeros((h, w), dtype=bool)
        mask[rows, cols] = True
        masks.append(mask)
    return masks


def _check_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` as an array after checking it is a 2-D, finite, binary mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be a 2-D (H, W) array, got shape {mask.shape}")
    if mask.dtype != bool and not np.all(np.isfinite(mask)):
        raise ValueError("mask must be finite (no NaN or inf)")
    if mask.dtype != bool and not np.isin(mask, (0, 1)).all():
        raise ValueError(
            "mask must be binary (0/1 or bool); for a probability map use "
            "detect_instances_from_probabilities"
        )
    return mask


def detect_instances(
    semantic_mask: np.ndarray, config: MannsteinConfig | None = None
) -> tuple[list[np.ndarray], list[float]]:
    """Split a binary contrail mask into separate contrails with plain Mannstein.

    Parameters
    ----------
    semantic_mask
        Binary 2D array ``(H, W)``: bool, or numbers in ``{0, 1}``.
    config
        Algorithm configuration. Defaults to ``MannsteinConfig()``.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        Boolean instance masks ``(H, W)`` and their quality scores in ``[0, 1]``, aligned
        one-to-one; both empty when no contrail is found. Contrails may share pixels where
        they cross.

    Raises
    ------
    ValueError
        If ``semantic_mask`` is not 2-D, not finite or not binary (a probability map belongs
        to ``detect_instances_from_probabilities``).
    """
    mask = _check_binary_mask(semantic_mask)
    if config is None:
        config = MannsteinConfig()
    h, w = mask.shape
    contrails, scores = _run_pipeline(mask, config)
    return _masks_from_contrails(contrails, h, w), scores
