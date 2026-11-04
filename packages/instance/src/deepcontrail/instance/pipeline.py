"""Instance detection from a semantic probability map.

Threshold, split with Mannstein, grow into moderate-probability pixels, then score
(optionally with a learned :class:`deepcontrail.instance.scoring.ScoreModel`).
See :class:`deepcontrail.instance.config.InstancePipelineConfig`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from deepcontrail.instance.config import InstancePipelineConfig
from deepcontrail.instance.mannstein import detect_instances
from deepcontrail.instance.refine import grow_instances, probability_weighted_scores
from deepcontrail.instance.scoring import scene_features

if TYPE_CHECKING:
    from deepcontrail.instance.scoring import ScoreModel

_TOLERANCE = 1e-3


def detect_instances_from_probabilities(
    probs: np.ndarray,
    config: InstancePipelineConfig | None = None,
    score_model: ScoreModel | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Detect contrail instances in a semantic probability map.

    Parameters
    ----------
    probs
        Probability map ``(H, W)`` with values in ``[0, 1]``, e.g. the sigmoid output of a
        semantic segmentation model.
    config
        Pipeline settings. Defaults to ``InstancePipelineConfig()``.
    score_model
        Optional learned scores: replaces each contrail's score by the model's prediction.
        It must have been trained on detections made with the same ``config``.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        Boolean instance masks ``(H, W)`` and their scores in ``[0, 1]``, aligned one-to-one.
        Instances may share pixels where contrails cross.

    Raises
    ------
    ValueError
        If ``probs`` is not 2-D, not finite or outside ``[0, 1]``, or if ``score_model`` was
        trained with another pipeline config.
    """
    if config is None:
        config = InstancePipelineConfig()
    if score_model is not None and score_model.pipeline_config != config:
        raise ValueError("the score model was trained with another pipeline config")
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must be a 2-D (H, W) array, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite (no NaN or inf)")
    if probs.size and (probs.min() < -_TOLERANCE or probs.max() > 1.0 + _TOLERANCE):
        raise ValueError("probabilities must lie in [0, 1]")

    binary = (probs > config.semantic_threshold).astype(np.uint8)
    if not binary.any():
        return [], []

    masks, scores = detect_instances(binary, config.mannstein)
    masks = grow_instances(
        masks,
        probs,
        scores,
        threshold=config.grow_threshold,
        steps=config.grow_steps,
        connectivity=config.grow_connectivity,
    )
    if config.probability_scores:
        scores = probability_weighted_scores(masks, probs, scores)
    if score_model is not None:
        scores = [float(v) for v in score_model.predict(scene_features(masks, probs, scores))]
    return masks, scores
