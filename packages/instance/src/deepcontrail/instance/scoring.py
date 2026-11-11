"""Learned contrail scores: re-rank detected contrails with a model trained on your data.

The pipeline's scores (Mannstein's straightness and size, times the mean probability) rank
detections only roughly. A :class:`ScoreModel` predicts, from a few features of each detected
contrail (:data:`FEATURES`), how well it matches the annotated contrails, and that prediction
replaces its score. The masks do not change, so only the ranking (and the mAP) changes.

The features depend on the semantic model's probabilities, so a score model belongs to one
semantic model and one pipeline config: train it on that model's predictions for scenes it was
not trained on (e.g. its validation split), with :func:`training_data` and
:meth:`ScoreModel.fit`. Fitting uses scikit-learn's gradient boosting; the fitted trees are
exported to plain JSON, so using a model needs numpy only and loading a file runs no code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from scipy.ndimage import binary_dilation

from deepcontrail.instance.config import InstancePipelineConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

FEATURES: tuple[str, ...] = (
    "n_pixels",
    "length",
    "width",
    "straightness",
    "mean_prob",
    "max_prob",
    "min_prob",
    "std_prob",
    "p10_prob",
    "p90_prob",
    "frac_above_0_7",
    "ring_mean_prob",
    "contrast",
    "pipeline_score",
    "n_scene_instances",
    "shared_pixels_frac",
    "scene_pred_area",
    "border_distance",
    "orientation_sin2",
    "orientation_cos2",
)
"""Names of the features of one detected contrail, in the order of :func:`instance_features`."""


def instance_features(
    mask: np.ndarray, probs: np.ndarray, score: float, scene_masks: Sequence[np.ndarray]
) -> np.ndarray:
    """Describe one detected contrail (size, shape, probabilities, surroundings).

    Parameters
    ----------
    mask
        Boolean ``(H, W)`` mask of the contrail.
    probs
        The scene's probability map ``(H, W)``.
    score
        The contrail's score from the pipeline.
    scene_masks
        All contrails detected in the scene, ``mask`` included.

    Returns
    -------
    np.ndarray
        One value per name in :data:`FEATURES`.
    """
    counts = _coverage(scene_masks)
    return _features(mask, probs, score, len(scene_masks), counts, _multiplicity(mask, scene_masks))


def _coverage(scene_masks: Sequence[np.ndarray]) -> np.ndarray | None:
    """Per pixel, how many of the scene's masks cover it (``None`` for an empty scene)."""
    if not scene_masks:
        return None
    stacked = np.stack([np.asarray(m, dtype=bool) for m in scene_masks])
    counts: np.ndarray = stacked.sum(axis=0, dtype=np.int64)
    return counts


def _multiplicity(mask: np.ndarray, scene_masks: Sequence[np.ndarray]) -> int:
    """How many times this very object is in ``scene_masks``: it never overlaps itself."""
    return sum(1 for m in scene_masks if m is mask)


def _features(
    mask: np.ndarray,
    probs: np.ndarray,
    score: float,
    n_masks: int,
    counts: np.ndarray | None,
    multiplicity: int,
) -> np.ndarray:
    """Compute :func:`instance_features` from the scene's per-pixel coverage, shared by a scene.

    ``counts`` (see ``_coverage``) replaces stacking every other mask for every contrail, which
    made a scene's features quadratic in its number of contrails.
    """
    # A 0/1 int mask indexes ``probs`` by position and ``~mask`` is a bitwise not, so compute on a
    # bool copy. The overlap discounts the mask by identity (``multiplicity``), not by value.
    bool_mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(bool_mask)
    n = len(ys)
    values = probs[bool_mask]
    coords = np.stack([ys, xs], axis=1).astype(float)
    centre = coords.mean(axis=0)
    if n >= 2:
        eigvals, eigvecs = np.linalg.eigh(np.cov((coords - centre).T))
        along = (coords - centre) @ eigvecs[:, -1]
        length = float(along.max() - along.min() + 1.0)
        straightness = float(1.0 - eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else 0.0
        angle = float(np.arctan2(eigvecs[0, -1], eigvecs[1, -1]))
    else:
        length, straightness, angle = float(n), 0.0, 0.0
    ring = binary_dilation(bool_mask, np.ones((3, 3), dtype=bool)) & ~bool_mask
    ring_mean = float(probs[ring].mean()) if ring.any() else 0.0
    if counts is None or n_masks == multiplicity:
        shared = 0.0  # no other contrail in the scene
    else:
        shared = float(((counts[bool_mask] - multiplicity) > 0).sum() / n)
    h, w = bool_mask.shape
    border = min(centre[0], centre[1], h - 1 - centre[0], w - 1 - centre[1]) / max(h, w)
    union_area = float((counts > 0).sum()) if counts is not None else 0.0
    return np.array(
        [
            n,
            length,
            n / length if length else 0.0,
            straightness,
            values.mean(),
            values.max(),
            values.min(),
            values.std(),
            np.percentile(values, 10),
            np.percentile(values, 90),
            np.mean(values > 0.7),
            ring_mean,
            values.mean() - ring_mean,
            score,
            n_masks,
            shared,
            union_area,
            border,
            np.sin(2 * angle),
            np.cos(2 * angle),
        ],
        dtype=float,
    )


def scene_features(
    masks: Sequence[np.ndarray], probs: np.ndarray, scores: Sequence[float]
) -> np.ndarray:
    """Return the ``(n_contrails, len(FEATURES))`` features of one scene's detections."""
    if not masks:
        return np.zeros((0, len(FEATURES)))
    counts = _coverage(masks)
    return np.stack(
        [
            _features(m, probs, s, len(masks), counts, _multiplicity(m, masks))
            for m, s in zip(masks, scores, strict=True)
        ]
    )


def _best_iou(mask: np.ndarray, gts: Sequence[np.ndarray]) -> float:
    best = 0.0
    for g in gts:
        union = np.logical_or(mask, g).sum()
        if union:
            best = max(best, float(np.logical_and(mask, g).sum() / union))
    return best


def match_quality(mask: np.ndarray, gts_per_annotator: Sequence[Sequence[np.ndarray]]) -> float:
    """Training label: mean best IoU of ``mask`` over the annotators who drew a contrail.

    Parameters
    ----------
    mask
        Boolean ``(H, W)`` mask of one detected contrail.
    gts_per_annotator
        For each annotator of the scene, the boolean masks of the contrails they drew.

    Returns
    -------
    float
        In ``[0, 1]``; 0.0 if no annotator drew anything.
    """
    drawn = [g for g in gts_per_annotator if g]
    return float(np.mean([_best_iou(mask, g) for g in drawn])) if drawn else 0.0


def training_data(
    probs: Sequence[np.ndarray],
    gts: Sequence[Sequence[Sequence[np.ndarray]]],
    config: InstancePipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the pipeline on each scene and return features and labels of every detection.

    Parameters
    ----------
    probs
        Probability maps of the training scenes.
    gts
        ``gts[i][a]`` holds the contrail masks drawn by annotator ``a`` on scene ``i``.
    config
        The pipeline config the score model will be used with.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features ``(n, len(FEATURES))`` and labels ``(n,)`` (:func:`match_quality`).
    """
    from deepcontrail.instance.pipeline import detect_instances_from_probabilities

    rows, labels = [], []
    for scene_probs, scene_gts in zip(probs, gts, strict=True):
        p = np.asarray(scene_probs, dtype=np.float32)
        masks, scores = detect_instances_from_probabilities(p, config)
        rows.append(scene_features(masks, p, scores))
        labels.extend(match_quality(m, scene_gts) for m in masks)
    x = np.concatenate(rows) if rows else np.zeros((0, len(FEATURES)))
    return x, np.asarray(labels, dtype=float)


_NODE_FIELDS = ("feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")

# The keys to_json writes per tree: _NODE_FIELDS plus the leaf values.
_TREE_KEYS = (*_NODE_FIELDS, "value")


def _check_tree(index: int, tree: dict[str, np.ndarray]) -> None:
    """Reject a malformed tree before :meth:`ScoreModel.predict` walks it.

    ``predict`` follows ``left``/``right`` and indexes ``feature_idx`` without checking bounds or
    cycles, so a corrupt file would loop forever (a child pointing at its parent), read the wrong
    feature (a negative index), or fail mid-prediction (ragged or non-finite arrays). The
    ``left``/``right``/``feature_idx``/``num_threshold`` of leaves are unchecked scikit-learn
    placeholders.
    """
    expected_keys = set(_TREE_KEYS)
    if set(tree) != expected_keys:
        raise ValueError(
            f"score model tree {index}: keys {sorted(set(tree))} are not {sorted(expected_keys)}"
        )
    n = len(tree["value"])
    lengths = [len(tree[key]) for key in expected_keys]
    if n < 1 or any(length != n for length in lengths):
        raise ValueError(
            f"score model tree {index}: node arrays have lengths {lengths}, expected one >= 1"
        )
    is_leaf = tree["is_leaf"].astype(bool)
    for j in np.nonzero(~is_leaf)[0]:
        left, right = int(tree["left"][j]), int(tree["right"][j])
        if not (j < left < n and j < right < n):
            raise ValueError(
                f"score model tree {index}: node {j} children {left}, {right} must come after it "
                f"and within its {n} nodes"
            )
        if not (0 <= int(tree["feature_idx"][j]) < len(FEATURES)):
            raise ValueError(
                f"score model tree {index}: node {j} feature index {int(tree['feature_idx'][j])} "
                f"is out of range for {len(FEATURES)} features"
            )
        if not np.isfinite(tree["num_threshold"][j]):
            raise ValueError(
                f"score model tree {index}: node {j} threshold {float(tree['num_threshold'][j])} "
                f"is not finite"
            )
    if not np.all(np.isfinite(tree["value"])):
        raise ValueError(f"score model tree {index}: leaf values are not all finite")


@dataclass(frozen=True)
class ScoreModel:
    """Gradient-boosted trees that predict a detected contrail's match quality.

    Build one with :meth:`fit` (needs scikit-learn) or :meth:`from_json` (numpy only).

    Attributes
    ----------
    baseline
        Constant start value of the prediction.
    trees
        One dict of node arrays per tree: ``feature_idx``, ``num_threshold``,
        ``missing_go_to_left``, ``left``, ``right``, ``is_leaf`` and ``value``.
    pipeline_config
        The pipeline config whose detections the model was trained on.
    features
        Feature names, checked on load against :data:`FEATURES`.
    trained_on
        What the model is valid for, since its features depend on the probability maps:
        ``semantic_model`` (and ``semantic_revision``), ``tta``, ``scene_shape`` ``[H, W]``,
        plus the ``scikit_learn_version`` and ``deepcontrail_instance_version`` that fitted it.
        ``scene_shape`` is checked by the pipeline, ``tta`` by :func:`load_instance_step`.
        Empty for files written before this record existed.
    """

    baseline: float
    trees: tuple[dict[str, np.ndarray], ...]
    pipeline_config: InstancePipelineConfig
    features: tuple[str, ...] = FEATURES
    trained_on: dict[str, Any] = field(default_factory=dict)

    FORMAT_VERSION: ClassVar[int] = 1
    """Version of the JSON layout; bump it when the features or the tree format change."""

    ESTIMATOR_SETTINGS: ClassVar[dict[str, Any]] = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "random_state": 0,
    }

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        pipeline_config: InstancePipelineConfig,
        trained_on: dict[str, Any] | None = None,
    ) -> ScoreModel:
        """Fit the trees on features ``x`` and labels ``y`` (fixed, seeded settings).

        ``trained_on`` describes the probability maps behind ``x`` (see the class attributes);
        the scikit-learn and package versions are added to it.
        """
        try:
            # The [scores] extra; resolvable only where it is installed.
            import sklearn.ensemble  # pyright: ignore[reportMissingImports]
        except ImportError as err:
            raise ImportError(
                "ScoreModel.fit needs scikit-learn: pip install 'deepcontrail-instance[scores]'"
            ) from err

        estimator = sklearn.ensemble.HistGradientBoostingRegressor(**cls.ESTIMATOR_SETTINGS).fit(
            x, y
        )
        trees = []
        for (predictor,) in estimator._predictors:
            nodes = predictor.nodes
            tree = {name: np.asarray(nodes[name]) for name in _NODE_FIELDS}
            tree["value"] = np.asarray(nodes["value"], dtype=float)
            trees.append(tree)
        baseline = float(np.ravel(estimator._baseline_prediction)[0])
        record = {
            **(trained_on or {}),
            "scikit_learn_version": sklearn.__version__,
            "deepcontrail_instance_version": version("deepcontrail-instance"),
        }
        return cls(baseline, tuple(trees), pipeline_config, trained_on=record)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the match quality of each row of ``x``, clipped to ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``x`` does not have one column per feature.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features per row, got shape {x.shape}")
        out = np.full(len(x), self.baseline)
        for tree in self.trees:
            node = np.zeros(len(x), dtype=int)
            active = ~tree["is_leaf"][node].astype(bool)
            while active.any():
                idx = np.nonzero(active)[0]
                current = node[idx]
                values = x[idx, tree["feature_idx"][current]]
                go_left = np.where(
                    np.isnan(values),
                    tree["missing_go_to_left"][current].astype(bool),
                    values <= tree["num_threshold"][current],
                )
                node[idx] = np.where(go_left, tree["left"][current], tree["right"][current])
                active[idx] = ~tree["is_leaf"][node[idx]].astype(bool)
            out += tree["value"][node]
        return np.clip(out, 0.0, 1.0)

    def to_json(self, path: str | Path) -> None:
        """Write the model as plain JSON."""
        data = {
            "format_version": self.FORMAT_VERSION,
            "features": list(self.features),
            "baseline": self.baseline,
            "pipeline_config": self.pipeline_config.model_dump(),
            "trained_on": self.trained_on,
            "trees": [{k: v.tolist() for k, v in tree.items()} for tree in self.trees],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def from_json(cls, path: str | Path) -> ScoreModel:
        """Load a model written by :meth:`to_json`.

        Raises
        ------
        ValueError
            If the file has another format version or other feature names.
        """
        data = json.loads(Path(path).read_text())
        if data.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"score model format version {data.get('format_version')!r} is not supported "
                f"(expected {cls.FORMAT_VERSION}); retrain it with this library version"
            )
        if tuple(data["features"]) != FEATURES:
            raise ValueError("score model features differ from this library's FEATURES")
        trees = tuple({k: np.asarray(v) for k, v in tree.items()} for tree in data["trees"])
        for i, tree in enumerate(trees):
            _check_tree(i, tree)
        config = InstancePipelineConfig.from_dict(data["pipeline_config"])
        return cls(
            float(data["baseline"]),
            trees,
            config,
            tuple(data["features"]),
            trained_on=dict(data.get("trained_on", {})),
        )
