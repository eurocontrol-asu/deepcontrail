"""Building blocks of the instance evaluation and tuning scripts.

Private module (leading underscore): it serves the repository scripts in ``scripts/`` and is
not part of the public API, so it may change between releases.

Annotator samples, scene lists and splits, loading predictions, running the detection in
parallel, and scoring it per annotator sample (pooled COCO mAP over IoU 0.25-0.75, see
``deepcontrail.instance.metrics``). The scripts in ``scripts/`` are thin command-line
wrappers around these functions.

Only ``instance_masks/<split>/<id>.npy`` is read from a dataset. Parallel runs use the default
multiprocessing start method, so they also work where ``fork`` is unavailable (Windows) or
unsafe (macOS).
"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from deepcontrail.core.readers import (
    binarize_mask,
    extract_all_annotator_instances,
    extract_annotator_instances,
    load_instance_masks,
    load_mask,
)
from deepcontrail.instance.labels import annotator_masks_from_instances
from deepcontrail.instance.mannstein import detect_instances
from deepcontrail.instance.pipeline import detect_instances_from_probabilities

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from deepcontrail.instance.config import InstancePipelineConfig, MannsteinConfig
    from deepcontrail.instance.scoring import ScoreModel

Detections = tuple[list[np.ndarray], list[float]]


def default_workers() -> int:
    """Return a safe default number of worker processes: at most 8, at most the CPU count."""
    return max(1, min(8, os.cpu_count() or 1))


@dataclass(frozen=True)
class Sample:
    """One annotator on one scene: their semantic mask and their instances (GT)."""

    stem: str
    slot: int
    annotator_mask: np.ndarray
    gt: list[np.ndarray]


def build_samples(instance_masks: np.ndarray, stem: str = "") -> list[Sample]:
    """Build every annotator sample of one scene, including empty ones.

    Parameters
    ----------
    instance_masks
        ``(N, H, W, A)`` instance masks; 1 = pixel of instance n for annotator a, -1 =
        annotator a did not draw instance n (never part of the GT). ``A`` may be 1.
    stem
        Scene id, kept on each sample for reporting.

    Returns
    -------
    list[Sample]
        One sample per annotator. Its mask is the union of the annotator's contrails. A sample
        with an empty GT is skipped by the scorer.
    """
    masks = annotator_masks_from_instances(instance_masks)
    return [
        Sample(stem, slot, masks[..., slot], extract_annotator_instances(instance_masks, slot))
        for slot in range(instance_masks.shape[-1])
    ]


def list_scenes(root: str | Path, split: str) -> list[str]:
    """Return the sorted scene ids of ``<root>/instance_masks/<split>/``.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    folder = Path(root) / "instance_masks" / split
    if not folder.is_dir():
        raise FileNotFoundError(f"directory not found: {folder}")
    return sorted(p.stem for p in folder.glob("*.npy"))


def split_scenes(
    stems: Sequence[str],
    seed: int,
    n_search: int | None = None,
    n_holdout: int | None = None,
) -> tuple[list[str], list[str]]:
    """Split scene ids into disjoint search and holdout sets, reproducibly.

    The ids are shuffled with ``seed``; the first ``n_search`` form the search set and the next
    ``n_holdout`` the holdout. By default the search set is half of the scenes and the holdout
    is the rest.

    Raises
    ------
    ValueError
        If either set would be empty, or if more scenes are asked for than exist.
    """
    n = len(stems)
    n_search = n // 2 if n_search is None else n_search
    n_holdout = n - n_search if n_holdout is None else n_holdout
    if n_search <= 0 or n_holdout <= 0:
        raise ValueError(
            f"{n} scenes give {n_search} search and {n_holdout} holdout scenes; both must be "
            "at least 1 (the holdout is what the results are judged on)"
        )
    if n_search + n_holdout > n:
        raise ValueError(f"need {n_search} + {n_holdout} scenes, found {n}")
    shuffled = [stems[int(i)] for i in np.random.default_rng(seed).permutation(n)]
    return shuffled[:n_search], shuffled[n_search : n_search + n_holdout]


def select_annotator_samples(
    root: str | Path, n_samples: int, n_holdout: int, seed: int, split: str = "train"
) -> tuple[list[Sample], list[Sample]]:
    """Pick non-empty annotator samples for a search and a holdout, scene by scene.

    Scenes are shuffled with ``seed`` and walked in order; each scene's non-empty annotator
    samples go to the search set until it holds at least ``n_samples``, then to the holdout
    until it holds at least ``n_holdout``. A scene is never split.

    Raises
    ------
    ValueError
        If the split has too few non-empty annotator samples.
    """
    root = Path(root)
    stems = list_scenes(root, split)
    order = np.random.default_rng(seed).permutation(len(stems))
    search: list[Sample] = []
    holdout: list[Sample] = []
    for i in order:
        if len(holdout) >= n_holdout:
            break
        raw = load_instance_masks(root / "instance_masks" / split / f"{stems[i]}.npy")
        scene = [s for s in build_samples(raw, stems[i]) if s.gt]
        (search if len(search) < n_samples else holdout).extend(scene)
    if len(search) < n_samples or len(holdout) < n_holdout:
        raise ValueError(
            f"need {n_samples} + {n_holdout} non-empty annotator samples, found "
            f"{len(search)} + {len(holdout)}"
        )
    return search, holdout


def load_prediction(path: str | Path) -> np.ndarray:
    """Load one scene's semantic prediction as a float32 map in ``[0, 1]``.

    Parameters
    ----------
    path
        A ``.npy`` file with probabilities of shape ``(H, W)``, or a ``.png`` grayscale
        mask where any non-zero pixel is foreground (returned as 0.0/1.0).

    Raises
    ------
    ValueError
        If the suffix is neither ``.npy`` nor ``.png``, if a ``.npy`` array is not 2-D, holds
        non-finite values, or has values outside ``[0, 1]`` (tolerance 1e-3).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".png":
        return binarize_mask(load_mask(path)).astype(np.float32)
    if suffix == ".npy":
        probs = np.load(path)
        if probs.ndim != 2:
            raise ValueError(f"prediction {path} must be a 2-D (H, W) array, got {probs.shape}")
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"prediction {path} has non-finite values (NaN or inf)")
        if np.any(probs < -1e-3) or np.any(probs > 1.0 + 1e-3):
            raise ValueError(f"prediction {path} has values outside [0, 1] (tolerance 1e-3)")
        return probs.astype(np.float32)
    raise ValueError(f"unsupported prediction suffix {suffix!r}; expected .npy or .png")


def prediction_path(pred_dir: str | Path, stem: str) -> Path:
    """Return ``<stem>.npy`` if present, else ``<stem>.png`` in ``pred_dir``."""
    npy = Path(pred_dir) / f"{stem}.npy"
    return npy if npy.exists() else Path(pred_dir) / f"{stem}.png"


def load_predictions(
    pred_dir: str | Path, root: str | Path, stems: Sequence[str], split: str
) -> tuple[list[np.ndarray], list[list[list[np.ndarray]]]]:
    """Load the predictions of ``stems`` and their GT contrails per annotator.

    Raises
    ------
    FileNotFoundError
        If a scene has no prediction file, or no instance masks in ``split``.
    ValueError
        If a prediction is not the size of its masks.
    """
    instance_dir = Path(root) / "instance_masks" / split
    missing = [s for s in stems if not prediction_path(pred_dir, s).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(stems)} scenes have no prediction in {pred_dir}: "
            + ", ".join(missing[:5])
        )
    no_gt = [s for s in stems if not (instance_dir / f"{s}.npy").exists()]
    if no_gt:
        raise FileNotFoundError(
            f"{len(no_gt)} scenes have no instance masks in {instance_dir}: " + ", ".join(no_gt[:5])
        )
    probs, gts = [], []
    for stem in stems:
        prob = load_prediction(prediction_path(pred_dir, stem))
        raw = load_instance_masks(instance_dir / f"{stem}.npy")
        check_size(prob, raw.shape[1:3], stem)
        probs.append(prob)
        gts.append(extract_all_annotator_instances(raw))
    return probs, gts


def check_size(prediction: np.ndarray, mask_shape: tuple[int, ...], stem: str) -> None:
    """Raise if a prediction is not on the pixel grid of its masks.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    if tuple(prediction.shape) != tuple(mask_shape):
        h, w = prediction.shape
        mh, mw = mask_shape
        raise ValueError(
            f"prediction of {stem} is {h}x{w} but its masks are {mh}x{mw}; predictions must be "
            "on the pixel grid of the masks"
        )


_WORKER: dict[str, Any] = {}


def _init_worker(state: dict[str, Any]) -> None:
    _WORKER.clear()
    _WORKER.update(state)


def _detect_probabilities(probs: np.ndarray) -> Detections:
    return detect_instances_from_probabilities(
        probs, _WORKER["config"], score_model=_WORKER.get("score_model")
    )


def _detect_mask(mask: np.ndarray) -> Detections:
    return detect_instances(mask, _WORKER["config"])


def _run_parallel(
    fn: Callable[[np.ndarray], Detections],
    items: Sequence[np.ndarray],
    state: dict[str, Any],
    workers: int,
) -> list[Detections]:
    """Apply ``fn`` to every item, in a process pool when ``workers > 1``.

    ``state`` (small: configs) reaches each worker once through the pool initializer; the
    items are sent with the tasks.
    """
    if workers <= 1 or len(items) <= 1:
        _init_worker(state)
        return [fn(item) for item in items]
    with multiprocessing.Pool(workers, initializer=_init_worker, initargs=(state,)) as pool:
        return pool.map(fn, items, chunksize=4)


def detect_all(
    probs: Sequence[np.ndarray],
    config: InstancePipelineConfig,
    workers: int,
    score_model: ScoreModel | None = None,
) -> list[Detections]:
    """Run ``detect_instances_from_probabilities`` on every map (optionally re-scored)."""
    state = {"config": config, "score_model": score_model}
    return _run_parallel(_detect_probabilities, probs, state, workers)


def detect_all_masks(
    masks: Sequence[np.ndarray], config: MannsteinConfig, workers: int
) -> list[Detections]:
    """Run plain Mannstein (``detect_instances``) on every binary mask."""
    return _run_parallel(_detect_mask, masks, {"config": config}, workers)


def evaluate_annotator_masks(
    samples: Sequence[Sample],
    config: MannsteinConfig,
    workers: int,
    per_threshold: bool = False,
    match_iou: float = 0.5,
) -> dict[str, Any]:
    """Run plain Mannstein on each sample's own annotator mask and score it (pooled mAP).

    Returns
    -------
    dict[str, Any]
        The result of ``evaluate_annotator_samples`` plus ``mean_pred_instances``.
    """
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    out = detect_all_masks([s.annotator_mask for s in samples], config, workers)
    preds = [o[0] for o in out]
    result = evaluate_annotator_samples(
        preds,
        [s.gt for s in samples],
        [o[1] for o in out],
        match_iou=match_iou,
        per_threshold=per_threshold,
    )
    result["mean_pred_instances"] = float(np.mean([len(p) for p in preds])) if preds else 0.0
    return result


def evaluate_pipeline(
    probs: Sequence[np.ndarray],
    gts: Sequence[Sequence[Sequence[np.ndarray]]],
    config: InstancePipelineConfig,
    workers: int,
    match_iou: float = 0.5,
    per_threshold: bool = False,
    consensus: Sequence[Sequence[np.ndarray]] | None = None,
    score_model: ScoreModel | None = None,
) -> dict[str, Any]:
    """Run the pipeline on every map and score it (pooled mAP).

    Parameters
    ----------
    probs
        Probability maps, one per scene.
    gts
        ``gts[i][a]`` holds the GT masks annotator ``a`` drew on scene ``i``.
    config
        Pipeline configuration.
    workers
        Worker processes (1 runs in-process).
    match_iou
        IoU threshold of the reference P/R/F1 matching.
    per_threshold
        Also return the AP at each IoU threshold.
    consensus
        If given, ``consensus[i]`` holds the consensus instances of scene ``i`` and each scene
        is scored once against them instead of once per annotator.
    score_model
        Optional learned scores.

    Returns
    -------
    dict[str, Any]
        ``evaluate_annotator_samples`` result (``map``, reference ``f1`` ...) plus
        ``mean_pred_instances`` per scene. Annotators who drew nothing are skipped.
    """
    from deepcontrail.instance.metrics import evaluate_annotator_samples

    detections = detect_all(probs, config, workers, score_model)
    if consensus is not None:
        preds = [d[0] for d in detections]
        scores = [d[1] for d in detections]
        targets = [list(c) for c in consensus]
    else:
        preds, scores, targets = [], [], []
        for (masks, scene_scores), scene_gt in zip(detections, gts, strict=True):
            for slot_gt in scene_gt:
                preds.append(masks)
                scores.append(scene_scores)
                targets.append(list(slot_gt))
    result = evaluate_annotator_samples(
        preds, targets, scores, match_iou=match_iou, per_threshold=per_threshold
    )
    result["mean_pred_instances"] = (
        float(np.mean([len(d[0]) for d in detections])) if detections else 0.0
    )
    return result
