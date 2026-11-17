"""Streamed whole-scene instance segmentation evaluation (pooled mAP over IoU 0.25-0.75).

Like ``evaluate_instances.py``, but built for large, crowded frames (MTG is 1024 x 1024 with
~30 contrails per scene and 600+ scenes): it scores one scene per worker task and keeps only
run-length encodings (pycocotools RLE dicts, a few KB each) and per-scene match counts, so the
dense ground-truth and predicted masks are never held at once. ``evaluate_instances.py`` loads
every scene's masks together and was killed at 80 GB on the MTG test split; this one stays
per-scene.

Scenes are the ``<id>.npy`` probability maps in ``--pred-dir`` (a ``manifest.json`` there is
ignored). Each scene's ground truth is ``instance_masks/<split>/<id>.npy``; the split dir comes
from ``metadata/scenes.json`` ("test" -> test/, anything else -> train/) and, without it, from
whichever of train/ or test/ holds the file. ``metadata/scenes.json`` also gives each scene its
``group`` (day id, for the bootstrap) and its ``contrails`` in instance order (the ``confidence``
of each, for ``--by-confidence``).

One sample is one annotator who drew something (as in ``evaluate_instances.py``); an instance an
annotator never drew (value -1) is absent from their ground truth. All non-empty samples are
pooled into one COCO segmentation mAP (the headline), with precision/recall/F1 from one-to-one
matching at ``--match-iou`` and a ``--include-empty``-style ``with_empty`` view that keeps the
empty scenes (every detection on them is false), closer to real use. ``--by-confidence`` also
reports the mAP and recall of each confidence level, scoring only that level's contrails and
ignoring the others. A day-bootstrap 95% interval for F1 and recall (from per-group counts)
rounds it off.

The pipeline is ``--pipeline-config`` (default ``InstancePipelineConfig()``), re-scored by
``--score-model``; ``--bundle`` takes both from a pretrained bundle folder
(``load_instance_step``).

Usage
-----
::

    uv run python packages/instance/scripts/evaluate_scenes_instances.py \
        --root /path/to/mtg_dataset --pred-dir runs/mtg/preds_test \
        --pipeline-config packages/instance/configs/instance_pipeline.yaml \
        --output runs/mtg/instance_scores.json --by-confidence
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from deepcontrail.core.contract import scene_stems
from deepcontrail.core.readers import extract_annotator_instances, load_instance_masks
from deepcontrail.instance._evaluation import default_workers
from deepcontrail.instance.bundle import load_instance_step
from deepcontrail.instance.config import InstancePipelineConfig
from deepcontrail.instance.metrics import (
    DEFAULT_IOU_THRESHOLDS,
    _preds_matched_to,
    _split_ignore,
    coco_map_from_rles,
    match_instances,
)
from deepcontrail.instance.pipeline import detect_instances_from_probabilities
from deepcontrail.instance.scoring import ScoreModel

logger = logging.getLogger(__name__)

_SCENES_META_PATH = "metadata/scenes.json"
_LEVELS = ("High", "Medium", "Low")
_BOOTSTRAP = 1000
_SEED = 0


# --- worker (one scene per task; only RLEs and counts cross the process boundary) ---


_WORKER: dict[str, Any] = {}


def _worker_pipeline(
    *,
    bundle: str | None,
    config: dict[str, Any],
    score_model: str | None,
) -> tuple[InstancePipelineConfig, ScoreModel | None]:
    """Load the run's pipeline config and optional score model once per worker process.

    A ``bundle`` folder provides both (``load_instance_step``); otherwise the config is rebuilt
    from its dict and the score model loaded from its path. Cached in a module global, so a
    worker that scores many scenes pays for both only once.
    """
    if "pipeline" not in _WORKER:
        if bundle is not None:
            step = load_instance_step(bundle)
            _WORKER["pipeline"] = (step.config, step.score_model)
        else:
            _WORKER["pipeline"] = (
                InstancePipelineConfig.from_dict(config),
                ScoreModel.from_json(score_model) if score_model else None,
            )
    return _WORKER["pipeline"]


def _encode(mask: np.ndarray) -> dict:
    """Encode one boolean mask as a pycocotools RLE dict (``pycocotools.mask`` imported once)."""
    if "_mask_utils" not in _WORKER:
        import pycocotools.mask as mask_utils

        _WORKER["_mask_utils"] = mask_utils
    return _WORKER["_mask_utils"].encode(np.asfortranarray(mask.astype(np.uint8)))


def _scene_confidences(meta: dict[str, Any] | None, n_instances: int) -> list[str | None]:
    """Confidence of each rasterised instance, in instance order, truncated to what exists."""
    if meta is None:
        return []
    contrails = meta.get("contrails") or []
    return [c.get("confidence") if isinstance(c, dict) else None for c in contrails[:n_instances]]


def evaluate_scene(
    scene_id: str,
    pred_path: str,
    gt_path: str,
    meta: dict[str, Any] | None,
    config: dict[str, Any],
    score_model: str | None,
    bundle: str | None,
    match_iou: float,
    by_confidence: bool,
) -> dict[str, Any]:
    """Detect, match and encode one scene; return only its RLEs, counts, group and confidences.

    The predicted masks come from :func:`detect_instances_from_probabilities`; the ground truth
    from the scene's instance-mask array. One sample is one annotator who drew something, and an
    instance that annotator marked ``-1`` is absent from their ground truth. Per-annotator match
    counts use one-to-one matching at ``match_iou``; with ``by_confidence``, per-level counts add
    the other levels' ground truth as ignored (dropped before matching), mirroring the pooled
    metric's ``gt_ignore``. Nothing dense is kept, so a worker holds one scene at a time.
    """
    probs = np.load(pred_path).astype(np.float32)
    config_obj, score_model_obj = _worker_pipeline(
        bundle=bundle, config=config, score_model=score_model
    )
    masks, scores = detect_instances_from_probabilities(probs, config_obj, score_model_obj)
    raw = load_instance_masks(gt_path)
    n_instances, height, width, n_annotators = raw.shape

    gt_per_annotator = [extract_annotator_instances(raw, a) for a in range(n_annotators)]
    annotator_gt_rles = [[_encode(m) for m in g] for g in gt_per_annotator]
    pred_rles = [_encode(m) for m in masks]
    annotator_counts = [list(match_instances(masks, g, match_iou)) for g in gt_per_annotator]

    result: dict[str, Any] = {
        "scene": scene_id,
        "group": scene_id if meta is None else str(meta.get("group", scene_id)),
        "shape": (int(height), int(width)),
        "pred_rles": pred_rles,
        "scores": [float(s) for s in scores],
        "annotator_gt_rles": annotator_gt_rles,
        "annotator_counts": annotator_counts,
    }

    if by_confidence:
        gt0 = gt_per_annotator[0]
        confidences = _scene_confidences(meta, n_instances)
        # keep each instance's confidence in the same order extract_annotator_instances kept it
        aligned = [
            confidences[n] if n < len(confidences) else None
            for n in range(n_instances)
            if (raw[n, :, :, 0] == 1).any()
        ]
        result["confidences"] = aligned
        level_counts: dict[str, list[int]] = {}
        for level in _LEVELS:
            ignore = [c != level for c in aligned]
            kept_g, ignored = _split_ignore(gt0, ignore)
            dropped = _preds_matched_to(masks, ignored, match_iou)
            kept_p = [m for k, m in enumerate(masks) if k not in dropped]
            level_counts[level] = list(match_instances(kept_p, kept_g, match_iou))
        result["level_counts"] = level_counts

    return result


def _evaluate_scene_wrapper(args: tuple) -> dict[str, Any]:
    """Unpack task tuple for ProcessPoolExecutor.map.

    starmap is a Pool method, not available on Executor. Must be module-level for pickling.
    """
    return evaluate_scene(*args)


# --- main: pool the per-scene RLEs into the reported metrics ---


def _load_scenes_meta(root: Path) -> dict[str, Any] | None:
    """Read ``metadata/scenes.json`` as a dict, or ``None`` when it is absent."""
    path = root / _SCENES_META_PATH
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_split_dir(root: Path, meta: dict[str, Any] | None, scene_id: str) -> str:
    """Resolve a scene's split directory: from scenes.json, else train/ then test/.

    With scenes.json any non-test split (train/val) lives in ``train/``; without it the masks
    are looked up in ``train/`` then ``test/``.
    """
    if meta is not None:
        return "test" if meta.get("split") == "test" else "train"
    for split in ("train", "test"):
        if (root / "instance_masks" / split / f"{scene_id}.npy").exists():
            return split
    raise FileNotFoundError(
        f"{scene_id}: not in scenes.json and no instance masks under train/ or test/ of {root}"
    )


def _pipeline_config(args: argparse.Namespace) -> InstancePipelineConfig:
    """Return the reported pipeline config from a bundle, a YAML file, or the defaults."""
    if args.bundle:
        return load_instance_step(args.bundle).config
    if args.pipeline_config:
        return InstancePipelineConfig.from_yaml(args.pipeline_config)
    return InstancePipelineConfig()


def _bootstrap_prf(group_counts: list[list[int]], seed: int = _SEED) -> dict[str, Any]:
    """Day-bootstrap 95% interval of pooled recall and F1 from per-group match counts.

    Groups are resampled with replacement and the pooled statistic recomputed from the resample's
    summed counts; the observed value is reported alongside the interval.
    """
    n = len(group_counts)
    if n == 0:
        return {"recall": [0.0, 0.0], "f1": [0.0, 0.0], "n_groups": 0}
    counts = np.array(group_counts, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, n, size=(_BOOTSTRAP, n))
    summed = counts[idx].sum(axis=1)
    precision = np.divide(
        summed[:, 0],
        summed[:, 0] + summed[:, 1],
        out=np.zeros_like(summed[:, 0]),
        where=(summed[:, 0] + summed[:, 1]) > 0,
    )
    recall = np.divide(
        summed[:, 0],
        summed[:, 0] + summed[:, 2],
        out=np.zeros_like(summed[:, 0]),
        where=(summed[:, 0] + summed[:, 2]) > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(summed[:, 0]),
        where=(precision + recall) > 0,
    )
    observed = _prf(*[float(v) for v in summed.sum(axis=0)])
    return {
        "recall": [float(np.percentile(recall, 2.5)), float(np.percentile(recall, 97.5))],
        "f1": [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))],
        "recall_point": observed[1],
        "f1_point": observed[2],
        "n_groups": n,
    }


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Precision, recall and F1 from one-to-one match counts (zeros when a denominator is 0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _by_confidence(results: list[dict[str, Any]]) -> dict[str, Any]:
    """MAP and recall per confidence level (only scenes with that level are scored for it).

    For level ``L`` the ground truth is annotator 0's instances of level ``L`` and every other
    instance is ignored (a crowd region); a scene with no ``L`` contrail is not scored for ``L``,
    so it neither adds false detections nor dilutes the recall.
    """
    buckets: dict[str, dict[str, list]] = {}
    for level in _LEVELS:
        buckets[level] = {"preds": [], "gts": [], "scores": [], "shapes": [], "ignore": []}
    level_tp = dict.fromkeys(_LEVELS, 0)
    level_fn = dict.fromkeys(_LEVELS, 0)
    for res in results:
        confidences = res["confidences"]
        gt0_rles = res["annotator_gt_rles"][0]
        present = {level for level in _LEVELS if any(c == level for c in confidences)}
        for level in present:
            bucket = buckets[level]
            bucket["preds"].append(res["pred_rles"])
            bucket["gts"].append(gt0_rles)
            bucket["scores"].append(res["scores"])
            bucket["shapes"].append(res["shape"])
            bucket["ignore"].append([c != level for c in confidences])
            level_tp[level] += res["level_counts"][level][0]
            level_fn[level] += res["level_counts"][level][2]
    out: dict[str, Any] = {}
    for level in _LEVELS:
        bucket = buckets[level]
        if bucket["gts"]:
            level_map = coco_map_from_rles(
                bucket["preds"],
                bucket["gts"],
                bucket["scores"],
                bucket["shapes"],
                gt_ignore=bucket["ignore"],
            )[0]
        else:
            level_map = 0.0
        total = level_tp[level] + level_fn[level]
        out[level] = {
            "map": float(level_map),
            "recall": level_tp[level] / total if total else 0.0,
        }
    return out


def _print_summary(result: dict[str, Any]) -> None:
    """Print a short human-readable view of the scores."""
    print("=" * 64)
    print("INSTANCE SEGMENTATION EVALUATION (streamed, per annotator sample)")
    print("=" * 64)
    cfg = result["config"]
    print(f"Scenes:                {cfg['n_scenes']}  workers: {cfg['workers']}")
    print(
        f"mAP@[0.25:0.75]:       {result['map']:.4f}   "
        f"reference @IoU {cfg['match_iou']}: P {result['precision']:.4f} / "
        f"R {result['recall']:.4f} / F1 {result['f1']:.4f}"
    )
    ap = "  ".join(f"{t:.2f}: {v:.4f}" for t, v in result["ap_per_threshold"].items())
    print(f"AP per IoU threshold:  {ap}")
    print(
        f"Instances per scene:   predicted {result['mean_pred_instances_per_scene']:.2f}   "
        f"(TP {result['tp']}, FP {result['fp']}, FN {result['fn']})"
    )
    we = result["with_empty"]
    print(
        f"With empty scenes:     mAP {we['map']:.4f}   "
        f"P {we['precision']:.4f} / R {we['recall']:.4f} / F1 {we['f1']:.4f} (FP {we['fp']})"
    )
    if result.get("by_confidence"):
        print("-" * 64)
        print("by confidence level:")
        for level, scores in result["by_confidence"].items():
            print(f"  {level:7s} mAP {scores['map']:.4f}   recall {scores['recall']:.4f}")
    b = result["bootstrap"]
    print(
        f"Bootstrap (n={b['n_groups']} groups): F1 [{b['f1'][0]:.4f}, {b['f1'][1]:.4f}]   "
        f"recall [{b['recall'][0]:.4f}, {b['recall'][1]:.4f}]"
    )
    print(f"Wrote {result['config']['output']}")
    print("=" * 64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Streamed whole-scene instance evaluation (pooled mAP)."
    )
    parser.add_argument("--root", required=True, help="Dataset root (labels).")
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Folder of per-scene <id>.npy probability maps (a manifest.json is ignored).",
    )
    parser.add_argument("--output", required=True, help="Where to write the scores JSON.")
    parser.add_argument(
        "--pipeline-config",
        default=None,
        help="InstancePipelineConfig YAML (default: InstancePipelineConfig()).",
    )
    parser.add_argument(
        "--score-model",
        default=None,
        help="ScoreModel JSON that re-scores the detections (one probability map per scene).",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Pretrained bundle folder: take the pipeline config and score model from it.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help=(
            "Max detections per sample kept by the COCO mAP (default: 300). Raise on crowded "
            "frames with more labelled contrails, or the cap truncates recall."
        ),
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU of the reference precision/recall/F1 (default: 0.5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Worker processes (default: up to 8, at most the CPU count).",
    )
    parser.add_argument(
        "--by-confidence",
        action="store_true",
        help="Also report mAP and recall per confidence level (needs metadata/scenes.json).",
    )
    parser.add_argument("--limit", type=int, default=None, help="First N scenes only.")
    args = parser.parse_args(argv)
    if args.bundle and (args.pipeline_config or args.score_model):
        parser.error("--bundle already provides the pipeline config and score model")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Score the maps in ``--pred-dir`` against the dataset at ``--root`` and write the JSON."""
    args = parse_args(argv)
    root = Path(args.root)
    pred_dir = Path(args.pred_dir)
    scenes_meta = _load_scenes_meta(root)
    if args.by_confidence and scenes_meta is None:
        raise FileNotFoundError("--by-confidence needs metadata/scenes.json in the dataset root")

    stems = scene_stems(pred_dir, ".npy")
    if args.limit is not None:
        stems = stems[: args.limit]
    if not stems:
        raise FileNotFoundError(f"no *.npy predictions in {pred_dir}")

    config_dict = _pipeline_config(args).model_dump()
    tasks = []
    for stem in stems:
        meta = scenes_meta.get(stem) if scenes_meta else None
        split_dir = _resolve_split_dir(root, meta, stem)
        tasks.append(
            (
                stem,
                str(pred_dir / f"{stem}.npy"),
                str(root / "instance_masks" / split_dir / f"{stem}.npy"),
                meta,
                config_dict,
                args.score_model,
                args.bundle,
                args.match_iou,
                args.by_confidence,
            )
        )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_evaluate_scene_wrapper, tasks, chunksize=1))

    # Pool the per-annotator samples: official skips empty-GT samples, with_empty keeps them.
    off = {"preds": [], "gts": [], "scores": [], "shapes": []}
    we = {"preds": [], "gts": [], "scores": [], "shapes": []}
    off_tp = off_fp = off_fn = 0
    we_tp = we_fp = we_fn = 0
    group_counts: dict[str, list[int]] = {}
    total_pred_instances = 0
    for res in results:
        total_pred_instances += len(res["pred_rles"])
        group = group_counts.setdefault(res["group"], [0, 0, 0])
        for gt_rles, (tp, fp, fn) in zip(
            res["annotator_gt_rles"], res["annotator_counts"], strict=True
        ):
            we["preds"].append(res["pred_rles"])
            we["gts"].append(gt_rles)
            we["scores"].append(res["scores"])
            we["shapes"].append(res["shape"])
            we_tp += tp
            we_fp += fp
            we_fn += fn
            if gt_rles:  # official skips empty-GT samples
                off["preds"].append(res["pred_rles"])
                off["gts"].append(gt_rles)
                off["scores"].append(res["scores"])
                off["shapes"].append(res["shape"])
                off_tp += tp
                off_fp += fp
                off_fn += fn
                group[0] += tp
                group[1] += fp
                group[2] += fn

    o_map, o_ap = coco_map_from_rles(off["preds"], off["gts"], off["scores"], off["shapes"])
    w_map, w_ap = coco_map_from_rles(we["preds"], we["gts"], we["scores"], we["shapes"])
    o_p, o_r, o_f1 = _prf(off_tp, off_fp, off_fn)
    w_p, w_r, w_f1 = _prf(we_tp, we_fp, we_fn)

    ap_per_threshold = {t: float(a) for t, a in zip(DEFAULT_IOU_THRESHOLDS, o_ap, strict=True)}
    result: dict[str, Any] = {
        "config": {
            "root": str(root),
            "pred_dir": str(pred_dir),
            "output": str(args.output),
            "pipeline_config": args.pipeline_config,
            "score_model": args.score_model,
            "bundle": args.bundle,
            "semantic_threshold": config_dict.get("semantic_threshold"),
            "grow_steps": config_dict.get("grow_steps"),
            "match_iou": args.match_iou,
            "max_detections": args.max_detections,
            "by_confidence": args.by_confidence,
            "workers": args.workers,
            "n_scenes": len(results),
        },
        "n_scenes": len(results),
        "map": float(o_map),
        "ap_per_threshold": ap_per_threshold,
        "precision": o_p,
        "recall": o_r,
        "f1": o_f1,
        "tp": off_tp,
        "fp": off_fp,
        "fn": off_fn,
        "mean_pred_instances_per_scene": total_pred_instances / len(results) if results else 0.0,
        "with_empty": {
            "map": float(w_map),
            "ap_per_threshold": {
                t: float(a) for t, a in zip(DEFAULT_IOU_THRESHOLDS, w_ap, strict=True)
            },
            "precision": w_p,
            "recall": w_r,
            "f1": w_f1,
            "tp": we_tp,
            "fp": we_fp,
            "fn": we_fn,
        },
        "bootstrap": _bootstrap_prf(list(group_counts.values())),
    }
    if args.by_confidence:
        result["by_confidence"] = _by_confidence(results)

    _print_summary(result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
