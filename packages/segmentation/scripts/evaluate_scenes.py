"""Evaluate whole-scene probability maps against a dataset's labels.

Reads the float32 maps written by ``export_predictions.py`` (one per scene in
``--pred-dir``) and the labels of a DeepContrail-format dataset root, and reports pooled
Dice over a threshold sweep, per-confidence-level recall/Dice, day-bootstrap confidence
intervals and per-project Dice.

Everything is accumulated as integer counts per scene; the maps are never held at once, so
a whole split can be scored with a small memory footprint.

``metadata/scenes.json`` (optional) gives each scene its ``split``, ``group`` (day id),
``project`` and ordered ``contrails`` list (matching the instances of
``instance_masks/<dir>/<id>.npy``). Without it, the split dir is found by looking for the
mask in ``train/`` then ``test/``, every scene is its own bootstrap group, and the
per-level/per-project sections are skipped.

Usage
-----
::

    uv run python packages/segmentation/scripts/evaluate_scenes.py \
        --root /path/to/mtg_dataset --pred-dir runs/mtg/preds_test --out runs/mtg/scores.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from deepcontrail.core.readers import load_mask

logger = logging.getLogger(__name__)

SCENES_META_PATH = "metadata/scenes.json"
_LEVELS = ("High", "Medium", "Low")


def _default_thresholds() -> list[float]:
    """Threshold sweep: 0.05..0.95 in 0.05 steps (0.5 already included)."""
    steps = list(np.around(np.arange(0.05, 1.0, 0.05), 2))
    if 0.5 not in steps:
        steps.append(0.5)
    return sorted(float(t) for t in steps)


def _ratio(num: int, den: int) -> float:
    """``num / den`` as a float, 0.0 when the denominator is 0."""
    return float(num) / den if den else 0.0


def _dice_from_sums(overlap: int, predicted: int, target: int) -> float:
    """Pooled Dice from summed counts: ``2*overlap / (predicted + target)``.

    Returns 1.0 when both sides are empty (``predicted + target == 0``), matching the
    convention of :func:`deepcontrail.core.metrics.dice_coefficient`.
    """
    denom = predicted + target
    return 1.0 if denom == 0 else 2.0 * overlap / denom


def _new_group() -> dict[str, Any]:
    """Build a fresh per-group accumulator: pooled Dice sums and per-level sums."""
    levels: dict[str, dict[str, int]] = {}
    for level in _LEVELS:
        levels[level] = {"num": 0, "other_pred": 0, "gt": 0}
    return {"overlap": 0, "predicted": 0, "target": 0, "levels": levels}


def _load_scenes_meta(root: Path) -> dict[str, Any] | None:
    """Read ``metadata/scenes.json`` as a dict, or ``None`` when it is absent."""
    path = root / SCENES_META_PATH
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_subdir(root: Path, meta: dict[str, Any] | None, scene_id: str) -> str:
    """Resolve a scene's split directory: from scenes.json, else train/ then test/.

    With scenes.json any non-test split (train/val) lives in ``train/``; without it the
    mask is looked up in ``train/`` then ``test/``.
    """
    if meta is not None:
        return "test" if meta.get("split") == "test" else "train"
    if (root / "semantic_masks" / "train" / f"{scene_id}.png").exists():
        return "train"
    if (root / "semantic_masks" / "test" / f"{scene_id}.png").exists():
        return "test"
    msg = f"{scene_id}: not in scenes.json and no mask under train/ or test/ of {root}"
    raise FileNotFoundError(msg)


def _load_instances(path: Path) -> np.ndarray[Any, Any]:
    """Load an ``instance_masks/<dir>/<id>.npy`` array as booleans ``(N, H, W)``.

    The stored array is ``(N, H, W, 1)`` int8 with 1 inside a contrail; the trailing
    singleton axis is dropped. A 3-D array is accepted as-is.
    """
    raw = np.load(path)
    if raw.ndim == 4:
        raw = raw[..., 0]
    return raw.astype(bool)


def _accumulate_levels(
    pred0: np.ndarray[Any, Any],
    gt: np.ndarray[Any, Any],
    contrails: list[dict[str, Any]],
    subdir: str,
    root: Path,
    scene_id: str,
    group: dict[str, Any],
    level_tot: dict[str, dict[str, int]],
    none_counter: dict[str, int],
) -> np.ndarray[Any, Any]:
    """Add one scene's per-confidence-level counts; return the union of its Low pixels.

    ``gt_L`` is the union of instances whose confidence is ``L``; ``other_L`` (the pixels
    of every other instance) is what a prediction is penalised for outside ``gt_L``. A
    null confidence is its own level, counted but not scored.
    """
    instances = _load_instances(root / "instance_masks" / subdir / f"{scene_id}.npy")
    n_inst = instances.shape[0]
    unions = {L: np.zeros(gt.shape, dtype=bool) for L in _LEVELS}
    none_pixels = np.zeros(gt.shape, dtype=bool)
    total = np.zeros(gt.shape, dtype=bool)
    for i, contrail in enumerate(contrails):
        if i >= n_inst:
            break
        mask = instances[i]
        total |= mask
        confidence = contrail.get("confidence")
        if confidence in unions:
            unions[confidence] |= mask
        elif confidence is None:
            none_pixels |= mask

    for level in _LEVELS:
        gt_level = unions[level]
        other = total & ~gt_level
        num = int((pred0 & gt_level).sum())
        other_pred = int((pred0 & ~other).sum())
        gt_count = int(gt_level.sum())
        for bucket in (group["levels"][level], level_tot[level]):
            bucket["num"] += num
            bucket["other_pred"] += other_pred
            bucket["gt"] += gt_count

    none_counter["n_instances"] += sum(1 for c in contrails if c.get("confidence") is None)
    none_counter["n_pixels"] += int(none_pixels.sum())
    return unions["Low"]


def _group_column(
    group_sums: dict[str, dict[str, Any]], group_ids: list[str], level: str, key: str
) -> np.ndarray[Any, Any]:
    """One count field of every group, as a float64 array (for bootstrap resampling)."""
    return np.array([group_sums[g]["levels"][level][key] for g in group_ids], dtype=np.float64)


def _bootstrap_intervals(
    group_sums: dict[str, dict[str, Any]], bootstrap: int, seed: int, has_contrails: bool
) -> dict[str, Any]:
    """Day-bootstrap 2.5/97.5 percentiles of pooled Dice and of each recall_L.

    Groups are resampled with replacement; the pooled statistic is recomputed from the
    per-group integer sums of the resample.
    """
    group_ids = list(group_sums)
    n_groups = len(group_ids)
    if n_groups == 0 or bootstrap <= 0:
        return {}
    overlap = np.array([group_sums[g]["overlap"] for g in group_ids], dtype=np.float64)
    predicted = np.array([group_sums[g]["predicted"] for g in group_ids], dtype=np.float64)
    target = np.array([group_sums[g]["target"] for g in group_ids], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(bootstrap, n_groups))
    ov, pr, tg = overlap[idx].sum(axis=1), predicted[idx].sum(axis=1), target[idx].sum(axis=1)
    denom = pr + tg
    dice = np.divide(2.0 * ov, denom, out=np.ones_like(denom), where=denom > 0)
    intervals: dict[str, Any] = {
        "dice": [float(np.percentile(dice, 2.5)), float(np.percentile(dice, 97.5))]
    }

    if has_contrails:
        recall: dict[str, Any] = {}
        for level in _LEVELS:
            num = _group_column(group_sums, group_ids, level, "num")
            gt = _group_column(group_sums, group_ids, level, "gt")
            gt_sum = gt[idx].sum(axis=1)
            resampled = np.divide(
                num[idx].sum(axis=1), gt_sum, out=np.zeros_like(gt_sum), where=gt_sum > 0
            )
            recall[level] = [
                float(np.percentile(resampled, 2.5)),
                float(np.percentile(resampled, 97.5)),
            ]
        intervals["recall"] = recall
    return intervals


def _print_summary(result: dict[str, Any]) -> None:
    """Print a short human-readable view of the scores."""
    pooled = result["pooled"]
    print("=" * 60)
    print(
        f"Scenes: {result['config']['n_scenes']}  "
        f"threshold: {pooled['threshold']}  bootstrap: {result['config']['bootstrap']}"
    )
    print(
        f"Dice @0.5: {pooled['dice_at_0.5']:.4f}   "
        f"best {pooled['best_dice']:.4f} @ {pooled['best_threshold']}"
    )
    print(
        f"Dice @{pooled['threshold']}: {pooled['dice_at_threshold']:.4f}   "
        f"precision: {pooled['precision']:.4f}   recall: {pooled['recall']:.4f}"
    )
    if "by_confidence" in result:
        print("by confidence level:")
        for level, scores in result["by_confidence"].items():
            detail = (
                (
                    f"dice={scores.get('dice', float('nan')):.4f}  "
                    f"recall={scores.get('recall', float('nan')):.4f}"
                )
                if "dice" in scores
                else f"instances={scores['n_instances']}"
            )
            print(f"  {level:7s} {detail}")
    if "bootstrap" in result:
        ci = result["bootstrap"]["dice"]
        print(f"Dice CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Wrote {result['config']['out']}")
    print("=" * 60)


def _dice_curve(
    thresholds: list[float],
    overlap: dict[float, int],
    predicted: dict[float, int],
    target: int,
) -> dict[str, float]:
    """Pooled Dice at each threshold, keyed by its string form (the JSON sweep curve)."""
    return {str(t): _dice_from_sums(overlap[t], predicted[t], target) for t in thresholds}


def _best_over_sweep(
    thresholds: list[float],
    overlap: dict[float, int],
    predicted: dict[float, int],
    target: int,
) -> tuple[float, float]:
    """Return ``(threshold, dice)`` with the highest pooled Dice over the sweep (first wins)."""
    first = thresholds[0]
    best_t, best_d = first, _dice_from_sums(overlap[first], predicted[first], target)
    for t in thresholds[1:]:
        dice = _dice_from_sums(overlap[t], predicted[t], target)
        if dice > best_d:
            best_t, best_d = t, dice
    return best_t, best_d


def main(argv: list[str] | None = None) -> int:
    """Score the maps in ``--pred-dir`` against the dataset at ``--root``; return 0."""
    parser = argparse.ArgumentParser(description="Evaluate whole-scene probability maps.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (labels).")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory of *.npy maps.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the scores JSON.")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed reporting threshold.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Threshold sweep (default: 0.05..0.95 step 0.05).",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Number of day-bootstrap resamples."
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    args = parser.parse_args(argv)

    scenes_meta = _load_scenes_meta(args.root)
    pred_files = sorted(p for p in args.pred_dir.iterdir() if p.suffix == ".npy")

    if args.thresholds is not None:
        thresholds = sorted(set(args.thresholds))
    else:
        thresholds = _default_thresholds()
    t0 = args.threshold if args.threshold is not None else 0.5
    acc_thresholds = sorted(set(thresholds) | {0.5, t0})

    has_contrails = scenes_meta is not None and any(
        c.get("contrails") for c in scenes_meta.values()
    )

    sweep_overlap = dict.fromkeys(acc_thresholds, 0)
    sweep_pred = dict.fromkeys(acc_thresholds, 0)
    target_total = 0
    group_sums: dict[str, dict[str, Any]] = {}
    project_sums: dict[str, dict[str, int]] = {}
    level_tot: dict[str, dict[str, int]] = {}
    for _level in _LEVELS:
        level_tot[_level] = {"num": 0, "other_pred": 0, "gt": 0}
    none_counter = {"n_instances": 0, "n_pixels": 0}
    nlow = {"overlap": 0, "predicted": 0, "target": 0}

    for path in pred_files:
        scene_id = path.name[: -len(".npy")]
        meta = scenes_meta.get(scene_id) if scenes_meta else None
        subdir = _resolve_subdir(args.root, meta, scene_id)

        pred = np.load(path).astype(np.float32, copy=False)
        gt = load_mask(args.root / "semantic_masks" / subdir / f"{scene_id}.png") > 0
        if pred.shape != gt.shape:
            msg = f"{scene_id}: prediction {pred.shape} and mask {gt.shape} differ in size"
            raise ValueError(msg)

        target_total += int(gt.sum())
        for t in acc_thresholds:
            above = pred > t
            sweep_pred[t] += int(above.sum())
            sweep_overlap[t] += int((above & gt).sum())

        pred0 = pred > t0
        group_id = scene_id if meta is None else str(meta.get("group", scene_id))
        group = group_sums.setdefault(group_id, _new_group())
        group["overlap"] += int((pred0 & gt).sum())
        group["predicted"] += int(pred0.sum())
        group["target"] += int(gt.sum())

        if scenes_meta is not None and meta is not None and meta.get("project") is not None:
            project = str(meta["project"])
            counts = project_sums.setdefault(
                project, {"overlap": 0, "predicted": 0, "target": 0, "n": 0}
            )
            counts["overlap"] += int((pred0 & gt).sum())
            counts["predicted"] += int(pred0.sum())
            counts["target"] += int(gt.sum())
            counts["n"] += 1

        low_pixels = np.zeros(gt.shape, dtype=bool)
        if meta is not None and meta.get("contrails"):
            low_pixels = _accumulate_levels(
                pred0,
                gt,
                meta["contrails"],
                subdir,
                args.root,
                scene_id,
                group,
                level_tot,
                none_counter,
            )
        gt_nlow = gt & ~low_pixels
        pred_nlow = pred0 & ~low_pixels
        nlow["target"] += int(gt_nlow.sum())
        nlow["predicted"] += int(pred_nlow.sum())
        nlow["overlap"] += int((gt_nlow & pred_nlow).sum())

    dice_at_05 = _dice_from_sums(sweep_overlap[0.5], sweep_pred[0.5], target_total)
    dice_at_t0 = _dice_from_sums(sweep_overlap[t0], sweep_pred[t0], target_total)
    best_threshold, best_dice = _best_over_sweep(
        thresholds, sweep_overlap, sweep_pred, target_total
    )

    result: dict[str, Any] = {
        "config": {
            "root": str(args.root),
            "pred_dir": str(args.pred_dir),
            "out": str(args.out),
            "threshold": t0,
            "thresholds": thresholds,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "n_scenes": len(pred_files),
            "has_scenes_meta": scenes_meta is not None,
        },
        "pooled": {
            "dice_at_0.5": dice_at_05,
            "best_threshold": best_threshold,
            "best_dice": best_dice,
            "threshold": t0,
            "dice_at_threshold": dice_at_t0,
            "precision": _ratio(sweep_overlap[t0], sweep_pred[t0]),
            "recall": _ratio(sweep_overlap[t0], target_total),
            "dice_without_low": _dice_from_sums(nlow["overlap"], nlow["predicted"], nlow["target"]),
        },
        "dice_over_thresholds": _dice_curve(thresholds, sweep_overlap, sweep_pred, target_total),
    }

    if has_contrails:
        by_confidence: dict[str, Any] = {}
        for level in _LEVELS:
            num = level_tot[level]["num"]
            other_pred = level_tot[level]["other_pred"]
            gt_count = level_tot[level]["gt"]
            denom = other_pred + gt_count
            by_confidence[level] = {
                "recall": _ratio(num, gt_count),
                "dice": 2.0 * num / denom if denom else 1.0,
                "gt_pixels": gt_count,
                "pred_correct_pixels": num,
                "other_pixels": other_pred,
            }
        by_confidence["None"] = {
            "n_instances": none_counter["n_instances"],
            "gt_pixels": none_counter["n_pixels"],
        }
        result["by_confidence"] = by_confidence

    if project_sums:
        result["by_project"] = {
            project: {
                "dice": _dice_from_sums(c["overlap"], c["predicted"], c["target"]),
                "n_scenes": c["n"],
            }
            for project, c in project_sums.items()
        }
    if group_sums:
        result["bootstrap"] = _bootstrap_intervals(
            group_sums, args.bootstrap, args.seed, has_contrails
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    _print_summary(result)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
