"""Train a learned contrail score (``deepcontrail.instance.ScoreModel``) for a semantic model.

A score model belongs to one semantic model and one pipeline config. Train it on that model's
probability maps for scenes it never trained on, such as its validation split (``--pred-dir``,
GT read from ``<root>/instance_masks/<split>``, ``--split`` default ``train``). The scenes are
split like ``scripts/optimize_pipeline.py`` (same arguments; by default half and half): the
model is fitted on the search scenes, and the holdout scenes compare the pipeline's scores with
the learned ones (pooled mAP 0.25-0.75 per annotator sample, and against the consensus
instances). The test split is never read.

Usage
-----
::

    uv run python packages/instance/scripts/train_score_model.py --root <dataset_root>
        --pred-dir <folder of validation probability maps>
        --pipeline-config packages/instance/configs/instance_pipeline.yaml
        --semantic-model <repo id or run name> --tta
        --workers 16 --output-dir <dir>

Outputs: ``score_model.json`` (load with ``ScoreModel.from_json``; pass it to
``detect_instances_from_probabilities(..., score_model=...)`` or ``evaluate_instances.py
--score-model``) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from deepcontrail.core.readers import load_instance_masks
from deepcontrail.instance import (
    InstancePipelineConfig,
    ScoreModel,
    match_quality,
)
from deepcontrail.instance.evaluation import (
    default_workers,
    detect_all,
    evaluate_pipeline,
    load_predictions,
    split_scenes,
)
from deepcontrail.instance.labels import consensus_instances
from deepcontrail.instance.scoring import scene_features

logger = logging.getLogger(__name__)


def holdout_scores(
    probs: list[np.ndarray],
    gts: list[list[list[np.ndarray]]],
    consensus: list[list[np.ndarray]],
    config: InstancePipelineConfig,
    score_model: ScoreModel | None,
    workers: int,
) -> dict[str, Any]:
    """Pooled mAP per annotator sample and against the consensus, on the given scenes."""
    annotators = evaluate_pipeline(
        probs, gts, config, workers, per_threshold=True, score_model=score_model
    )
    cons = evaluate_pipeline(
        probs,
        gts,
        config,
        workers,
        per_threshold=True,
        consensus=consensus,
        score_model=score_model,
    )
    return {
        "annotators_map": annotators["map"],
        "annotators_ap": annotators["ap_per_threshold"],
        "annotators_f1": annotators["f1"],
        "consensus_map": cons["map"],
        "consensus_ap": cons["ap_per_threshold"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a learned contrail score model.")
    parser.add_argument("--root", required=True, help="Dataset root (GT in instance_masks/).")
    parser.add_argument("--pred-dir", required=True, help="Validation-split probability maps.")
    parser.add_argument(
        "--split", default="train", help="Split folder holding their GT (default: train)."
    )
    parser.add_argument("--pipeline-config", default=None, help="Default: library defaults.")
    parser.add_argument(
        "--n-images", type=int, default=None, help="Training scenes (default: half)."
    )
    parser.add_argument(
        "--n-holdout", type=int, default=None, help="Holdout scenes (default: the rest)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Scene split seed.")
    parser.add_argument(
        "--workers", type=int, default=default_workers(), help="Worker processes (default: ≤ 8)."
    )
    parser.add_argument(
        "--semantic-model", default=None, help="Model that made the maps (recorded in the file)."
    )
    parser.add_argument("--semantic-revision", default=None, help="Its revision (recorded).")
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether the maps were made with TTA (recorded; checked when the model is loaded).",
    )
    parser.add_argument("--output-dir", required=True, help="score_model.json, summary.json.")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Fit the score model on the search scenes and compare scores on the holdout."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = (
        InstancePipelineConfig.from_yaml(args.pipeline_config)
        if args.pipeline_config
        else InstancePipelineConfig()
    )
    stems = sorted(p.stem for p in Path(args.pred_dir).glob("*.npy"))
    search_stems, holdout_stems = split_scenes(stems, args.seed, args.n_images, args.n_holdout)

    probs, gts = load_predictions(args.pred_dir, args.root, search_stems, args.split)
    detections = detect_all(probs, config, args.workers)
    x = np.concatenate(
        [scene_features(m, p, s) for (m, s), p in zip(detections, probs, strict=True)]
    )
    y = np.array(
        [match_quality(m, g) for (masks, _), g in zip(detections, gts, strict=True) for m in masks]
    )
    logger.info("Training on %d contrails from %d scenes", len(x), len(search_stems))
    shapes = {p.shape for p in probs}
    if len(shapes) != 1:
        raise ValueError(f"all probability maps must share one size, got {sorted(shapes)}")
    trained_on = {
        "semantic_model": args.semantic_model,
        "semantic_revision": args.semantic_revision,
        "tta": args.tta,
        "scene_shape": list(shapes.pop()),
    }
    model = ScoreModel.fit(x, y, config, {k: v for k, v in trained_on.items() if v is not None})
    model.to_json(out / "score_model.json")

    probs, gts = load_predictions(args.pred_dir, args.root, holdout_stems, args.split)
    instance_dir = Path(args.root) / "instance_masks" / args.split
    consensus = [
        consensus_instances(load_instance_masks(instance_dir / f"{s}.npy")) for s in holdout_stems
    ]
    summary: dict[str, Any] = {
        "pipeline_config": args.pipeline_config or "library defaults",
        "seed": args.seed,
        "n_training_scenes": len(search_stems),
        "n_training_contrails": len(x),
        "n_holdout_scenes": len(holdout_stems),
        "holdout": {
            "pipeline_scores": holdout_scores(probs, gts, consensus, config, None, args.workers),
            "learned_scores": holdout_scores(probs, gts, consensus, config, model, args.workers),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["holdout"], indent=2, default=str))
    return summary


if __name__ == "__main__":
    main(parse_args())
