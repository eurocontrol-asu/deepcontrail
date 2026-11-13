"""Optuna search over the probability-map instance pipeline.

Tunes ``InstancePipelineConfig`` (semantic threshold, growth into moderate-probability
pixels, and the Mannstein settings) for the pooled mAP 0.25-0.75 per annotator sample
(one sample per scene and annotator, empty samples skipped) on a folder of semantic-model
probability maps. Use scenes the semantic model never trained on, such as its validation
split, so the search sees realistic predictions without touching the test set. The ground
truth is read from ``<root>/instance_masks/<split>`` (``--split``, default ``train``).

The scenes of ``--pred-dir`` are split into a search set and a disjoint holdout set (by default
half and half; the holdout must not be empty). The baseline (``--baseline-config`` Mannstein
settings, threshold 0.5, no growth), the baseline with one growth step and the shipped defaults
(``InstancePipelineConfig()``) are enqueued first. The best trial and these three starting
configs are re-scored on the holdout, which is the number to trust.

``--objective consensus`` scores each scene once against its consensus instances (contrails
drawn by more than half of the annotators) instead of once per annotator.

Usage
-----
::

    systemd-run --user --scope -p MemoryMax=32G
        uv run python packages/instance/scripts/optimize_pipeline.py
        --root <dataset_root>
        --pred-dir <folder of validation probability maps>
        --baseline-config packages/instance/configs/mannstein_annotator_masks.yaml
        --n-trials 100 --workers 16 --output-dir <output_dir>

Outputs: ``study.db`` (resumable), ``best_pipeline.yaml`` (loadable with
``InstancePipelineConfig.from_yaml``) and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from deepcontrail.core import scene_stems
from deepcontrail.core.readers import load_instance_masks
from deepcontrail.instance import InstancePipelineConfig, MannsteinConfig
from deepcontrail.instance._evaluation import (
    default_workers,
    evaluate_pipeline,
    load_predictions,
    split_scenes,
)
from deepcontrail.instance._tuning import pipeline_trial_params, suggest_pipeline_config
from deepcontrail.instance.labels import consensus_instances

if TYPE_CHECKING:
    from optuna.trial import BaseTrial

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna search over the instance pipeline on semantic-model predictions."
    )
    parser.add_argument("--root", required=True, help="Dataset root (GT in instance_masks/).")
    parser.add_argument(
        "--pred-dir", required=True, help="Folder of <stem>.npy probability maps to tune on."
    )
    parser.add_argument(
        "--split", default="train", help="Split folder holding their GT (default: train)."
    )
    parser.add_argument(
        "--baseline-config",
        default=None,
        help="YAML with a mannstein: section for the baseline (default: MannsteinConfig()).",
    )
    parser.add_argument(
        "--n-images", type=int, default=None, help="Search-set size (default: half the scenes)."
    )
    parser.add_argument(
        "--n-holdout", type=int, default=None, help="Holdout size (default: the other scenes)."
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials.")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Split and sampler seed.")
    parser.add_argument(
        "--workers", type=int, default=default_workers(), help="Worker processes (default: ≤ 8)."
    )
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU of the reference P/R/F1.")
    parser.add_argument(
        "--objective",
        choices=["annotators", "consensus"],
        default="annotators",
        help="Score per annotator sample (default) or once per scene against the consensus.",
    )
    parser.add_argument("--output-dir", required=True, help="Output folder.")
    parser.add_argument("--study-name", default="pipeline", help="Optuna study name.")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Run the search and write ``best_pipeline.yaml`` and ``summary.json``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = scene_stems(args.pred_dir, ".npy")
    if not stems:
        raise FileNotFoundError(f"no .npy predictions in {args.pred_dir}")
    search_stems, holdout_stems = split_scenes(stems, args.seed, args.n_images, args.n_holdout)
    search_probs, search_gts = load_predictions(args.pred_dir, args.root, search_stems, args.split)
    holdout_probs, holdout_gts = load_predictions(
        args.pred_dir, args.root, holdout_stems, args.split
    )
    logger.info("Search scenes: %d, holdout scenes: %d", len(search_stems), len(holdout_stems))
    search_consensus = holdout_consensus = None
    if args.objective == "consensus":
        instance_dir = Path(args.root) / "instance_masks" / args.split
        search_consensus = [
            consensus_instances(load_instance_masks(instance_dir / f"{stem}.npy"))
            for stem in search_stems
        ]
        holdout_consensus = [
            consensus_instances(load_instance_masks(instance_dir / f"{stem}.npy"))
            for stem in holdout_stems
        ]

    mannstein = (
        MannsteinConfig.from_yaml(args.baseline_config)
        if args.baseline_config
        else MannsteinConfig()
    )
    baseline = InstancePipelineConfig(semantic_threshold=0.5, grow_steps=0, mannstein=mannstein)
    baseline_grown = InstancePipelineConfig(
        semantic_threshold=0.5,
        grow_threshold=0.4,
        grow_steps=1,
        grow_connectivity=8,
        mannstein=mannstein,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=f"sqlite:///{(output_dir / 'study.db').as_posix()}",
        load_if_exists=True,
    )
    starting = {
        "baseline": baseline,
        "baseline_grown": baseline_grown,
        "defaults": InstancePipelineConfig(),
    }
    if not study.trials:
        for config in starting.values():
            study.enqueue_trial(pipeline_trial_params(config))

    def objective(trial: BaseTrial) -> float:
        metrics = evaluate_pipeline(
            search_probs,
            search_gts,
            suggest_pipeline_config(trial),
            args.workers,
            args.match_iou,
            consensus=search_consensus,
        )
        for key in ("f1", "precision", "recall", "mean_pred_instances"):
            trial.set_user_attr(key, metrics[key])
        return metrics["map"]

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best = suggest_pipeline_config(optuna.trial.FixedTrial(study.best_trial.params))
    best.to_yaml(output_dir / "best_pipeline.yaml")
    holdout = {
        name: evaluate_pipeline(
            holdout_probs,
            holdout_gts,
            config,
            args.workers,
            args.match_iou,
            per_threshold=True,
            consensus=holdout_consensus,
        )
        for name, config in (starting | {"best": best}).items()
    }
    summary: dict[str, Any] = {
        "seed": args.seed,
        "objective": args.objective,
        "search_size": len(search_stems),
        "holdout_size": len(holdout_stems),
        "n_trials_completed": sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        ),
        "search_map": {"baseline": study.trials[0].value, "best": study.best_value},
        "holdout": holdout,
        "best_config": best.model_dump(),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps({k: summary[k] for k in ("search_map", "holdout")}, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main(parse_args())
