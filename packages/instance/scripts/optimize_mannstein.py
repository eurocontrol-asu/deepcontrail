"""Optuna search for the Mannstein detector on train annotator samples (pooled mAP).

One sample is one annotator on one training scene: that annotator's own semantic mask is
the input and their own instances are the ground truth; annotators who drew nothing are
skipped. The objective is the COCO segmentation mAP over IoU 0.25-0.75, pooled over all
search samples. Scenes are shuffled with a seed and split into a search set and a disjoint
holdout set (a scene never contributes to both). The test split is never read.

Only ``instance_masks/train`` is read from the dataset (any number of annotators, one included).

Configs given with ``--baseline-configs`` (e.g. previously tuned or reference settings)
are enqueued as the first trials and re-scored on the holdout with the winner.

Usage
-----
::

    systemd-run --user --scope -p MemoryMax=32G
        uv run python packages/instance/scripts/optimize_mannstein.py
        --root <dataset_root>
        --n-samples 1000 --n-holdout 1000 --n-trials 200 --workers 16
        --baseline-configs packages/instance/configs/mannstein_annotator_masks.yaml
        --output-dir <output_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from deepcontrail.instance._evaluation import (
    default_workers,
    evaluate_annotator_masks,
    select_annotator_samples,
)
from deepcontrail.instance._tuning import mannstein_trial_params, suggest_mannstein_config
from deepcontrail.instance.config import MannsteinConfig

if TYPE_CHECKING:
    from optuna.trial import BaseTrial

logger = logging.getLogger(__name__)


def write_best_yaml(config: MannsteinConfig, path: str | Path) -> None:
    """Write ``config`` as a ``mannstein:`` section (loadable by ``MannsteinConfig.from_yaml``)."""
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        yaml.safe_dump({"mannstein": config.model_dump()}, f, sort_keys=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Optuna search for Mannstein on train annotator samples (pooled mAP)."
    )
    parser.add_argument("--root", required=True, help="Dataset root path.")
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Search samples (default: 1000)."
    )
    parser.add_argument("--n-holdout", type=int, default=1000, help="Holdout samples.")
    parser.add_argument("--n-trials", type=int, default=200, help="Trials (default: 200).")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Scene shuffle and sampler seed.")
    parser.add_argument(
        "--workers", type=int, default=default_workers(), help="Worker processes (default: ≤ 8)."
    )
    parser.add_argument(
        "--baseline-configs",
        nargs="*",
        default=[],
        help="YAML files with a mannstein: section, enqueued first and re-scored on the holdout.",
    )
    parser.add_argument("--output-dir", required=True, help="study.db, best yaml, summary.json.")
    parser.add_argument("--study-name", default="mannstein_map", help="Optuna study name.")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Run the search, then write the best config and a holdout summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search, holdout = select_annotator_samples(args.root, args.n_samples, args.n_holdout, args.seed)
    logger.info(
        "Search: %d samples / %d scenes; holdout: %d samples / %d scenes",
        len(search),
        len({s.stem for s in search}),
        len(holdout),
        len({s.stem for s in holdout}),
    )
    baselines = [MannsteinConfig.from_yaml(p) for p in args.baseline_configs]

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=f"sqlite:///{(output_dir / 'study.db').as_posix()}",
        load_if_exists=True,
    )
    if not study.trials:
        for config in baselines:
            study.enqueue_trial(mannstein_trial_params(config))

    def objective(trial: BaseTrial) -> float:
        metrics = evaluate_annotator_masks(search, suggest_mannstein_config(trial), args.workers)
        for key in ("f1", "precision", "recall", "mean_pred_instances"):
            trial.set_user_attr(key, metrics[key])
        return metrics["map"]

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best = suggest_mannstein_config(optuna.trial.FixedTrial(study.best_trial.params))
    write_best_yaml(best, output_dir / "best_mannstein.yaml")
    candidates = {f"baseline_{i}": c for i, c in enumerate(baselines)} | {"best": best}
    summary: dict[str, Any] = {
        "seed": args.seed,
        "n_search_samples": len(search),
        "n_holdout_samples": len(holdout),
        "n_trials_completed": sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        ),
        "baseline_configs": list(args.baseline_configs),
        "search_map": {"best": study.best_value},
        "holdout": {
            name: evaluate_annotator_masks(holdout, config, args.workers, per_threshold=True)
            for name, config in candidates.items()
        },
        "best_config": best.model_dump(),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ("search_map", "holdout")}, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main(parse_args())
