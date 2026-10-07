"""Standalone evaluation of a saved DeepContrail checkpoint.

Loads a checkpoint and experiment config, runs evaluation on val and/or test
with threshold search, and optionally logs results to MLflow.

Usage
-----
::

    # Re-evaluate a checkpoint on val+test
    uv run python scripts/evaluate.py \
        --checkpoint path/to/best.ckpt \
        --config configs/experiment.yaml

    # Skip test, use a fixed threshold
    uv run python scripts/evaluate.py \
        --checkpoint path/to/best.ckpt \
        --config configs/experiment.yaml \
        --skip-test --threshold 0.31

    # Log to an existing MLflow run
    uv run python scripts/evaluate.py \
        --checkpoint path/to/best.ckpt \
        --config configs/experiment.yaml \
        --run-id abc123def456
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch.utils.data import DataLoader

from deepcontrail.data import DataConfig, GOESSemanticDataset
from deepcontrail.losses import build_loss
from deepcontrail.models import build_model
from deepcontrail.training import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    ContrailSegmentationModule,
    evaluate_global,
    load_configs,
)

logger = logging.getLogger(__name__)


def _build_eval_loader(data_cfg: DataConfig) -> DataLoader[Any]:
    """Build a DataLoader for evaluation (no shuffle, no drop_last)."""
    ds = GOESSemanticDataset(data_cfg)
    return DataLoader(
        ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def main(args: argparse.Namespace) -> None:
    """Load checkpoint, evaluate, print results, optionally log to MLflow."""
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    eval_args = argparse.Namespace(
        epochs=None,
        lr=None,
        batch_size=None,
        image_size=None,
        encoder=None,
        architecture=None,
        patience=None,
        warmup_epochs=None,
        scheduler_interval=None,
    )
    _, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, _seed = load_configs(
        config_path, eval_args
    )

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    # Rebuild model + loss, load checkpoint
    model = build_model(model_cfg)
    loss_fn = build_loss(loss_cfg)
    module = ContrailSegmentationModule.load_from_checkpoint(
        str(checkpoint_path),
        model=model,
        loss_fn=loss_fn,
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
        scheduler=training_cfg.scheduler,
        epochs=training_cfg.epochs,
    )
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # Val evaluation
    val_loader = _build_eval_loader(val_data_cfg)
    logger.info("Evaluating on val set (%d samples)...", len(val_loader.dataset))  # type: ignore[arg-type]

    val_threshold = args.threshold
    val_metrics = evaluate_global(module, val_loader, device, prefix="val", threshold=val_threshold)

    optimal_threshold = val_metrics["final/val_threshold"]
    logger.info("Val threshold: %.4f", optimal_threshold)

    # Test evaluation
    test_metrics: dict[str, float] = {}
    if not args.skip_test:
        test_loader = _build_eval_loader(test_data_cfg)
        logger.info(
            "Evaluating on test set (%d samples, threshold=%.4f)...",
            len(test_loader.dataset),  # type: ignore[arg-type]
            optimal_threshold,
        )
        test_threshold = args.threshold if args.threshold is not None else optimal_threshold
        test_metrics = evaluate_global(
            module, test_loader, device, prefix="test", threshold=test_threshold
        )

    all_metrics = {**val_metrics, **test_metrics}

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint:           {checkpoint_path}")
    print(f"Config:               {config_path}")
    for key, value in all_metrics.items():
        print(f"{key:30s} {value:.4f}")
    print("=" * 60)

    # MLflow logging
    if args.run_id or args.run_name:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        eval_metrics = {f"eval/{k}": v for k, v in all_metrics.items()}

        if args.run_id:
            with mlflow.start_run(run_id=args.run_id):
                mlflow.log_metrics(eval_metrics)
            logger.info("Logged eval metrics to existing run %s", args.run_id)
        else:
            with mlflow.start_run(run_name=args.run_name):
                mlflow.log_metrics(eval_metrics)
                mlflow.log_param("eval/checkpoint", str(checkpoint_path))
                mlflow.log_param("eval/config", str(config_path))
                mlflow.log_artifact(str(config_path))
            logger.info("Logged eval metrics to new run '%s'", args.run_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved DeepContrail checkpoint without retraining."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip evaluation on the test set.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold (skip search on val).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Append eval metrics to an existing MLflow run.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Create a new MLflow run with this name for eval metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Default: cuda if available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
