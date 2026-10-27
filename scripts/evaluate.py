"""Evaluate trained DeepContrail checkpoints (single model or ensemble).

Supports two modes:

- **Single-model** (``--checkpoint``): loads one checkpoint and evaluates on
  val and/or test with threshold search, optionally logging to MLflow.
- **Ensemble** (``--checkpoints``): loads one or more per-fold checkpoints,
  averages their sigmoid probabilities pixel-wise, and reports metrics via a
  threshold search.

Usage
-----
::

    # Single-model evaluation (backward compatible)
    uv run python scripts/evaluate.py \
        --checkpoint path/to/best.ckpt \
        --config configs/experiment.yaml

    # Ensemble evaluation (4 folds)
    uv run python scripts/evaluate.py \
        --config configs/experiment.yaml \
        --checkpoints ckpt_fold0.ckpt ckpt_fold1.ckpt ckpt_fold2.ckpt ckpt_fold3.ckpt

    # Apply a fixed threshold to the test split
    uv run python scripts/evaluate.py \
        --config configs/experiment.yaml \
        --checkpoints ckpt_fold0.ckpt ckpt_fold1.ckpt ckpt_fold2.ckpt ckpt_fold3.ckpt \
        --split test --threshold 0.42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from deepcontrail.segmentation import GOESSemanticDataset, SegmentationDataConfig
from deepcontrail.segmentation.losses import build_loss
from deepcontrail.segmentation.models import build_model
from deepcontrail.segmentation.training import (
    ContrailSegmentationModule,
    evaluate_ensemble,
    evaluate_global,
    load_configs,
)

logger = logging.getLogger(__name__)


def _build_loader(cfg: SegmentationDataConfig) -> DataLoader[Any]:
    """Build a no-augmentation ``DataLoader`` for the given ``SegmentationDataConfig``."""
    ds = GOESSemanticDataset(cfg)
    return DataLoader(
        ds,  # type: ignore[arg-type]
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _eval_config(val_data_cfg: SegmentationDataConfig, split: str) -> SegmentationDataConfig:
    """Build a no-augmentation eval config for the requested split."""
    no_aug = {**val_data_cfg.model_dump(), "augmentation": None}
    if split == "val":
        return SegmentationDataConfig.from_dict(no_aug)
    return SegmentationDataConfig.from_dict({**no_aug, "split": "test", "fold": None})


# ---------------------------------------------------------------------------
# Single-model evaluation (backward compatible path)
# ---------------------------------------------------------------------------


def _run_single(args: argparse.Namespace) -> None:
    """Evaluate a single checkpoint on val and/or test."""
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, _seed = load_configs(
        config_path, args
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        warmup_epochs=training_cfg.warmup_epochs,
        scheduler_interval=training_cfg.scheduler_interval,
    )
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    val_eval_cfg = _eval_config(val_data_cfg, "val")
    val_loader = _build_loader(val_eval_cfg)
    logger.info("Evaluating on val set (%d samples)...", len(val_loader.dataset))  # type: ignore[arg-type]

    val_threshold = args.threshold
    val_metrics = evaluate_global(module, val_loader, device, prefix="val", threshold=val_threshold)
    optimal_threshold = val_metrics["final/val_threshold"]

    val_tta_metrics: dict[str, float] = {}
    if args.tta:
        logger.info("Evaluating on val set with D4 TTA...")
        val_tta_metrics = evaluate_global(module, val_loader, device, prefix="val_tta", tta=True)

    test_metrics: dict[str, float] = {}
    test_tta_metrics: dict[str, float] = {}
    if not args.skip_test:
        test_eval_cfg = _eval_config(test_data_cfg, "test")
        test_loader = _build_loader(test_eval_cfg)
        test_threshold = args.threshold if args.threshold is not None else optimal_threshold
        logger.info(
            "Evaluating on test set (%d samples, threshold=%.4f)...",
            len(test_loader.dataset),  # type: ignore[arg-type]
            test_threshold,
        )
        test_metrics = evaluate_global(
            module, test_loader, device, prefix="test", threshold=test_threshold
        )
        if args.tta:
            tta_threshold = val_tta_metrics.get("final/val_tta_threshold", test_threshold)
            logger.info("Evaluating on test set with D4 TTA...")
            test_tta_metrics = evaluate_global(
                module,
                test_loader,
                device,
                prefix="test_tta",
                threshold=tta_threshold,
                tta=True,
            )

    all_metrics = {**val_metrics, **val_tta_metrics, **test_metrics, **test_tta_metrics}

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint:           {checkpoint_path}")
    print(f"Config:               {config_path}")
    for key, value in all_metrics.items():
        print(f"{key:30s} {value:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------


def _run_ensemble(args: argparse.Namespace) -> None:
    """Evaluate an ensemble of checkpoints on a single split."""
    config_path = Path(args.config)
    _, val_data_cfg, _, model_cfg, loss_cfg, training_cfg, _seed = load_configs(config_path, args)

    eval_cfg = _eval_config(val_data_cfg, args.split)
    loader = _build_loader(eval_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Evaluating ensemble of %d checkpoint(s) on the %s split (%d samples)...",
        len(args.checkpoints),
        args.split,
        len(loader.dataset),  # type: ignore[arg-type]
    )

    metrics = evaluate_ensemble(
        checkpoints=args.checkpoints,
        model_config=model_cfg,
        loss_config=loss_cfg,
        training_config=training_cfg,
        data_loader=loader,
        device=device,
        steps_per_epoch=len(loader),
        prefix=args.split,
        threshold=args.threshold,
    )

    tta_metrics: dict[str, float] = {}
    if args.tta:
        logger.info("Evaluating ensemble with D4 TTA...")
        tta_metrics = evaluate_ensemble(
            checkpoints=args.checkpoints,
            model_config=model_cfg,
            loss_config=loss_cfg,
            training_config=training_cfg,
            data_loader=loader,
            device=device,
            steps_per_epoch=len(loader),
            prefix=f"{args.split}_tta",
            threshold=args.threshold,
            tta=True,
        )

    all_metrics = {**metrics, **tta_metrics}

    print("\n" + "=" * 60)
    print(f"ENSEMBLE EVALUATION ({args.split})")
    print("=" * 60)
    for key, value in all_metrics.items():
        print(f"{key:32s} {value:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Dispatch to single-model or ensemble evaluation."""
    if args.checkpoint:
        _run_single(args)
    else:
        _run_ensemble(args)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeepContrail checkpoints (single model or ensemble)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to the experiment YAML config (default: configs/experiment.yaml).",
    )

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a single .ckpt file (single-model evaluation).",
    )
    ckpt_group.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="Paths to per-fold checkpoints (ensemble evaluation).",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split to evaluate on (default: val). Only used with --checkpoints.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold to apply (default: auto-search the optimal threshold).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test-set evaluation. Only used with --checkpoint.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Apply D4 test-time augmentation (8 geometric transforms, ~8x slower).",
    )

    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size.")
    parser.add_argument("--fold", type=int, default=None, help="CV fold override.")
    parser.add_argument("--num-folds", type=int, default=None, help="Total number of CV folds.")
    parser.add_argument("--encoder", type=str, default=None, help="Override encoder name.")
    parser.add_argument(
        "--architecture", type=str, default=None, help="Override model architecture."
    )
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Number of warmup epochs.")
    parser.add_argument(
        "--scheduler-interval",
        type=str,
        default=None,
        choices=["epoch", "step"],
        help="LR scheduler step interval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
