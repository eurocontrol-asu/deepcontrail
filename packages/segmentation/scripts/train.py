"""Training and MLflow tracking for DeepContrail segmentation.

Trains a segmentation model on the GOES-16 semantic dataset, evaluates on
the validation split, and logs all parameters, per-epoch metrics, and the
best checkpoint to MLflow.

Usage
-----
::

    # Default config
    uv run python packages/segmentation/scripts/train.py

    # Explicit config path
    uv run python packages/segmentation/scripts/train.py \
        --config packages/segmentation/configs/experiment.yaml

    # CLI overrides
    uv run python packages/segmentation/scripts/train.py --epochs 10 --lr 1e-3

    # Smoke test with a small encoder
    uv run python packages/segmentation/scripts/train.py \
        --encoder resnet18 --epochs 2 --batch-size 4
"""

from __future__ import annotations

import argparse
import datetime
import logging
import platform
import time
from pathlib import Path
from typing import Any

import mlflow
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from deepcontrail.segmentation import GOESSemanticDataset, SegmentationDataConfig
from deepcontrail.segmentation.losses import LossConfig, build_loss
from deepcontrail.segmentation.models import ModelConfig, build_model
from deepcontrail.segmentation.training import (
    CHECKPOINT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    ContrailSegmentationModule,
    TrainingConfig,
    evaluate_global,
    load_configs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


def build_dataloaders(
    train_cfg: SegmentationDataConfig, val_cfg: SegmentationDataConfig
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Construct train and validation ``DataLoader`` instances.

    Parameters
    ----------
    train_cfg
        Data configuration for the training split (with augmentation).
    val_cfg
        Data configuration for the validation split (no augmentation).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``.
    """
    train_ds = GOESSemanticDataset(train_cfg)
    val_ds = GOESSemanticDataset(val_cfg)

    persistent = train_cfg.num_workers > 0

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,
        batch_size=val_cfg.batch_size,
        shuffle=False,
        num_workers=val_cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------


def _flatten_configs(
    train_data_cfg: SegmentationDataConfig,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    training_cfg: TrainingConfig,
    seed: int,
) -> dict[str, Any]:
    """Flatten all configs into a single param dict for MLflow.

    Parameters
    ----------
    train_data_cfg
        Training data configuration.
    model_cfg
        Model architecture configuration.
    loss_cfg
        Loss function configuration.
    training_cfg
        Training hyper-parameters.
    seed
        Global random seed.

    Returns
    -------
    dict[str, Any]
        Flattened parameters keyed by ``{section}/{field}``.
    """
    params: dict[str, Any] = {}

    for prefix, cfg in [
        ("data", train_data_cfg),
        ("model", model_cfg),
        ("loss", loss_cfg),
        ("training", training_cfg),
    ]:
        for key, value in cfg.model_dump().items():
            params[f"{prefix}/{key}"] = value

    params["seed"] = seed
    return params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_run_name(model_cfg: ModelConfig) -> str:
    """Generate a descriptive MLflow run name.

    Parameters
    ----------
    model_cfg
        Model configuration.

    Returns
    -------
    str
        Run name in the format ``{architecture}-{encoder}-{timestamp}``.
    """
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{model_cfg.architecture}-{model_cfg.encoder_name}-{ts}"


def main(args: argparse.Namespace) -> None:
    """Orchestrate training, evaluation, and MLflow logging.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    """
    config_path = Path(args.config)
    train_data_cfg, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, seed = (
        load_configs(config_path, args)
    )

    # Reproducibility
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Data
    logger.info("Building data loaders...")
    train_loader, val_loader = build_dataloaders(train_data_cfg, val_data_cfg)

    # Masks are returned at the dataset's native resolution; the module resizes
    # its output back to that size so loss/metric computation runs at mask
    # resolution even when images are upscaled (image_size > native_size).
    native_size = train_loader.dataset.native_size

    logger.info(
        "Train: %d samples, Val: %d samples",
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),  # type: ignore[arg-type]
    )

    # Model + loss
    model = build_model(model_cfg)
    loss_fn = build_loss(loss_cfg)
    module = ContrailSegmentationModule(
        model=model,
        loss_fn=loss_fn,
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
        scheduler=training_cfg.scheduler,
        epochs=training_cfg.epochs,
        warmup_epochs=training_cfg.warmup_epochs,
        scheduler_interval=training_cfg.scheduler_interval,
        steps_per_epoch=len(train_loader),
        native_size=native_size,
    )

    # Run name
    run_name = args.run_name if args.run_name else make_run_name(model_cfg)

    # MLflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=run_name,
        log_model=False,
    )

    # Checkpoint callback
    ckpt_dir = CHECKPOINT_ROOT / run_name
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    # Callbacks
    callbacks: list[pl.Callback] = [checkpoint_cb]

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=training_cfg.epochs,
        precision=training_cfg.precision,
        callbacks=callbacks,
        default_root_dir=str(CHECKPOINT_ROOT),
        logger=mlf_logger,
        deterministic=False,
        gradient_clip_val=training_cfg.gradient_clip_val,
    )

    # Bridge the MLFlowLogger's run with the global mlflow API so we can
    # use mlflow.log_params / log_metrics / log_artifact on the same run.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _ = mlf_logger.experiment  # force run creation
    run_id = mlf_logger.run_id

    with mlflow.start_run(run_id=run_id):
        params = _flatten_configs(train_data_cfg, model_cfg, loss_cfg, training_cfg, seed)
        mlflow.log_params(params)

        # System info
        system_params: dict[str, Any] = {
            "system/torch_version": torch.__version__,
            "system/smp_version": smp.__version__,
            "system/cuda_version": torch.version.cuda or "N/A",
            "system/python_version": platform.python_version(),
        }
        if torch.cuda.is_available():
            system_params["system/gpu_name"] = torch.cuda.get_device_name(0)
        mlflow.log_params(system_params)

        # Dataset sample counts
        test_ds_for_count = GOESSemanticDataset(test_data_cfg) if not args.skip_test else None
        mlflow.log_param("data/train_samples", len(train_loader.dataset))  # type: ignore[arg-type]
        mlflow.log_param("data/val_samples", len(val_loader.dataset))  # type: ignore[arg-type]
        mlflow.log_param("data/test_samples", len(test_ds_for_count) if test_ds_for_count else 0)

        # Train
        logger.info("Starting training for %d epochs...", training_cfg.epochs)
        t0 = time.monotonic()
        trainer.fit(module, train_loader, val_loader)
        wall_seconds = time.monotonic() - t0
        mlflow.log_metric("training_wall_seconds", wall_seconds)

        # Post-training global evaluation
        best_ckpt_path = checkpoint_cb.best_model_path
        if best_ckpt_path:
            logger.info("Loading best checkpoint: %s", best_ckpt_path)
            best_module = ContrailSegmentationModule.load_from_checkpoint(
                best_ckpt_path,
                model=build_model(model_cfg),
                loss_fn=build_loss(loss_cfg),
                lr=training_cfg.lr,
                weight_decay=training_cfg.weight_decay,
                scheduler=training_cfg.scheduler,
                epochs=training_cfg.epochs,
                warmup_epochs=training_cfg.warmup_epochs,
                scheduler_interval=training_cfg.scheduler_interval,
                steps_per_epoch=len(train_loader),
                native_size=native_size,
            )
        else:
            best_module = module

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Global evaluation on val (with threshold search)
        logger.info("Evaluating on val set with threshold search...")
        val_metrics = evaluate_global(best_module, val_loader, device, prefix="val")
        mlflow.log_metrics(val_metrics)
        val_threshold = val_metrics["final/val_threshold"]
        logger.info("Optimal val threshold: %.2f", val_threshold)

        # Val with D4 TTA
        logger.info("Evaluating on val set with D4 TTA...")
        val_tta_metrics = evaluate_global(
            best_module, val_loader, device, prefix="val_tta", tta=True
        )
        mlflow.log_metrics(val_tta_metrics)
        val_tta_threshold = val_tta_metrics["final/val_tta_threshold"]

        # Global evaluation on test (apply val threshold — no search)
        test_metrics: dict[str, float] = {}
        test_tta_metrics: dict[str, float] = {}
        if not args.skip_test:
            logger.info("Evaluating on test set (threshold=%.2f from val)...", val_threshold)
            test_ds = test_ds_for_count or GOESSemanticDataset(test_data_cfg)
            test_loader: DataLoader[Any] = DataLoader(
                test_ds,
                batch_size=test_data_cfg.batch_size,
                shuffle=False,
                num_workers=test_data_cfg.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            test_metrics = evaluate_global(
                best_module, test_loader, device, prefix="test", threshold=val_threshold
            )
            mlflow.log_metrics(test_metrics)

            logger.info("Evaluating on test set with D4 TTA...")
            test_tta_metrics = evaluate_global(
                best_module,
                test_loader,
                device,
                prefix="test_tta",
                threshold=val_tta_threshold,
                tta=True,
            )
            mlflow.log_metrics(test_tta_metrics)

        if checkpoint_cb.best_model_score is not None:
            mlflow.log_metric("best_epoch_val_dice", float(checkpoint_cb.best_model_score))

        mlflow.log_artifact(str(config_path))
        for split in ["train", "val", "test"]:
            split_file = Path(train_data_cfg.root) / f"splits_{split}.txt"
            if split_file.exists():
                mlflow.log_artifact(str(split_file))
        if best_ckpt_path:
            mlflow.log_artifact(best_ckpt_path)

    # Summary
    all_metrics = {**val_metrics, **val_tta_metrics, **test_metrics, **test_tta_metrics}
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run name:             {run_name}")
    print(f"Best checkpoint:      {best_ckpt_path}")
    for key, value in all_metrics.items():
        print(f"{key:30s} {value:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a DeepContrail segmentation model with MLflow tracking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to the experiment YAML config (default: the package's configs/experiment.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size.")
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="K-fold CV: hold out this fold for validation (enables cross-validation).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Total number of CV folds (only used with --fold).",
    )
    parser.add_argument("--encoder", type=str, default=None, help="Override encoder name.")
    parser.add_argument(
        "--architecture", type=str, default=None, help="Override model architecture."
    )
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name override.")
    parser.add_argument(
        "--skip-test", action="store_true", help="Skip evaluation on the holdout test set."
    )
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Number of warmup epochs.")
    parser.add_argument(
        "--scheduler-interval",
        type=str,
        default=None,
        choices=["epoch", "step"],
        help="LR scheduler step interval ('epoch' or 'step').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
