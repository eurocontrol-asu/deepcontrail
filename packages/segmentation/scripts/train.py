"""Training and MLflow tracking for DeepContrail segmentation.

Trains a segmentation model on a DeepContrail-format dataset, evaluates on
the validation split, and logs all parameters, per-epoch metrics, and the
best checkpoint to MLflow.

Where runs are written has no default. Set it once in a ``.env`` file at the
repository root (``cp .env.example .env``, then edit the paths), or export the
variables, or pass ``--checkpoint-root`` / ``--mlflow-uri``. A flag wins over a
variable set in the shell, which wins over ``.env``::

    DEEPCONTRAIL_CHECKPOINT_ROOT=/path/to/checkpoints
    MLFLOW_TRACKING_URI=sqlite:////path/to/mlflow.db

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
import logging
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import segmentation_models_pytorch as smp
import timm
import torch
from dotenv import find_dotenv, load_dotenv
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from deepcontrail.core.contract import (
    BAND_MANIFEST_PATH,
    DATASET_INFO_PATH,
    read_dataset_info,
)
from deepcontrail.segmentation import SegmentationDataConfig, SegmentationDataset
from deepcontrail.segmentation.losses import LossConfig, build_loss
from deepcontrail.segmentation.models import ModelConfig, build_model
from deepcontrail.segmentation.training import (
    SegmentationModule,
    TrainingConfig,
    evaluate_global,
    prepare_training,
    resolve_run_paths,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT_NAME = "deepcontrail-segmentation"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"


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
# Training curves
# ---------------------------------------------------------------------------

# Categorical slots 1 and 2 of the validated default palette, in fixed order:
# blue = train / val Dice, orange = val loss / val IoU. Ink and grid stay neutral.
_BLUE, _BLUE_LIGHT, _ORANGE = "#2a78d6", "#86b6ef", "#eb6834"
_INK, _INK_MUTED, _GRID, _SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def plot_training_curves(
    history: dict[str, list[tuple[int, float]]], steps_per_epoch: int, best_epoch: int | None
) -> Figure:
    """Draw loss, validation scores and learning rate against epochs, one panel each.

    Parameters
    ----------
    history
        Metric name -> ``(global_step, value)`` points, as logged by Lightning
        (``train_loss``, ``val_loss``, ``val_dice``, ``val_iou``, ``train_loss_step``,
        ``lr-<optimizer>``); missing series are skipped.
    steps_per_epoch
        Optimizer steps per epoch, to turn global steps into epochs.
    best_epoch
        Epoch of the best checkpoint (1-based), marked on the scores panel.

    Returns
    -------
    Figure
        Three panels: ``Loss``, ``Validation Dice / IoU``, ``Learning rate``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def points(key: str) -> tuple[list[float], list[float]]:
        pts = sorted(history.get(key, []))
        return [(step + 1) / steps_per_epoch for step, _ in pts], [v for _, v in pts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    fig.patch.set_facecolor(_SURFACE)
    loss_ax, score_ax, lr_ax = axes

    def draw(ax: Any, key: str, label: str, color: str, **style: Any) -> None:  # noqa: ANN401
        x, y = points(key)
        if x:
            ax.plot(x, y, label=label, color=color, linewidth=style.pop("linewidth", 2), **style)

    draw(loss_ax, "train_loss_step", "train loss (per step)", _BLUE_LIGHT, linewidth=1, alpha=0.6)
    draw(loss_ax, "train_loss", "train loss", _BLUE, marker="o", markersize=4)
    draw(loss_ax, "val_loss", "val loss", _ORANGE, marker="o", markersize=4)
    draw(score_ax, "val_dice", "val Dice", _BLUE, marker="o", markersize=4)
    draw(score_ax, "val_iou", "val IoU", _ORANGE, marker="o", markersize=4)
    lr_key = next((k for k in history if k.startswith("lr-")), None)
    if lr_key:
        draw(lr_ax, lr_key, "learning rate", _BLUE)
        lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if best_epoch is not None:
        score_ax.axvline(
            best_epoch, color=_INK_MUTED, linestyle="--", linewidth=1, label="best epoch"
        )

    for ax, title, ylabel in (
        (loss_ax, "Loss", "loss"),
        (score_ax, "Validation Dice / IoU", "score"),
        (lr_ax, "Learning rate", "learning rate"),
    ):
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, color=_INK, loc="left")
        ax.set_xlabel("epoch", color=_INK_MUTED)
        ax.set_ylabel(ylabel, color=_INK_MUTED)
        ax.grid(color=_GRID, linewidth=0.8)
        ax.tick_params(colors=_INK_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        if len(ax.get_lines()) >= 2:
            ax.legend(frameon=False, labelcolor=_INK_MUTED)
    return fig


def _git_dirty(repo: Path) -> str:
    """Return ``"true"``/``"false"`` for uncommitted changes in ``repo``, else ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return "true" if out.strip() else "false"


def _metric_history(client: Any, run_id: str) -> dict[str, list[tuple[int, float]]]:  # noqa: ANN401
    """Return every metric of the run as ``(step, value)`` points."""
    keys = client.get_run(run_id).data.metrics
    return {k: [(m.step, m.value) for m in client.get_metric_history(run_id, k)] for k in keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    """Orchestrate training, evaluation, and MLflow logging.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    """
    # Fail before any work if the run has nowhere to write. The nearest .env (searched
    # from the working directory upwards) fills in variables the shell does not set;
    # CLI flags win over both.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    checkpoint_root, mlflow_uri = resolve_run_paths(args.checkpoint_root, args.mlflow_uri)
    config_path = Path(args.config)

    # Data, model, loss, checkpointing and Lightning's Trainer come from the library:
    # the same run as `deepcontrail-train --logger mlflow`, plus the experiment
    # tooling below (MLflow params and artifacts, threshold search, TTA, test).
    run = prepare_training(
        config_path,
        output_dir=checkpoint_root,
        run_name=args.run_name,
        logger="mlflow",
        overrides=args,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
        mlflow_uri=mlflow_uri,
        accelerator="gpu",
    )
    train_data_cfg, test_data_cfg = run.train_data_cfg, run.test_data_cfg
    model_cfg, loss_cfg, training_cfg = run.model_cfg, run.loss_cfg, run.training_cfg
    seed = run.seed
    train_loader, val_loader = run.train_loader, run.val_loader
    native_size = run.module.native_size
    run_name = run.run_dir.name
    checkpoint_cb = run.checkpoint
    logger.info(
        "Train: %d samples, Val: %d samples",
        len(train_loader.dataset),  # type: ignore[arg-type]
        len(val_loader.dataset),  # type: ignore[arg-type]
    )

    # Bridge the MLFlowLogger's run with the global mlflow API so we can
    # use mlflow.log_params / log_metrics / log_artifact on the same run.
    mlf_logger = run.trainer.logger
    assert isinstance(mlf_logger, MLFlowLogger)
    mlflow.set_tracking_uri(mlflow_uri)
    _ = mlf_logger.experiment  # force run creation
    run_id = mlf_logger.run_id
    assert run_id is not None, "the MLflow run must exist before logging to it"

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
        system_params |= {
            "system/lightning_version": version("pytorch-lightning"),
            "system/timm_version": timm.__version__,
            "system/deepcontrail_version": version("deepcontrail-segmentation"),
        }
        mlflow.log_params(system_params)
        mlflow.set_tag("git.dirty", _git_dirty(Path(__file__).resolve().parent))

        # Model size and the dataset the run read (its format metadata as artifacts)
        mlflow.log_param("model/num_params", sum(p.numel() for p in run.module.model.parameters()))
        mlflow.log_param("training/steps_per_epoch", len(train_loader))
        info = read_dataset_info(train_data_cfg.root)
        mlflow.log_params(
            {
                "data/export": info.export,
                "data/source": info.source,
                "data/format_version": info.format_version,
            }
        )
        for relative in (DATASET_INFO_PATH, BAND_MANIFEST_PATH):
            mlflow.log_artifact(str(Path(train_data_cfg.root) / relative), artifact_path="dataset")

        # Dataset sample counts
        test_ds_for_count = SegmentationDataset(test_data_cfg) if not args.skip_test else None
        mlflow.log_param("data/train_samples", len(train_loader.dataset))  # type: ignore[arg-type]
        mlflow.log_param("data/val_samples", len(val_loader.dataset))  # type: ignore[arg-type]
        mlflow.log_param("data/test_samples", len(test_ds_for_count) if test_ds_for_count else 0)

        # Train
        logger.info("Starting training for %d epochs...", training_cfg.epochs)
        t0 = time.monotonic()
        run.fit()
        wall_seconds = time.monotonic() - t0
        mlflow.log_metric("training_wall_seconds", wall_seconds)
        if torch.cuda.is_available():
            mlflow.log_metric("peak_gpu_memory_gb", torch.cuda.max_memory_allocated() / 1e9)

        # Training curves against epochs, and the epoch of the best checkpoint
        history = _metric_history(mlflow.MlflowClient(), run_id)
        val_dice = [v for _, v in sorted(history.get("val_dice", []))]
        best_epoch = val_dice.index(max(val_dice)) + 1 if val_dice else None
        if best_epoch is not None:
            mlflow.log_metric("best_epoch", best_epoch)
        figure = plot_training_curves(history, len(train_loader), best_epoch)
        mlflow.log_figure(figure, "training_curves.png")

        # Post-training global evaluation
        best_ckpt_path = checkpoint_cb.best_model_path
        if best_ckpt_path:
            logger.info("Loading best checkpoint: %s", best_ckpt_path)
            best_module = SegmentationModule.load_from_checkpoint(
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
            best_module = run.module

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
            test_ds = test_ds_for_count or SegmentationDataset(test_data_cfg)
            test_loader: DataLoader[Any] = DataLoader(
                test_ds,  # type: ignore[arg-type]  # map-style dataset
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
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root overriding data.root of the config.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Directory for checkpoints (default: $DEEPCONTRAIL_CHECKPOINT_ROOT, or .env).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI, e.g. sqlite:////abs/path/mlflow.db "
        "(default: $MLFLOW_TRACKING_URI, or .env).",
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
