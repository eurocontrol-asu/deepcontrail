"""Training configuration and constants."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import argparse

import yaml
from pydantic import BaseModel, ConfigDict, Field

from deepcontrail.data import DataConfig
from deepcontrail.losses import LossConfig
from deepcontrail.models import ModelConfig

CHECKPOINT_ROOT = Path("/data/aheffar/deepcontrail_checkpoints")
MLFLOW_TRACKING_URI = "sqlite:///data/aheffar/mlruns/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "deepcontrail-segmentation"


class TrainingConfig(BaseModel):
    """Training hyper-parameters.

    Parameters
    ----------
    epochs
        Number of training epochs.
    lr
        Learning rate for the optimizer.
    optimizer
        Optimizer name (``"AdamW"`` or ``"Adam"``).
    scheduler
        Learning rate scheduler name.
    weight_decay
        Weight decay for the optimizer.
    precision
        Mixed-precision mode for Lightning (e.g. ``"16-mixed"``).
    gradient_clip_val
        Gradient clipping value, or ``None`` to disable.
    """

    model_config = ConfigDict(frozen=True)

    epochs: int = Field(default=50, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    warmup_epochs: int = Field(default=0, ge=0)
    scheduler_interval: Literal["epoch", "step"] = "epoch"
    weight_decay: float = Field(default=1e-4, ge=0)
    precision: str = "16-mixed"
    gradient_clip_val: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Build from a plain dictionary."""
        return cls.model_validate(d)


def load_configs(
    config_path: Path, args: argparse.Namespace
) -> tuple[DataConfig, DataConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig, int]:
    """Load experiment config from YAML and apply CLI overrides.

    Parameters
    ----------
    config_path
        Path to the experiment YAML file.
    args
        Parsed CLI arguments (may contain override values).

    Returns
    -------
    tuple
        ``(train_data_cfg, val_data_cfg, test_data_cfg, model_cfg,
        loss_cfg, training_cfg, seed)``.
    """
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    data_raw: dict[str, Any] = dict(raw["data"])
    model_raw: dict[str, Any] = dict(raw["model"])
    loss_raw: dict[str, Any] = dict(raw.get("loss", {}))
    training_raw: dict[str, Any] = dict(raw.get("training", {}))

    if args.epochs is not None:
        training_raw["epochs"] = args.epochs
    if args.lr is not None:
        training_raw["lr"] = args.lr
    if args.batch_size is not None:
        data_raw["batch_size"] = args.batch_size
    if args.image_size is not None:
        data_raw["image_size"] = args.image_size
    if args.fold is not None:
        data_raw["fold"] = args.fold
    if args.num_folds is not None:
        data_raw["num_folds"] = args.num_folds
    if args.encoder is not None:
        model_raw["encoder_name"] = args.encoder
    if args.architecture is not None:
        model_raw["architecture"] = args.architecture
    if args.warmup_epochs is not None:
        training_raw["warmup_epochs"] = args.warmup_epochs
    if args.scheduler_interval is not None:
        training_raw["scheduler_interval"] = args.scheduler_interval

    train_data_cfg = DataConfig.from_dict(data_raw)
    no_aug = {**train_data_cfg.model_dump(), "augmentation": None}
    # Evaluation always uses the strict consensus ground truth ("binary").
    val_data_cfg = DataConfig.from_dict({**no_aug, "split": "val", "mask_type": "binary"})
    test_data_cfg = DataConfig.from_dict(
        {**no_aug, "split": "test", "fold": None, "mask_type": "binary"}
    )
    model_cfg = ModelConfig.from_dict(model_raw)
    loss_cfg = LossConfig.from_dict(loss_raw)
    training_cfg = TrainingConfig.from_dict(training_raw)

    seed = train_data_cfg.seed

    return train_data_cfg, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, seed
