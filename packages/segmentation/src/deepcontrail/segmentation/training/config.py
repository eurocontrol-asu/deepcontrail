"""Training configuration and constants."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepcontrail.segmentation import SegmentationDataConfig
from deepcontrail.segmentation.losses import LossConfig
from deepcontrail.segmentation.models import ModelConfig

CHECKPOINT_ROOT_ENV = "DEEPCONTRAIL_CHECKPOINT_ROOT"
MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"


def resolve_run_paths(
    checkpoint_root: str | Path | None,
    mlflow_uri: str | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, str]:
    """Return the checkpoint root and MLflow tracking URI of a training run.

    Each value comes from the explicit argument (a CLI flag), else from the
    environment (``DEEPCONTRAIL_CHECKPOINT_ROOT``, ``MLFLOW_TRACKING_URI``). There is
    no built-in default, so a run never logs to an unexpected place.

    Raises
    ------
    ValueError
        If a value is missing, or if the URI is a relative ``sqlite:///`` path (it
        would resolve against the working directory; use ``sqlite:////abs/path``).
    """
    env = os.environ if environ is None else environ
    root = checkpoint_root if checkpoint_root is not None else env.get(CHECKPOINT_ROOT_ENV)
    uri = mlflow_uri if mlflow_uri is not None else env.get(MLFLOW_TRACKING_URI_ENV)
    if not root:
        msg = f"no checkpoint root: pass --checkpoint-root or set {CHECKPOINT_ROOT_ENV}"
        raise ValueError(msg)
    if not uri:
        msg = f"no MLflow tracking URI: pass --mlflow-uri or set {MLFLOW_TRACKING_URI_ENV}"
        raise ValueError(msg)
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        msg = (
            f"{uri} is a relative sqlite path (it depends on the working directory); "
            "use an absolute one: sqlite:////abs/path/mlflow.db"
        )
        raise ValueError(msg)
    return Path(root), uri


class TrainingConfig(BaseModel):
    """Training hyper-parameters.

    Parameters
    ----------
    epochs
        Number of training epochs.
    lr
        Learning rate for the optimizer.
    optimizer
        ``"AdamW"`` (decoupled weight decay) or ``"Adam"`` (weight decay as an L2 penalty). The
        two are identical when ``weight_decay`` is 0.
    scheduler
        Learning rate schedule; ``"CosineAnnealingLR"`` is the one implemented.
    warmup_epochs
        Epochs of linear warm-up before the cosine decay; fewer than ``epochs``.
    scheduler_interval
        Whether the schedule advances every ``"epoch"`` or every optimizer ``"step"``.
    weight_decay
        Weight decay for the optimizer.
    precision
        Mixed-precision mode for Lightning (e.g. ``"16-mixed"``).
    gradient_clip_val
        Gradient clipping value, or ``None`` to disable.
    logger
        Where Lightning logs metrics: ``"csv"`` (default, no extra package) or
        ``"mlflow"`` (extra ``deepcontrail-segmentation[mlflow]``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(default=50, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    optimizer: Literal["AdamW", "Adam"] = "AdamW"
    scheduler: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    warmup_epochs: int = Field(default=0, ge=0)
    scheduler_interval: Literal["epoch", "step"] = "epoch"
    weight_decay: float = Field(default=1e-4, ge=0)
    precision: str = "16-mixed"
    gradient_clip_val: float | None = None
    logger: Literal["csv", "mlflow"] = "csv"

    @model_validator(mode="after")
    def _check_warmup(self) -> TrainingConfig:
        """Warm-up must leave epochs for the cosine decay."""
        if self.warmup_epochs >= self.epochs:
            msg = (
                f"warmup_epochs ({self.warmup_epochs}) must be smaller than epochs ({self.epochs})"
            )
            raise ValueError(msg)
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Build from a plain dictionary."""
        return cls.model_validate(d)


_SECTIONS = {"data", "model", "loss", "training"}

_OVERRIDES = (
    "data_root",
    "epochs",
    "lr",
    "batch_size",
    "image_size",
    "fold",
    "num_folds",
    "encoder",
    "architecture",
    "warmup_epochs",
    "scheduler_interval",
)


def load_configs(
    config_path: str | Path, overrides: Mapping[str, Any] | argparse.Namespace | None = None
) -> tuple[
    SegmentationDataConfig,
    SegmentationDataConfig,
    SegmentationDataConfig,
    ModelConfig,
    LossConfig,
    TrainingConfig,
    int,
]:
    """Load experiment config from YAML and apply CLI overrides.

    Parameters
    ----------
    config_path
        Path to the experiment YAML file.
    overrides
        Values replacing the YAML ones, by CLI name (``data_root``, ``epochs``,
        ``lr``, ``batch_size``, ``image_size``, ``fold``, ``num_folds``, ``encoder``,
        ``architecture``, ``warmup_epochs``, ``scheduler_interval``); ``None`` or a
        missing key keeps the YAML value. Parsed CLI arguments work as is.

    Returns
    -------
    tuple
        ``(train_data_cfg, val_data_cfg, test_data_cfg, model_cfg,
        loss_cfg, training_cfg, seed)``.
    """
    with Path(config_path).open() as f:
        raw = yaml.safe_load(f)
    unknown_sections = sorted(set(raw) - _SECTIONS)
    if unknown_sections:
        msg = f"{config_path}: unknown sections {unknown_sections}; expected {sorted(_SECTIONS)}"
        raise ValueError(msg)
    if overrides is None:
        given: dict[str, Any] = {}
    elif isinstance(overrides, Mapping):
        given = dict(overrides)
        unknown = sorted(set(given) - set(_OVERRIDES))
        if unknown:
            raise ValueError(f"unknown overrides {unknown}; expected some of {list(_OVERRIDES)}")
    else:
        given = vars(overrides)  # a CLI namespace may carry flags that are not overrides
    args = argparse.Namespace(**{k: given.get(k) for k in _OVERRIDES})

    data_raw: dict[str, Any] = dict(raw["data"])
    model_raw: dict[str, Any] = dict(raw["model"])
    loss_raw: dict[str, Any] = dict(raw.get("loss", {}))
    training_raw: dict[str, Any] = dict(raw.get("training", {}))

    if args.data_root is not None:
        data_raw["root"] = args.data_root
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

    train_data_cfg = SegmentationDataConfig.from_dict(data_raw)
    no_aug = {**train_data_cfg.model_dump(), "augmentation": None}
    # Evaluation always uses the strict consensus ground truth ("binary").
    val_data_cfg = SegmentationDataConfig.from_dict(
        {**no_aug, "split": "val", "mask_type": "binary"}
    )
    test_data_cfg = SegmentationDataConfig.from_dict(
        {**no_aug, "split": "test", "fold": None, "mask_type": "binary"}
    )
    model_cfg = ModelConfig.from_dict(model_raw)
    loss_cfg = LossConfig.from_dict(loss_raw)
    training_cfg = TrainingConfig.from_dict(training_raw)

    seed = train_data_cfg.seed

    return train_data_cfg, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, seed
