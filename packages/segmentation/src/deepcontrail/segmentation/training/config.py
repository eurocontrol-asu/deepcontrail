"""Training configuration and constants."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepcontrail.segmentation import SegmentationDataConfig
from deepcontrail.segmentation.losses import LossConfig
from deepcontrail.segmentation.models import ModelConfig


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
        Lightning precision. The default ``"32-true"`` trains safely on any hardware.
        ``"bf16-mixed"`` is faster and is what the reference configuration uses; it needs an
        Ampere or newer GPU. Avoid ``"16-mixed"`` with attention encoders such as CAFormer:
        float16 overflows and the loss becomes NaN.
    gradient_clip_val
        Gradient clipping value, or ``None`` to disable.
    logger
        Where Lightning logs metrics: ``"csv"`` (default, no extra package) or
        ``"mlflow"`` (extra ``deepcontrail-segmentation[mlflow]``).
    init_from
        Weights to start from instead of the encoder's ImageNet weights: a pretrained bundle
        (a local folder or a Hugging Face repo id, as accepted by ``load_pretrained``) or a
        Lightning ``.ckpt`` file of this package. The architecture in ``model`` must match;
        weights load strictly.
    init_revision
        Hub revision (tag, branch or commit) when ``init_from`` is a repo id.
    checkpoint_every_epoch
        Also save the weights (weights only) at the end of every epoch, as
        ``<run dir>/epochs/epoch-NN.ckpt``, for averaging several epochs later. The best and last
        checkpoints are unchanged.
    gradient_checkpointing
        Recompute the encoder's activations during the backward pass instead of storing them:
        the same gradients for much less GPU memory, at about 20% more time (Unet++ CAFormer-B36,
        batch 32 at 512 px: 89.6 GiB -> 22.3 GiB). Needs a timm encoder (``tu-...``) whose stages
        support it (CAFormer, ConvNeXt, Swin, ...); with any other encoder the run refuses to start.
        ``False`` (default) stores the activations, unchanged.
    encoder_lr_scale
        The encoder's learning rate is ``lr * encoder_lr_scale`` (warm-up and cosine apply to
        both groups). ``1.0`` (default) keeps one learning rate for all weights, unchanged.
    freeze_encoder_epochs
        The encoder's weights are not updated during the first N epochs (``requires_grad``
        False, BatchNorm/LayerNorm statistics frozen too: eval mode), then train normally.
        ``0`` (default) never freezes; must be smaller than ``epochs``.
    trainable
        Prefixes of the network's parameter names (as in ``model.named_parameters()``, e.g.
        ``decoder`` or ``encoder.encoder.model.stages_0``) that train; every other parameter is
        frozen for the whole run (``requires_grad`` False, its modules in eval mode so
        normalisation statistics and dropout are frozen too). ``None`` (default) trains
        everything. Mutually exclusive with ``freeze_encoder_epochs`` (choose one freezing
        option); an empty list is an error.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(default=50, gt=0)
    lr: float = Field(default=1e-4, gt=0)
    optimizer: Literal["AdamW", "Adam"] = "AdamW"
    scheduler: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    warmup_epochs: int = Field(default=0, ge=0)
    scheduler_interval: Literal["epoch", "step"] = "epoch"
    encoder_lr_scale: float = Field(default=1.0, gt=0)
    freeze_encoder_epochs: int = Field(default=0, ge=0)
    trainable: list[str] | None = None
    weight_decay: float = Field(default=1e-4, ge=0)
    precision: str = "32-true"
    gradient_clip_val: float | None = None
    logger: Literal["csv", "mlflow"] = "csv"
    init_from: str | None = None
    init_revision: str | None = None
    checkpoint_every_epoch: bool = False
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def _check_warmup(self) -> TrainingConfig:
        """Warm-up must leave epochs for the cosine decay."""
        if self.warmup_epochs >= self.epochs:
            msg = (
                f"warmup_epochs ({self.warmup_epochs}) must be smaller than epochs ({self.epochs})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_freeze_encoder(self) -> TrainingConfig:
        """Freezing must leave epochs of normal training after the encoder thaws."""
        if self.freeze_encoder_epochs >= self.epochs:
            msg = (
                f"freeze_encoder_epochs ({self.freeze_encoder_epochs}) must be smaller than "
                f"epochs ({self.epochs})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_trainable(self) -> TrainingConfig:
        """Trainable must name at least one prefix and not clash with encoder freezing."""
        if self.trainable is not None and len(self.trainable) == 0:
            raise ValueError("trainable must list at least one parameter-name prefix")
        if self.trainable is not None and self.freeze_encoder_epochs > 0:
            raise ValueError("choose one freezing option")
        return self

    @model_validator(mode="after")
    def _check_init_revision(self) -> TrainingConfig:
        """Require ``init_from`` whenever ``init_revision`` is given."""
        if self.init_revision is not None and self.init_from is None:
            raise ValueError("init_revision needs init_from")
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
    # Validation and test never see a label of the previous frame: their track memory is empty, so
    # no label leaks into selection (the final, sequential eval feeds the model's own maps).
    memory = "empty" if train_data_cfg.memory_prior != "none" else "none"
    no_aug = {
        **train_data_cfg.model_dump(),
        "augmentation": None,
        "memory_prior": memory,
        "memory_corruption": None,
        "memory_maps": None,
        "memory_maps_rate": 0.0,
    }
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
