"""Train a segmentation model from an experiment YAML: ``train()`` and ``deepcontrail-train``.

Lightning's ``Trainer`` runs the training; this module only assembles it from the
config: data loaders, model, loss, ``SegmentationModule``, a checkpoint callback
keeping the best ``val_dice``, and a logger (``csv`` by default, or ``mlflow`` with
the ``deepcontrail-segmentation[mlflow]`` extra). Each run writes into ``<output_dir>/<run_name>/``.

From Python::

    from deepcontrail.segmentation import train

    run = train("experiment.yaml", output_dir="runs")
    print(run.best_checkpoint, run.best_val_dice)

From the command line::

    deepcontrail-train experiment.yaml --output-dir runs [--logger mlflow]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from deepcontrail.core.splits import split_fingerprint
from deepcontrail.segmentation.dataset import SegmentationDataset
from deepcontrail.segmentation.losses import build_loss
from deepcontrail.segmentation.models import build_model
from deepcontrail.segmentation.training.config import load_configs
from deepcontrail.segmentation.training.module import SegmentationModule

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytorch_lightning.loggers import Logger

    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.losses import LossConfig
    from deepcontrail.segmentation.models import ModelConfig
    from deepcontrail.segmentation.training.config import TrainingConfig

LoggerKind = Literal["csv", "mlflow"]
DEFAULT_EXPERIMENT_NAME = "deepcontrail-segmentation"

# Package that each optional logger needs, and the extra that installs it.
_LOGGER_PACKAGES: dict[str, tuple[str, str]] = {"mlflow": ("mlflow", "mlflow")}


@dataclasses.dataclass
class TrainingRun:
    """Everything a training run is made of; ``fit()`` trains it.

    Attributes
    ----------
    run_dir
        ``<output_dir>/<run_name>``: checkpoints and local logs.
    trainer
        The Lightning ``Trainer`` (with the logger and checkpoint callback).
    module
        The ``SegmentationModule`` being trained.
    train_loader, val_loader
        Data loaders of the train and validation splits.
    checkpoint
        Callback saving ``best-{val_dice}.ckpt`` and ``last.ckpt`` in ``run_dir``.
    train_data_cfg, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, seed
        The configs the run was built from (val/test use binary masks, no augmentation).
    """

    run_dir: Path
    trainer: pl.Trainer
    module: SegmentationModule
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    checkpoint: ModelCheckpoint
    train_data_cfg: SegmentationDataConfig
    val_data_cfg: SegmentationDataConfig
    test_data_cfg: SegmentationDataConfig
    model_cfg: ModelConfig
    loss_cfg: LossConfig
    training_cfg: TrainingConfig
    seed: int

    def fit(self) -> None:
        """Train with Lightning's ``Trainer.fit``."""
        self.trainer.fit(self.module, self.train_loader, self.val_loader)

    @property
    def best_checkpoint(self) -> Path | None:
        """Path of the best checkpoint by ``val_dice``, once trained."""
        path = self.checkpoint.best_model_path
        return Path(path) if path else None

    @property
    def best_val_dice(self) -> float | None:
        """Best validation Dice reached during training, once trained."""
        score = self.checkpoint.best_model_score
        return None if score is None else float(score)


def default_run_name(model_cfg: ModelConfig) -> str:
    """Return ``{architecture}-{encoder}-{UTC timestamp}``."""
    ts = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{model_cfg.architecture}-{model_cfg.encoder_name}-{ts}"


def build_dataloaders(
    train_cfg: SegmentationDataConfig, val_cfg: SegmentationDataConfig
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Construct the train (shuffled, augmented) and validation ``DataLoader``."""
    train_ds = SegmentationDataset(train_cfg)
    val_ds = SegmentationDataset(val_cfg)
    persistent = train_cfg.num_workers > 0

    train_loader: DataLoader[Any] = DataLoader(
        train_ds,  # type: ignore[arg-type]  # map-style: __len__ + __getitem__
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=val_cfg.batch_size,
        shuffle=False,
        num_workers=val_cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=False,
    )
    return train_loader, val_loader


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_logger(
    kind: LoggerKind,
    *,
    output_dir: str | Path,
    run_name: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_uri: str | None = None,
) -> Logger:
    """Build the Lightning logger of a run.

    ``csv`` writes ``metrics.csv`` into ``<output_dir>/<run_name>/``. ``mlflow``
    logs to ``mlflow_uri``, else ``$MLFLOW_TRACKING_URI``, else a SQLite database
    ``<output_dir>/mlflow.db``.

    Raises
    ------
    ImportError
        If the logger's package is not installed; the message names the extra.
    """
    output_dir = Path(output_dir)
    if kind in _LOGGER_PACKAGES:
        package, extra = _LOGGER_PACKAGES[kind]
        if not _module_available(package):
            msg = (
                f"the {kind} logger needs {package}: "
                f"pip install 'deepcontrail-segmentation[{extra}]'"
            )
            raise ImportError(msg)
    if kind == "csv":
        return CSVLogger(save_dir=output_dir, name=run_name, version="")
    from pytorch_lightning.loggers import MLFlowLogger

    uri = (
        mlflow_uri
        or os.environ.get("MLFLOW_TRACKING_URI")
        or f"sqlite:///{(output_dir / 'mlflow.db').resolve()}"
    )
    return MLFlowLogger(
        experiment_name=experiment_name, tracking_uri=uri, run_name=run_name, log_model=False
    )


def _record_split(
    run_dir: Path, trainer: pl.Trainer, train_loader: DataLoader[Any], val_loader: DataLoader[Any]
) -> None:
    """Write the exact train/val records to ``run_dir/split/`` and log their fingerprints."""
    split_dir = run_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = {}
    for name, loader in (("train", train_loader), ("val", val_loader)):
        stems: list[str] = loader.dataset.stems  # type: ignore[attr-defined]
        (split_dir / f"{name}.txt").write_text("".join(f"{s}\n" for s in stems))
        fingerprints[f"data/{name}_split_fingerprint"] = split_fingerprint(stems)
    if trainer.logger is not None:
        trainer.logger.log_hyperparams(fingerprints)


def load_initial_weights(model: torch.nn.Module, source: str, revision: str | None = None) -> None:
    """Load ``source`` weights into ``model`` in place (strict). See TrainingConfig.init_from.

    A 3-channel model initialises a temporal model: the weights go on the centre frame and the
    neighbour frames start at zero, so before training the model gives exactly the source model's
    output on the centre frame. Every input convolution whose channel count is an odd multiple
    greater than one of the source's is widened this way (its neighbour blocks zeroed); every other
    weight is copied as-is, and any other shape mismatch still raises (strict load).

    Parameters
    ----------
    model
        The freshly built network to load the weights into.
    source
        A pretrained bundle (a local folder or a Hugging Face repo id, as accepted by
        ``load_pretrained``) or a Lightning ``.ckpt`` file of this package.
    revision
        Hub revision (tag, branch or commit) when ``source`` is a repo id.

    Raises
    ------
    RuntimeError
        If the weights' architecture does not match ``model``: the message is the one from
        ``load_state_dict`` (a size or key mismatch).
    """
    if source.endswith(".ckpt"):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        # A Lightning checkpoint keeps the module's weights under a "model." prefix.
        weights = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        from deepcontrail.segmentation.pretrained import load_pretrained

        weights = load_pretrained(source, revision=revision).model.state_dict()
    _widen_temporal_inputs(model, weights)
    model.load_state_dict(weights, strict=True)


def _centre_widening_factor(
    source_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> int | None:
    """Return ``m`` when ``target_shape`` widens ``source_shape`` by ``m`` on the input axis.

    ``m`` is the factor by which dimension 1 (a convolution weight's input-channel axis) grows, and
    is returned only when the shapes are identical everywhere else and ``m`` is an odd integer
    greater than one — the ``3*(2k+1)`` stacking of an odd number of frames. ``None`` means the
    shapes are not a temporal widening (equal, or a mismatch that strict loading must reject).
    """
    if len(source_shape) < 2 or len(source_shape) != len(target_shape):
        return None
    if source_shape[0] != target_shape[0] or source_shape[2:] != target_shape[2:]:
        return None
    source_in, target_in = source_shape[1], target_shape[1]
    if source_in == 0 or target_in % source_in != 0:
        return None
    factor = target_in // source_in
    return factor if factor > 1 and factor % 2 == 1 else None


def _widen_temporal_inputs(model: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    """Widen a 3-channel model's input convolutions to initialise a temporal model.

    For every weight present in both ``weights`` (the source) and ``model`` whose shape is a
    temporal widening on the input axis, replace it with a zero tensor of the target shape holding
    the source weights on the centre frame (see :func:`_centre_widening_factor`); the neighbour
    frames start at zero, so before training the model gives exactly the source model's output on
    the centre frame. One line is printed per widened key. Weights that are not a temporal widening
    are left untouched, so a plain 3-channel load is unchanged.
    """
    target_state = model.state_dict()
    for key, source in weights.items():
        target = target_state.get(key)
        if target is None:
            continue
        factor = _centre_widening_factor(tuple(source.shape), tuple(target.shape))
        if factor is None:
            continue
        source_in = source.shape[1]
        widened = torch.zeros_like(target)
        widened[:, (factor // 2) * source_in : (factor // 2 + 1) * source_in] = source
        weights[key] = widened
        print(
            f"Initialising temporal input {key}: 3-channel weights on the centre frame, "
            "neighbour frames at zero"
        )


def _validate_input_channels(context_frames: int, in_channels: int) -> None:
    """Reject a model whose input channels do not match the data's temporal context.

    ``context_frames == k`` stacks ``2k+1`` frames, so the model must take ``3*(2k+1)`` channels;
    with no context it must take the plain three channels.

    Raises
    ------
    ValueError
        If ``in_channels`` is not what ``context_frames`` requires; the message names both the
        required and the given number of channels.
    """
    frames = 2 * context_frames + 1
    expected = 3 * frames
    if context_frames > 0:
        if in_channels != expected:
            msg = (
                f"context_frames={context_frames} stacks {frames} frames, so in_channels must be "
                f"{expected} (3 x {frames}); got {in_channels}"
            )
            raise ValueError(msg)
    elif in_channels != 3:
        msg = (
            f"context_frames=0 needs in_channels=3, got {in_channels}: a model with more input "
            "channels needs context_frames > 0"
        )
        raise ValueError(msg)


def enable_gradient_checkpointing(model: torch.nn.Module) -> int:
    """Make the network's timm stages recompute their activations during the backward pass.

    Sets ``grad_checkpointing = True`` on every module that has a ``blocks`` attribute and a
    boolean ``grad_checkpointing`` flag: timm's stages (CAFormer/MetaFormer, ConvNeXt, Swin, ...)
    then keep only the inputs of their blocks and recompute the rest in the backward pass. Timm's
    feature wrapper (``FeatureListNet``) has the flag but no ``blocks`` and is left off: on top of
    the stages it would only recompute every block a second time.

    Parameters
    ----------
    model
        The network; modified in place.

    Returns
    -------
    int
        The number of stages switched on.

    Raises
    ------
    ValueError
        If the network has no such stage (e.g. a non-timm encoder such as ``resnet18``), since the
        option would otherwise silently do nothing.
    """
    stages = [
        module
        for module in model.modules()
        if hasattr(module, "blocks")
        and isinstance(getattr(module, "grad_checkpointing", None), bool)
    ]
    if not stages:
        msg = (
            "gradient_checkpointing: the network has no timm stage that supports it (a module with "
            "`blocks` and a `grad_checkpointing` flag); use a timm encoder (tu-...) or turn it off"
        )
        raise ValueError(msg)
    for stage in stages:
        stage.grad_checkpointing = True  # pyright: ignore[reportArgumentType]
    return len(stages)


def prepare_training(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    run_name: str | None = None,
    logger: LoggerKind | None = None,
    overrides: Mapping[str, Any] | argparse.Namespace | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_uri: str | None = None,
    accelerator: str = "auto",
) -> TrainingRun:
    """Assemble a training run from an experiment YAML, without training it.

    Parameters
    ----------
    config_path
        Experiment YAML (``data``, ``model``, ``loss``, ``training`` sections).
    output_dir
        Directory holding one sub-directory per run.
    run_name
        Name of the run; default ``{architecture}-{encoder}-{timestamp}``.
    logger
        Overrides ``training.logger`` of the config.
    overrides
        Values replacing the YAML ones (see ``load_configs``).
    experiment_name, mlflow_uri
        MLflow experiment and tracking URI, for the ``mlflow`` logger.
    accelerator
        Lightning accelerator (``"auto"``, ``"gpu"``, ``"cpu"``); one device is used.

    Returns
    -------
    TrainingRun
        Call ``fit()`` on it to train.
    """
    train_data_cfg, val_data_cfg, test_data_cfg, model_cfg, loss_cfg, training_cfg, seed = (
        load_configs(config_path, overrides)
    )
    # The model's input channels must match the data's temporal context before anything is built.
    _validate_input_channels(train_data_cfg.context_frames, model_cfg.in_channels)
    if logger is not None:
        # Record the logger actually used, so a logged or printed config tells the truth.
        training_cfg = training_cfg.model_copy(update={"logger": logger})
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    train_loader, val_loader = build_dataloaders(train_data_cfg, val_data_cfg)
    # Masks stay at the dataset's native resolution; the module resizes its output
    # back to it, so the loss runs at mask resolution even when images are upscaled.
    native_size = train_loader.dataset.native_size  # type: ignore[attr-defined]
    model = build_model(model_cfg)
    if training_cfg.init_from is not None:
        # Fine-tuning: start from a pretrained bundle or a Lightning checkpoint, not scratch.
        load_initial_weights(model, training_cfg.init_from, training_cfg.init_revision)
        message = f"Initial weights: {training_cfg.init_from}"
        if training_cfg.init_revision is not None:
            message += f" (revision {training_cfg.init_revision})"
        print(message)
    if training_cfg.gradient_checkpointing:
        # Same gradients, far less GPU memory: activations are recomputed in the backward pass.
        stages = enable_gradient_checkpointing(model)
        print(f"Gradient checkpointing: on for {stages} encoder stages")
    module = SegmentationModule(
        model=model,
        loss_fn=build_loss(loss_cfg),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
        scheduler=training_cfg.scheduler,
        epochs=training_cfg.epochs,
        native_size=native_size,
        warmup_epochs=training_cfg.warmup_epochs,
        scheduler_interval=training_cfg.scheduler_interval,
        steps_per_epoch=len(train_loader),
        optimizer=training_cfg.optimizer,
        encoder_lr_scale=training_cfg.encoder_lr_scale,
        freeze_encoder_epochs=training_cfg.freeze_encoder_epochs,
        trainable=training_cfg.trainable,
    )

    name = run_name or default_run_name(model_cfg)
    run_dir = Path(output_dir) / name
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        checkpoint,
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(),
        ModelSummary(),
    ]
    if training_cfg.checkpoint_every_epoch:
        # Weights of every epoch, for averaging several epochs later; the best/last stay untouched.
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(run_dir / "epochs"),
                filename="epoch-{epoch:02d}",
                save_top_k=-1,
                every_n_epochs=1,
                save_weights_only=True,
            )
        )
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=training_cfg.epochs,
        precision=training_cfg.precision,  # type: ignore[arg-type]
        # Plain-text progress bar and summary: Lightning picks its Rich versions whenever
        # `rich` is installed, and those print nothing when output goes to a log file.
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        logger=build_logger(
            training_cfg.logger,
            output_dir=output_dir,
            run_name=name,
            experiment_name=experiment_name,
            mlflow_uri=mlflow_uri,
        ),
        deterministic=False,
        gradient_clip_val=training_cfg.gradient_clip_val,
    )
    _record_split(run_dir, trainer, train_loader, val_loader)
    return TrainingRun(
        run_dir=run_dir,
        trainer=trainer,
        module=module,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint=checkpoint,
        train_data_cfg=train_data_cfg,
        val_data_cfg=val_data_cfg,
        test_data_cfg=test_data_cfg,
        model_cfg=model_cfg,
        loss_cfg=loss_cfg,
        training_cfg=training_cfg,
        seed=seed,
    )


def train(config_path: str | Path, **kwargs: Any) -> TrainingRun:  # noqa: ANN401
    """Train a model from an experiment YAML; see ``prepare_training`` for the options.

    Returns
    -------
    TrainingRun
        The trained run: ``best_checkpoint``, ``best_val_dice``, ``run_dir``.
    """
    run = prepare_training(config_path, **kwargs)
    run.fit()
    return run


def main(argv: list[str] | None = None) -> int:
    """``deepcontrail-train``: train from an experiment YAML; return 0."""
    parser = argparse.ArgumentParser(
        prog="deepcontrail-train",
        description="Train a DeepContrail segmentation model from an experiment YAML.",
    )
    parser.add_argument("config", type=Path, help="Experiment YAML.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs"), help="Runs directory (default: runs)."
    )
    parser.add_argument("--run-name", default=None, help="Run name (default: model + time).")
    parser.add_argument(
        "--logger",
        choices=["csv", "mlflow"],
        default=None,
        help="Overrides training.logger of the config (default there: csv).",
    )
    parser.add_argument(
        "--accelerator", default="auto", help="Lightning accelerator (default: auto)."
    )
    parser.add_argument("--data-root", default=None, help="Overrides data.root.")
    parser.add_argument("--epochs", type=int, default=None, help="Overrides training.epochs.")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in {"data_root": args.data_root, "epochs": args.epochs}.items() if v}
    run = train(
        args.config,
        output_dir=args.output_dir,
        run_name=args.run_name,
        logger=args.logger,
        overrides=overrides,
        accelerator=args.accelerator,
    )
    print(f"Run directory:   {run.run_dir}")
    print(f"Best checkpoint: {run.best_checkpoint}")
    print(f"Best val Dice:   {run.best_val_dice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
