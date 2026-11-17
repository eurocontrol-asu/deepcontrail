"""Lightning module for contrail segmentation training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from deepcontrail.core.metrics import dice_coefficient, iou_score


def _prefix_matches(param_name: str, prefix: str) -> bool:
    """Match a parameter name against a prefix: equal, or starting with ``prefix + "."``."""
    return param_name == prefix or param_name.startswith(prefix + ".")


def _name_hint(param_name: str) -> str:
    """Return the first two name components, as a hint of the available prefixes."""
    return ".".join(param_name.split(".")[:2])


def _any_matches(names: list[str], prefix: str) -> bool:
    """Whether ``prefix`` matches at least one of ``names``."""
    return any(_prefix_matches(name, prefix) for name in names)


class SegmentationModule(pl.LightningModule):
    """Thin Lightning wrapper around a plain ``nn.Module`` segmentation model.

    Parameters
    ----------
    model
        Segmentation model returning logits of shape ``(B, 1, H, W)``.
    loss_fn
        Loss function accepting ``(logits, target)`` pairs.
    lr
        Learning rate.
    weight_decay
        Weight decay for the optimizer.
    scheduler
        Scheduler name.
    optimizer
        ``"AdamW"`` (decoupled weight decay) or ``"Adam"`` (weight decay as an L2 penalty).
    epochs
        Total training epochs (used for ``CosineAnnealingLR``'s ``T_max``).
    native_size
        Native mask resolution (height and width) the model output is resized
        back to before loss/metric computation. The dataset keeps masks at this
        size even when images are upscaled to ``image_size`` (``SegmentationDataset.native_size``).
    encoder_lr_scale
        The encoder's learning rate is ``lr * encoder_lr_scale``; ``1.0`` (default) keeps one
        learning rate for all weights (a single optimizer param group, unchanged). Any other
        value splits the parameters into an ``encoder`` group and "the rest", each with its
        own learning rate (warm-up and cosine then scale each group's own lr).
    freeze_encoder_epochs
        The number of first epochs during which the encoder is frozen (``requires_grad`` False
        and ``model.encoder`` in eval mode, so its BatchNorm/LayerNorm statistics are not
        updated); ``0`` (default) never freezes. The encoder thaws once ``current_epoch``
        reaches this value.
    trainable
        Prefixes of the network's parameter names (as in ``model.named_parameters()``) that
        train; every other parameter is frozen for the whole run (``requires_grad`` False and
        its modules in eval mode, so normalisation statistics and dropout are frozen too).
        ``None`` (default) trains everything. Each prefix must match at least one parameter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        lr: float,
        weight_decay: float,
        scheduler: str,
        epochs: int,
        native_size: int,
        warmup_epochs: int = 0,
        scheduler_interval: str = "epoch",
        steps_per_epoch: int = 1,
        optimizer: str = "AdamW",
        encoder_lr_scale: float = 1.0,
        freeze_encoder_epochs: int = 0,
        trainable: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.scheduler_interval = scheduler_interval
        self.steps_per_epoch = steps_per_epoch
        self.optimizer_name = optimizer
        self.native_size = native_size
        self.encoder_lr_scale = encoder_lr_scale
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.trainable = trainable

        # Partial freezing for fine-tuning: freeze every parameter not under a trainable prefix.
        # The prefixes are validated here (each must match a real parameter) and the names of the
        # permanently frozen modules are computed once, to be put in eval mode each epoch.
        self._frozen_module_names: list[str] = []
        if trainable is not None:
            self._freeze_non_trainable()
        # Every run reports how many weights it trains, frozen or not.
        self._report_trainable_counts()

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        # Running (overlap, predicted, target) pixel counts of the training epoch: global
        # train Dice/IoU at constant memory. A plain attribute, not a buffer, so the
        # state_dict (and loading existing checkpoints) is unchanged.
        self._train_counts: torch.Tensor | None = None

    def _encoder(self) -> torch.nn.Module:
        """Return the ``model.encoder`` module (for Unet++ it also holds the added conv stem).

        Raises
        ------
        ValueError
            If the model has no ``encoder`` attribute while an encoder-specific option is set
            (a non-unit ``encoder_lr_scale`` or a positive ``freeze_encoder_epochs``).
        """
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            raise ValueError(
                f"model {type(self.model).__name__} has no 'encoder' attribute; "
                "encoder_lr_scale and freeze_encoder_epochs need an SMP-style model"
            )
        return encoder

    def _encoder_params(self) -> list[torch.nn.Parameter]:
        """Every parameter of the encoder, in one list (used for scaling and freezing)."""
        return list(self._encoder().parameters())

    def _freeze_non_trainable(self) -> None:
        """Validate the trainable prefixes, freeze every other parameter, cache frozen modules.

        Raises
        ------
        ValueError
            If a prefix matches no parameter of ``self.model``; the message lists the unknown
            prefixes and, as a hint, the distinct first two name components available.
        """
        assert self.trainable is not None  # only called when trainable is set
        names = [name for name, _ in self.model.named_parameters()]
        unknown = [prefix for prefix in self.trainable if not _any_matches(names, prefix)]
        if unknown:
            hint = sorted({_name_hint(n) for n in names})
            msg = (
                f"trainable prefixes {unknown} match no parameter of {type(self.model).__name__}; "
                f"parameter names start with {hint}"
            )
            raise ValueError(msg)
        trainable_names = {n for n in names if any(_prefix_matches(n, p) for p in self.trainable)}
        for name, param in self.model.named_parameters():
            param.requires_grad = name in trainable_names
        # A module is frozen when no trainable parameter lives inside it (and it is not the root
        # ""): put it in eval mode each epoch so its norm statistics and dropout stay frozen.
        self._frozen_module_names = [
            module_name
            for module_name, _ in self.model.named_modules()
            if module_name and not any(tn.startswith(module_name + ".") for tn in trainable_names)
        ]

    def _report_trainable_counts(self) -> None:
        """Print how many weights (parameter elements) train vs freeze; every run reports it."""
        prefixes = self.trainable
        n_total = n_train = 0
        for name, param in self.model.named_parameters():
            n_total += param.numel()
            if prefixes is None or any(_prefix_matches(name, p) for p in prefixes):
                n_train += param.numel()
        n_frozen = n_total - n_train
        pct = 100.0 * n_train / n_total if n_total else 0.0
        print(f"Trainable weights: {n_train:,} of {n_total:,} ({pct:.1f}%), frozen {n_frozen:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, resizing model output back to native mask resolution.

        The model runs at the (possibly upscaled) input resolution. Its output
        is interpolated back to ``native_size`` so it matches the native-
        resolution masks used for loss and metric computation.
        """
        output = self.model(x)
        if output.shape[-2] != self.native_size:
            output = F.interpolate(
                output, size=self.native_size, mode="bilinear", align_corners=False
            )
        return output  # type: ignore[no-any-return]

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute the training loss for one batch."""
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.float())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # Per-step loss (every `log_every_n_steps`) for the fine-grained training curve.
        self.log("train_loss_step", loss, on_step=True, on_epoch=False)

        with torch.no_grad():
            preds = torch.sigmoid(logits).squeeze(1) > 0.5
            # Soft labels: a pixel is contrail when more than half the annotators marked it.
            targets = masks > 0.5
            # Exact integer pixel counts; float64 would fail on devices without it (Apple MPS).
            counts = torch.stack([(preds & targets).sum(), preds.sum(), targets.sum()])
            self._train_counts = (
                counts if self._train_counts is None else self._train_counts + counts
            )
        return loss

    def on_train_epoch_start(self) -> None:
        """Freeze the encoder for the first ``freeze_encoder_epochs`` epochs, then thaw it.

        While frozen the encoder's parameters have ``requires_grad`` False (so AdamW skips
        them; they stay in the optimizer and are simply not updated) and ``model.encoder`` is
        put in eval mode so its BatchNorm/LayerNorm statistics are not updated. The thaw happens
        once, when ``current_epoch`` reaches ``freeze_encoder_epochs``; afterwards the encoder
        trains normally (Lightning already put the module in train mode at the epoch start).
        Does nothing for the encoder when ``freeze_encoder_epochs`` is 0. Afterwards the
        permanently frozen submodules (partial freezing via ``trainable``) are put back in eval
        mode too, so their statistics stay frozen for the whole run.
        """
        if self.freeze_encoder_epochs > 0:
            if self.current_epoch < self.freeze_encoder_epochs:
                for param in self._encoder_params():
                    param.requires_grad = False
                # Lightning puts the whole module in train mode before this hook; put the encoder
                # back in eval mode here so its norm statistics stay frozen while it is not trained.
                self._encoder().eval()
                print("encoder frozen")
            elif self.current_epoch == self.freeze_encoder_epochs:
                for param in self._encoder_params():
                    param.requires_grad = True
                print("encoder unfrozen")
        # Lightning put the whole module in train mode before this hook; put the permanently
        # frozen submodules back in eval mode so their norm statistics and dropout stay frozen.
        for module_name in self._frozen_module_names:
            self.model.get_submodule(module_name).eval()

    def on_train_epoch_end(self) -> None:
        """Log global train Dice and IoU (threshold 0.5) for the epoch, then reset."""
        if self._train_counts is None:
            return
        overlap, predicted, target = self._train_counts.tolist()
        smooth = 1e-6  # as in deepcontrail.core.metrics: empty prediction and target give 1
        self.log(
            "train_dice", (2 * overlap + smooth) / (predicted + target + smooth), prog_bar=True
        )
        self.log("train_iou", (overlap + smooth) / (predicted + target - overlap + smooth))
        self._train_counts = None

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Accumulate predictions and targets for global Dice at epoch end."""
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks.float())

        preds_binary = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
        self._val_preds.append(preds_binary.cpu())
        self._val_targets.append(masks.cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Compute global Dice and IoU across the full validation set."""
        all_preds = torch.cat(self._val_preds)
        all_targets = torch.cat(self._val_targets)

        global_dice = dice_coefficient(all_preds, all_targets)
        global_iou = iou_score(all_preds, all_targets)

        self.log("val_dice", global_dice, prog_bar=True)
        self.log("val_iou", global_iou, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        """Build optimizer and learning rate scheduler."""
        optimizers = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}
        if self.optimizer_name not in optimizers:
            raise ValueError(
                f"unknown optimizer {self.optimizer_name!r}; expected {list(optimizers)}"
            )
        if self.encoder_lr_scale == 1.0:
            if self.trainable is None:
                # Default: one flat param iterable, exactly as before — same optimizer, same result.
                params: Any = self.parameters()
            else:
                # Partial freezing: only the trainable parameters go into the optimizer.
                params = [p for p in self.parameters() if p.requires_grad]
        else:
            # Separate the encoder into its own param group at a scaled learning rate; every
            # parameter ends up in exactly one group. The schedulers scale each group's own lr.
            encoder_ids = {id(p) for p in self._encoder_params()}
            groups = [
                {
                    "params": [p for p in self.model.parameters() if id(p) in encoder_ids],
                    "lr": self.lr * self.encoder_lr_scale,
                },
                {
                    "params": [p for p in self.parameters() if id(p) not in encoder_ids],
                    "lr": self.lr,
                },
            ]
            if self.trainable is not None:
                # Partial freezing keeps the grouping but drops frozen params and empty groups.
                kept = []
                for group in groups:
                    trainable = [p for p in group["params"] if p.requires_grad]
                    if trainable:
                        group["params"] = trainable
                        kept.append(group)
                groups = kept
            params = groups
        optimizer = optimizers[self.optimizer_name](
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        per_step = self.scheduler_interval == "step"
        unit = self.steps_per_epoch if per_step else 1

        if self.warmup_epochs > 0 and self.scheduler_name == "CosineAnnealingLR":
            warmup_iters = self.warmup_epochs * unit
            decay_iters = (self.epochs - self.warmup_epochs) * unit

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=decay_iters,
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_iters]
            )
        else:
            t_max = self.epochs * unit if per_step else self.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
            },
        }
