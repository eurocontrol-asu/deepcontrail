"""Lightning module for contrail segmentation training."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from deepcontrail.core.metrics import dice_coefficient, iou_score


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
        Does nothing when ``freeze_encoder_epochs`` is 0, so the default path never touches the
        encoder.
        """
        if self.freeze_encoder_epochs == 0:
            return
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
            # Default: one flat param iterable, exactly as before — same optimizer, same result.
            params: Any = self.parameters()
        else:
            # Separate the encoder into its own param group at a scaled learning rate; every
            # parameter ends up in exactly one group. The schedulers scale each group's own lr.
            encoder_ids = {id(p) for p in self._encoder_params()}
            params = [
                {
                    "params": [p for p in self.model.parameters() if id(p) in encoder_ids],
                    "lr": self.lr * self.encoder_lr_scale,
                },
                {
                    "params": [p for p in self.parameters() if id(p) not in encoder_ids],
                    "lr": self.lr,
                },
            ]
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
