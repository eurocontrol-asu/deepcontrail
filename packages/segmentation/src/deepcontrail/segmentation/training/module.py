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
    epochs
        Total training epochs (used for ``CosineAnnealingLR``'s ``T_max``).
    native_size
        Native mask resolution (height and width) the model output is resized
        back to before loss/metric computation. The dataset keeps masks at this
        size even when images are upscaled to ``image_size`` (``SegmentationDataset.native_size``).
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
        self.native_size = native_size

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

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
        return loss

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
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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
