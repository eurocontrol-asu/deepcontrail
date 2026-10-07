"""Differentiable loss functions for binary segmentation training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from deepcontrail.losses.config import LossConfig


class DiceLoss(nn.Module):
    """Binary Dice loss wrapping ``smp.losses.DiceLoss``.

    Parameters
    ----------
    smooth
        Smoothing factor.
    from_logits
        If ``True``, sigmoid is applied internally.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True) -> None:
        super().__init__()
        self._loss = smp.losses.DiceLoss(
            mode="binary",
            smooth=smooth,
            from_logits=from_logits,
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the Dice loss."""
        return self._loss(y_pred, y_true)  # type: ignore[no-any-return]


class TverskyLoss(nn.Module):
    """Binary Tversky loss wrapping ``smp.losses.TverskyLoss``.

    Parameters
    ----------
    alpha
        Weight for false positives.
    beta
        Weight for false negatives.
    gamma
        Focal parameter.
    smooth
        Smoothing factor.
    from_logits
        If ``True``, sigmoid is applied internally.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.0,
        smooth: float = 0.0,
        from_logits: bool = True,
    ) -> None:
        super().__init__()
        self._loss = smp.losses.TverskyLoss(
            mode="binary",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            smooth=smooth,
            from_logits=from_logits,
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the Tversky loss."""
        return self._loss(y_pred, y_true)  # type: ignore[no-any-return]


class CompositeLoss(nn.Module):
    """Weighted combination of multiple loss functions.

    Parameters
    ----------
    losses
        Mapping of loss name to ``(loss_module, weight)`` pairs.
    """

    def __init__(self, losses: dict[str, tuple[nn.Module, float]]) -> None:
        super().__init__()
        self._components: list[tuple[nn.Module, float]] = []
        for name, (loss_fn, weight) in losses.items():
            self.add_module(name, loss_fn)
            self._components.append((loss_fn, weight))

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the weighted sum of component losses."""
        total = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        for loss_fn, weight in self._components:
            total = total + loss_fn(y_pred, y_true) * weight
        return total

    @classmethod
    def from_config(cls, config: LossConfig) -> CompositeLoss:
        """Build a ``CompositeLoss`` from a ``LossConfig``."""
        components: dict[str, tuple[nn.Module, float]] = {}

        dice_weight = config.composite_weights.get("dice", 0.0)
        if dice_weight > 0:
            components["dice"] = (
                DiceLoss(smooth=config.smooth, from_logits=config.from_logits),
                dice_weight,
            )

        tversky_weight = config.composite_weights.get("tversky", 0.0)
        if tversky_weight > 0:
            components["tversky"] = (
                TverskyLoss(
                    alpha=config.alpha,
                    beta=config.beta,
                    gamma=config.gamma,
                    smooth=0.0,
                    from_logits=config.from_logits,
                ),
                tversky_weight,
            )

        return cls(components)


def build_loss(config: LossConfig) -> DiceLoss | TverskyLoss | CompositeLoss:
    """Construct a loss function from config.

    Parameters
    ----------
    config
        Loss configuration.

    Returns
    -------
    DiceLoss | TverskyLoss | CompositeLoss
        The constructed loss function.

    Raises
    ------
    ValueError
        If ``config.name`` is not a recognized loss name.
    """
    if config.name == "dice":
        return DiceLoss(smooth=config.smooth, from_logits=config.from_logits)
    if config.name == "tversky":
        return TverskyLoss(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            smooth=0.0,
            from_logits=config.from_logits,
        )
    if config.name == "composite":
        return CompositeLoss.from_config(config)
    msg = f"Unknown loss name: {config.name!r}. Expected 'dice', 'tversky', or 'composite'."
    raise ValueError(msg)
