"""Differentiable loss functions for binary segmentation training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from deepcontrail.segmentation.losses.config import LossConfig


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


class BCELoss(nn.Module):
    """Binary cross-entropy loss wrapping ``smp.losses.SoftBCEWithLogitsLoss``.

    Parameters
    ----------
    pos_weight
        Weight for positive (contrail) class.  Set to ``num_bg / num_fg`` for
        class-imbalanced datasets (e.g. ``175.0`` for 0.567 % foreground).
        ``None`` means no weighting.
    smooth_factor
        Label smoothing factor.  ``None`` means no smoothing.
    """

    def __init__(
        self,
        pos_weight: float | None = None,
        smooth_factor: float | None = None,
    ) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self._loss = smp.losses.SoftBCEWithLogitsLoss(
            pos_weight=pw,
            smooth_factor=smooth_factor,
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute BCE loss, broadcasting target to match prediction rank."""
        if y_true.dim() == y_pred.dim() - 1:
            y_true = y_true.unsqueeze(1)
        return self._loss(y_pred, y_true)  # type: ignore[no-any-return]


class FocalLoss(nn.Module):
    """Binary focal loss wrapping ``smp.losses.FocalLoss``.

    Parameters
    ----------
    gamma
        Focusing parameter.  ``gamma=2.0`` is standard; down-weights easy
        examples and concentrates training on hard ones.  ``gamma=0`` is
        equivalent to BCE.
    alpha
        Balance factor for positive class.  ``None`` means no balancing.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self._loss = smp.losses.FocalLoss(
            mode="binary",
            gamma=gamma,
            alpha=alpha,
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute focal loss."""
        return self._loss(y_pred, y_true)  # type: ignore[no-any-return]


def _soft_erode(x: Tensor) -> Tensor:
    """Soft (differentiable) erosion: cross-shaped min-pooling of ``x``.

    Min-pooling is expressed as ``-maxpool(-x)`` over a cross kernel (a 3x1 and
    a 1x3 pool, taking the element-wise minimum).

    Parameters
    ----------
    x
        Input map, shaped ``(B, 1, H, W)``.

    Returns
    -------
    Tensor
        Eroded map, same shape as ``x``.
    """
    p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: Tensor) -> Tensor:
    """Soft (differentiable) dilation: 3x3 max-pooling of ``x``.

    Parameters
    ----------
    x
        Input map, shaped ``(B, 1, H, W)``.

    Returns
    -------
    Tensor
        Dilated map, same shape as ``x``.
    """
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _soft_open(x: Tensor) -> Tensor:
    """Soft (differentiable) opening: erosion followed by dilation of ``x``.

    Parameters
    ----------
    x
        Input map, shaped ``(B, 1, H, W)``.

    Returns
    -------
    Tensor
        Opened map, same shape as ``x``.
    """
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(x: Tensor, iterations: int) -> Tensor:
    """Soft (differentiable) skeleton of a fuzzy binary map.

    Follows Shit et al. ("clDice: A Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation", CVPR 2021). The skeleton is obtained by
    repeated soft opening; it highlights the thin central structure of connected
    components, so comparing skeletons rewards getting thin lines right.

    Parameters
    ----------
    x
        Fuzzy binary map in ``[0, 1]``, shaped ``(B, 1, H, W)``.
    iterations
        Number of erosion iterations, about the largest half-width (in pixels)
        of the structures the skeleton follows.

    Returns
    -------
    Tensor
        Soft skeleton, same shape as ``x``, values in ``[0, 1]``.
    """
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iterations):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CLDiceLoss(nn.Module):
    """Soft clDice loss for thin, tubular structures (Shit et al., CVPR 2021).

    Compares the soft skeletons of prediction and target rather than the raw
    maps, which rewards getting thin lines (a few pixels wide) right. Intended
    as a component of the composite loss, not on its own.

    Parameters
    ----------
    iterations
        Soft-skeleton iterations, about the largest half-width (in pixels) of
        the structures the loss follows.
    smooth
        Smoothing factor added to numerators and denominators.
    from_logits
        If ``True``, sigmoid is applied to ``y_pred`` internally.
    """

    def __init__(self, iterations: int = 5, smooth: float = 1.0, from_logits: bool = True) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the clDice loss between prediction and target.

        Parameters
        ----------
        y_pred
            Prediction, shaped ``(B, 1, H, W)``: logits if ``from_logits`` else
            probabilities in ``[0, 1]``.
        y_true
            Target, shaped ``(B, 1, H, W)`` or ``(B, H, W)`` (channel axis added
            when missing), soft probabilities or binary in ``[0, 1]``.

        Returns
        -------
        Tensor
            Scalar clDice loss in ``[0, 1]``, pooled over all pixels.
        """
        if y_true.dim() == y_pred.dim() - 1:
            y_true = y_true.unsqueeze(1)
        y = y_true.float()
        # Float32: under bf16 autocast the logits are bfloat16, too coarse for sums over millions of pixels.
        p = torch.sigmoid(y_pred.float()) if self.from_logits else y_pred.float()
        sp = soft_skeleton(p, self.iterations)
        sy = soft_skeleton(y, self.iterations)
        tprec = (torch.sum(sp * y) + self.smooth) / (torch.sum(sp) + self.smooth)
        tsens = (torch.sum(sy * p) + self.smooth) / (torch.sum(sy) + self.smooth)
        return 1 - 2 * tprec * tsens / (tprec + tsens)  # type: ignore[no-any-return]


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

        bce_weight = config.composite_weights.get("bce", 0.0)
        if bce_weight > 0:
            components["bce"] = (
                BCELoss(
                    pos_weight=config.pos_weight,
                    smooth_factor=config.smooth_factor,
                ),
                bce_weight,
            )

        focal_weight = config.composite_weights.get("focal", 0.0)
        if focal_weight > 0:
            components["focal"] = (
                FocalLoss(
                    gamma=config.focal_gamma,
                    alpha=config.focal_alpha,
                ),
                focal_weight,
            )

        cldice_weight = config.composite_weights.get("cldice", 0.0)
        if cldice_weight > 0:
            components["cldice"] = (
                CLDiceLoss(
                    iterations=config.cldice_iterations,
                    from_logits=config.from_logits,
                ),
                cldice_weight,
            )

        return cls(components)


def build_loss(config: LossConfig) -> DiceLoss | TverskyLoss | BCELoss | FocalLoss | CompositeLoss:
    """Construct a loss function from config.

    Parameters
    ----------
    config
        Loss configuration.

    Returns
    -------
    DiceLoss | TverskyLoss | BCELoss | FocalLoss | CompositeLoss
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
    if config.name == "bce":
        return BCELoss(pos_weight=config.pos_weight, smooth_factor=config.smooth_factor)
    if config.name == "focal":
        return FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    if config.name == "composite":
        return CompositeLoss.from_config(config)
    msg = (
        f"Unknown loss name: {config.name!r}. "
        "Expected 'dice', 'tversky', 'bce', 'focal', or 'composite'."
    )
    raise ValueError(msg)
