"""Pydantic configuration model for loss functions.

Serializable via ``model_dump()`` for MLflow logging.
Uses only pydantic — no torch dependency.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

COMPOSITE_COMPONENTS = ("dice", "tversky", "bce", "focal")


class LossConfig(BaseModel):
    """Loss function configuration for training.

    Parameters
    ----------
    name
        Loss function name: ``"dice"``, ``"tversky"``, ``"bce"``,
        ``"focal"``, or ``"composite"``.
    smooth
        Smoothing factor of the Dice loss. The Tversky loss always uses 0.
    alpha
        Tversky alpha (weight for false positives).
    beta
        Tversky beta (weight for false negatives).
    gamma
        Tversky focusing parameter.
    from_logits
        Whether the Dice and Tversky losses receive raw logits (sigmoid applied internally). The
        BCE and focal losses always take logits.
    composite_weights
        Weights of the composite loss components, among ``"dice"``, ``"tversky"``, ``"bce"`` and
        ``"focal"``: none negative, at least one positive.
        Keys may be ``"dice"``, ``"tversky"``, ``"bce"``, ``"focal"``.
        Only used when ``name="composite"``.
    pos_weight
        BCE positive-class weight.  Set to ``num_bg / num_fg`` to compensate
        for class imbalance (e.g. ``175.0`` for 0.567 % foreground).
        ``None`` means no weighting.
    smooth_factor
        BCE label-smoothing factor.  ``None`` means no smoothing.
    focal_gamma
        Focal loss focusing parameter — down-weights easy (confident) examples.
        ``gamma=0`` is equivalent to BCE; ``gamma=2`` is the standard value.
    focal_alpha
        Focal loss class-balance factor.  ``None`` means no balancing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["dice", "tversky", "bce", "focal", "composite"] = "dice"
    smooth: Annotated[float, Field(ge=0.0)] = 1.0
    alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    beta: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    gamma: Annotated[float, Field(gt=0.0)] = 1.0
    from_logits: bool = True
    composite_weights: dict[str, float] = {"dice": 0.5, "tversky": 0.5}
    pos_weight: float | None = None
    smooth_factor: float | None = None
    focal_gamma: Annotated[float, Field(ge=0.0)] = 2.0
    focal_alpha: float | None = None

    @model_validator(mode="after")
    def _check_composite_weights(self) -> LossConfig:
        """Composite components must be known, non-negative, and at least one positive."""
        unknown = sorted(set(self.composite_weights) - set(COMPOSITE_COMPONENTS))
        if unknown:
            expected = list(COMPOSITE_COMPONENTS)
            msg = f"unknown composite_weights {unknown}; expected some of {expected}"
            raise ValueError(msg)
        negative = sorted(k for k, w in self.composite_weights.items() if w < 0)
        if negative:
            raise ValueError(f"composite_weights must not be negative: {negative}")
        if self.name == "composite" and not any(w > 0 for w in self.composite_weights.values()):
            raise ValueError(
                "a composite loss needs at least one positive weight in composite_weights"
            )
        return self

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LossConfig:
        """Build a ``LossConfig`` from a plain dictionary."""
        return cls.model_validate(d)
