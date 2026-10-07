"""Pydantic configuration model for loss functions.

Serializable via ``model_dump()`` for MLflow logging.
Uses only pydantic — no torch dependency.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class LossConfig(BaseModel):
    """Loss function configuration for training.

    Parameters
    ----------
    name
        Loss function name: ``"dice"``, ``"tversky"``, or ``"composite"``.
    smooth
        Smoothing factor for Dice/Tversky loss.
    alpha
        Tversky alpha (weight for false positives).
    beta
        Tversky beta (weight for false negatives).
    gamma
        Tversky focusing parameter.
    from_logits
        Whether inputs are raw logits (sigmoid applied internally).
    composite_weights
        Weights for composite loss components ``{"dice": w1, "tversky": w2}``.
        Only used when ``name="composite"``.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "dice"
    smooth: Annotated[float, Field(ge=0.0)] = 1.0
    alpha: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    beta: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    gamma: Annotated[float, Field(gt=0.0)] = 1.0
    from_logits: bool = True
    composite_weights: dict[str, float] = {"dice": 0.5, "tversky": 0.5}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LossConfig:
        """Build a ``LossConfig`` from a plain dictionary."""
        return cls.model_validate(d)
