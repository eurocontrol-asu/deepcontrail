"""Tests for deepcontrail.losses.config — LossConfig Pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_loss_config_defaults():
    from deepcontrail.losses.config import LossConfig

    cfg = LossConfig()

    assert cfg.name == "dice"
    assert cfg.smooth == 1.0
    assert cfg.alpha == 0.7
    assert cfg.beta == 0.3
    assert cfg.gamma == 1.0
    assert cfg.from_logits is True
    assert cfg.composite_weights == {"dice": 0.5, "tversky": 0.5}


# ---------------------------------------------------------------------------
# Frozen
# ---------------------------------------------------------------------------


def test_loss_config_frozen():
    from deepcontrail.losses.config import LossConfig

    cfg = LossConfig()

    with pytest.raises(ValidationError):
        cfg.name = "tversky"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_loss_config_rejects_negative_smooth():
    from deepcontrail.losses.config import LossConfig

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        LossConfig(smooth=-1.0)


# ---------------------------------------------------------------------------
# from_dict / roundtrip
# ---------------------------------------------------------------------------


def test_loss_config_from_dict():
    from deepcontrail.losses.config import LossConfig

    d = {
        "name": "tversky",
        "alpha": 0.5,
        "beta": 0.5,
        "gamma": 2.0,
        "from_logits": False,
    }

    cfg = LossConfig.from_dict(d)

    assert cfg.name == "tversky"
    assert cfg.alpha == 0.5
    assert cfg.gamma == 2.0
    assert cfg.from_logits is False


def test_loss_config_model_dump_roundtrip():
    from deepcontrail.losses.config import LossConfig

    original = LossConfig(
        name="composite",
        alpha=0.6,
        beta=0.4,
        composite_weights={"dice": 0.3, "tversky": 0.7},
    )

    d = original.model_dump()
    restored = LossConfig.from_dict(d)

    assert restored == original, "model_dump -> from_dict should round-trip"
