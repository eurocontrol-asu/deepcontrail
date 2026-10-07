"""Tests for TrainingConfig, constants, and load_configs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepcontrail.training import (
    CHECKPOINT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TrainingConfig,
)

# --- Defaults ---


def test_training_config_default_epochs() -> None:
    cfg = TrainingConfig()

    assert cfg.epochs == 50


def test_training_config_default_lr() -> None:
    cfg = TrainingConfig()

    assert cfg.lr == 1e-4


def test_training_config_default_optimizer() -> None:
    cfg = TrainingConfig()

    assert cfg.optimizer == "AdamW"


def test_training_config_default_scheduler() -> None:
    cfg = TrainingConfig()

    assert cfg.scheduler == "CosineAnnealingLR"


def test_training_config_default_weight_decay() -> None:
    cfg = TrainingConfig()

    assert cfg.weight_decay == 1e-4


def test_training_config_default_precision() -> None:
    cfg = TrainingConfig()

    assert cfg.precision == "16-mixed"


def test_training_config_default_gradient_clip_val_is_none() -> None:
    cfg = TrainingConfig()

    assert cfg.gradient_clip_val is None


def test_training_config_default_early_stopping_patience_is_none() -> None:
    cfg = TrainingConfig()

    assert cfg.early_stopping_patience is None


# --- Frozen ---


def test_training_config_cannot_mutate_epochs() -> None:
    cfg = TrainingConfig()

    with pytest.raises(ValidationError):
        cfg.epochs = 10  # type: ignore[misc]


def test_training_config_cannot_mutate_lr() -> None:
    cfg = TrainingConfig()

    with pytest.raises(ValidationError):
        cfg.lr = 0.1  # type: ignore[misc]


# --- from_dict ---


def test_training_config_from_dict_with_overrides() -> None:
    cfg = TrainingConfig.from_dict({"epochs": 10, "lr": 1e-3, "optimizer": "Adam"})

    assert cfg.epochs == 10
    assert cfg.lr == 1e-3
    assert cfg.optimizer == "Adam"


def test_training_config_from_dict_empty_uses_defaults() -> None:
    cfg = TrainingConfig.from_dict({})

    assert cfg.epochs == 50
    assert cfg.lr == 1e-4


def test_training_config_from_dict_rejects_invalid_epochs() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict({"epochs": 0})


def test_training_config_from_dict_rejects_negative_weight_decay() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict({"weight_decay": -0.1})


# --- Constants ---


def test_checkpoint_root_is_on_data_volume() -> None:
    assert str(CHECKPOINT_ROOT).startswith("/data/")


def test_mlflow_tracking_uri_is_set() -> None:
    assert len(MLFLOW_TRACKING_URI) > 0


def test_mlflow_experiment_name_is_set() -> None:
    assert len(MLFLOW_EXPERIMENT_NAME) > 0
