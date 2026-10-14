"""Tests for TrainingConfig, constants, and load_configs."""

from __future__ import annotations

import argparse

import pytest
import yaml
from pydantic import ValidationError

from deepcontrail.training import (
    CHECKPOINT_ROOT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    TrainingConfig,
    load_configs,
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


# --- load_configs cross-validation handling ---


def _args_namespace(**overrides: object) -> argparse.Namespace:
    base = {
        "epochs": None,
        "lr": None,
        "batch_size": None,
        "image_size": None,
        "encoder": None,
        "architecture": None,
        "warmup_epochs": None,
        "scheduler_interval": None,
        "fold": None,
        "num_folds": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)  # type: ignore[arg-type]


def _write_config(tmp_path: object, data: object) -> object:
    path = tmp_path / "experiment.yaml"  # type: ignore[operator]
    path.write_text(yaml.dump(data))  # type: ignore[attr-defined]
    return path


def test_load_configs_fold_enables_cv_on_train_and_val(tmp_path):
    from deepcontrail.training.config import ModelConfig  # noqa: F401

    yaml_data = {
        "data": {"root": "/data/x", "split": "train", "image_size": 384},
        "model": {"architecture": "Unet", "encoder_name": "resnet18"},
        "loss": {"name": "bce"},
        "training": {"epochs": 2},
    }
    config_path = _write_config(tmp_path, yaml_data)

    train, val, test, _model_cfg, _loss_cfg, _train_cfg, _seed = load_configs(
        config_path,  # type: ignore[arg-type]
        _args_namespace(fold=2),  # type: ignore[arg-type]
    )

    assert train.split == "train", "train split keeps its identity under CV"
    assert train.fold == 2, "train inherits the requested fold"
    assert val.split == "val", "val keeps its split under CV"
    assert val.fold == 2, "val inherits fold for CV"
    assert test.fold is None, "test set is independent and must not be folded"


def test_load_configs_num_folds_override(tmp_path):
    yaml_data = {
        "data": {"root": "/data/x", "split": "train"},
        "model": {"architecture": "Unet", "encoder_name": "resnet18"},
        "loss": {"name": "bce"},
        "training": {"epochs": 2},
    }
    config_path = _write_config(tmp_path, yaml_data)

    train, val, _test, *_ = load_configs(
        config_path,  # type: ignore[arg-type]
        _args_namespace(fold=1, num_folds=8),  # type: ignore[arg-type]
    )

    assert train.num_folds == 8, "train inherits the num_folds override"
    assert train.fold == 1, "train keeps its fold under the override"
    assert val.num_folds == 8, "val inherits the num_folds override"
    assert val.fold == 1, "val keeps its fold under the override"


def test_load_configs_fold_disabled_by_default(tmp_path):
    yaml_data = {
        "data": {"root": "/data/x", "split": "train"},
        "model": {"architecture": "Unet", "encoder_name": "resnet18"},
        "loss": {"name": "bce"},
        "training": {"epochs": 2},
    }
    config_path = _write_config(tmp_path, yaml_data)

    train, val, test, *_ = load_configs(
        config_path,  # type: ignore[arg-type]
        _args_namespace(),  # type: ignore[arg-type]
    )

    assert train.fold is None, "train has no fold by default"
    assert val.fold is None, "val has no fold by default"
    assert test.fold is None, "test has no fold by default"


def test_load_configs_eval_always_uses_binary_masks(tmp_path):
    yaml_data = {
        "data": {"root": "/data/x", "split": "train", "mask_type": "probability"},
        "model": {"architecture": "Unet", "encoder_name": "resnet18"},
        "loss": {"name": "bce"},
        "training": {"epochs": 2},
    }
    config_path = _write_config(tmp_path, yaml_data)

    train, val, test, *_ = load_configs(
        config_path,  # type: ignore[arg-type]
        _args_namespace(),  # type: ignore[arg-type]
    )

    assert train.mask_type == "probability", "train keeps soft labels"
    assert val.mask_type == "binary", "val always uses ground truth"
    assert test.mask_type == "binary", "test always uses ground truth"
