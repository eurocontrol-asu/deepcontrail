"""Tests for TrainingConfig, constants, and load_configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from deepcontrail.segmentation.training import (
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


def test_training_config_default_precision_is_full_float32() -> None:
    cfg = TrainingConfig()

    assert cfg.precision == "32-true", (
        "the default must train safely on any hardware: float16 gives NaN losses with "
        "attention encoders, and bfloat16 needs an Ampere or newer GPU"
    )


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
        "data_root": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)  # type: ignore[arg-type]


RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _write_config(tmp_path: object, data: object) -> object:
    path = tmp_path / "experiment.yaml"  # type: ignore[operator]
    path.write_text(yaml.dump(data))  # type: ignore[attr-defined]
    return path


def test_load_configs_fold_enables_cv_on_train_and_val(tmp_path):
    from deepcontrail.segmentation.training.config import ModelConfig  # noqa: F401

    yaml_data = {
        "data": {"root": "/data/x", "false_color": RECIPE, "split": "train", "image_size": 384},
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
        "data": {"root": "/data/x", "false_color": RECIPE, "split": "train"},
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
        "data": {"root": "/data/x", "false_color": RECIPE, "split": "train"},
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
        "data": {
            "root": "/data/x",
            "false_color": RECIPE,
            "split": "train",
            "mask_type": "probability",
        },
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


def test_load_configs_data_root_overrides_every_split(tmp_path):
    yaml_data = {
        "data": {"root": "/data/x", "false_color": RECIPE},
        "model": {"architecture": "Unet", "encoder_name": "resnet18"},
    }
    config_path = _write_config(tmp_path, yaml_data)

    train, val, test, *_ = load_configs(
        config_path,  # type: ignore[arg-type]
        _args_namespace(data_root="/elsewhere"),  # type: ignore[arg-type]
    )

    assert {train.root, val.root, test.root} == {"/elsewhere"}


# --- Validation: names, ranges and unknown keys ---


@pytest.mark.parametrize("field", [{"optimizer": "SGD"}, {"scheduler": "StepLR"}])
def test_training_config_rejects_an_optimizer_or_scheduler_that_is_not_implemented(field) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.from_dict(field)


def test_training_config_rejects_a_warmup_as_long_as_the_training() -> None:
    with pytest.raises(ValidationError, match="warmup_epochs"):
        TrainingConfig.from_dict({"epochs": 5, "warmup_epochs": 5})


def test_training_config_rejects_an_unknown_field() -> None:
    with pytest.raises(ValidationError, match="epoch"):
        TrainingConfig.from_dict({"epoch": 5})  # typo of "epochs"


def test_load_configs_rejects_an_unknown_section(tmp_path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump({"data": {}, "model": {}, "trainig": {"epochs": 3}}))

    with pytest.raises(ValueError, match="trainig"):
        load_configs(path)


def test_load_configs_rejects_an_unknown_override_key(tmp_path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.dump({"data": {}, "model": {}}))

    with pytest.raises(ValueError, match="epoch"):
        load_configs(path, {"epoch": 3})  # typo of "epochs"


def test_load_configs_accepts_a_cli_namespace_with_other_flags() -> None:
    reference = Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"
    args = argparse.Namespace(data_root="/data/x", checkpoint_root="/ckpt", run_name="r")

    configs = load_configs(reference, args)

    assert configs[0].root == "/data/x", "a CLI namespace may carry flags load_configs ignores"
