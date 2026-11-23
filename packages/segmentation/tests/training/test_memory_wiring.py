"""Wiring of the track-memory channel between the data and the model config.

The data must add the memory channel exactly when the model asks for it, and validation and
test must carry an empty memory so no previous-frame label leaks into model selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _config(
    tmp_path: Path,
    root: Path,
    *,
    data: dict | None = None,
    model_extra: dict | None = None,
    **training: object,
) -> Path:
    """Write a tiny experiment on ``root``; ``data``/``model_extra`` merge into their sections."""
    data_section = {
        "false_color": RECIPE,
        "image_size": 32,
        "batch_size": 2,
        "num_workers": 0,
        "val_fraction": 0.2,
        **(data or {}),
    }
    config = {
        "data": {"root": str(root), **data_section},
        "model": {
            "architecture": "Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
            **(model_extra or {}),
        },
        "loss": {"name": "bce"},
        "training": {"epochs": 1, "precision": "32-true", **training},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_load_configs_gives_validation_and_test_an_empty_memory(
    tmp_path: Path, synthetic_root: Path
):
    from deepcontrail.segmentation.training.config import load_configs

    path = _config(
        tmp_path,
        synthetic_root,
        data={"memory_prior": "labels", "memory_corruption": {}},
    )

    train, val, test, *_ = load_configs(path)  # type: ignore[arg-type]

    assert train.memory_prior == "labels", "training keeps the label memory it was asked for"
    assert train.memory_corruption is not None, "training keeps the corruption of that memory"
    assert val.memory_prior == "empty", (
        "validation sees an empty memory, so no previous-frame label leaks into selection"
    )
    assert val.memory_corruption is None, "validation carries no corruption either"
    assert test.memory_prior == "empty", "test sees an empty memory too"
    assert test.memory_corruption is None


def test_load_configs_without_memory_keeps_none(tmp_path: Path, synthetic_root: Path):
    from deepcontrail.segmentation.training.config import load_configs

    path = _config(tmp_path, synthetic_root)

    train, val, test, *_ = load_configs(path)  # type: ignore[arg-type]

    assert train.memory_prior == "none", "no memory_prior keeps the plain image channels"
    assert val.memory_prior == "none"
    assert test.memory_prior == "none"
    assert train.memory_corruption is None, "no corruption without a label memory"
    assert val.memory_corruption is None
    assert test.memory_corruption is None


def test_prepare_training_rejects_a_memory_model_without_memory_data(
    tmp_path: Path, synthetic_root: Path
):
    from pydantic import ValidationError

    from deepcontrail.segmentation.training import prepare_training

    path = _config(tmp_path, synthetic_root, data={}, model_extra={"track_memory": True})

    with pytest.raises((ValueError, ValidationError), match="track_memory"):
        prepare_training(path, output_dir=tmp_path / "runs", accelerator="cpu")


def test_prepare_training_rejects_memory_data_for_a_model_without_memory(
    tmp_path: Path, synthetic_root: Path
):
    from deepcontrail.segmentation.training import prepare_training

    path = _config(tmp_path, synthetic_root, data={"memory_prior": "empty"})

    with pytest.raises(ValueError, match="track_memory"):
        prepare_training(path, output_dir=tmp_path / "runs", accelerator="cpu")
