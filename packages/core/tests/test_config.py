"""Tests for deepcontrail.core.config — DatasetConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

RECIPE = {
    "channels": [
        {"bands": ["b", "a"], "bounds": [-4, 2]},
        {"bands": ["a", "c"], "bounds": [-4, 5]},
        {"bands": ["a"], "bounds": [243, 303]},
    ]
}


def test_dataset_config_defaults():
    from deepcontrail.core.config import DatasetConfig

    cfg = DatasetConfig(root="/data", false_color=RECIPE)

    assert (cfg.split, cfg.seed, cfg.val_fraction, cfg.fold, cfg.pixel_shift) == (
        "train",
        42,
        0.1,
        None,
        0.0,
    )


def test_dataset_config_is_frozen():
    from deepcontrail.core.config import DatasetConfig

    cfg = DatasetConfig(root="/data", false_color=RECIPE)

    with pytest.raises(ValidationError):
        cfg.root = "/elsewhere"  # type: ignore[misc]


def test_dataset_config_rejects_fold_out_of_range():
    from deepcontrail.core.config import DatasetConfig

    with pytest.raises(ValidationError, match="out of range"):
        DatasetConfig(root="/data", false_color=RECIPE, fold=4, num_folds=4)


def test_dataset_config_from_yaml_ignores_task_fields(tmp_path: Path):
    from deepcontrail.core.config import DatasetConfig

    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {"data": {"root": "/data", "false_color": RECIPE, "image_size": 512, "seed": 3}}
        )
    )

    cfg = DatasetConfig.from_yaml(path)

    assert type(cfg) is DatasetConfig
    assert (cfg.root, cfg.seed) == ("/data", 3), "dataset fields must be read from data:"


def test_dataset_config_requires_a_false_color_recipe():
    from deepcontrail.core.config import DatasetConfig

    with pytest.raises(ValidationError, match="false_color"):
        DatasetConfig(root="/data")  # type: ignore[call-arg]
