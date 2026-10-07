"""Tests for deepcontrail.data.config — Pydantic config models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
import yaml
from pydantic import ValidationError


def test_normalize_config_defaults():
    from deepcontrail.data.config import NormalizeConfig

    cfg = NormalizeConfig()

    assert cfg.mean == (0.485, 0.456, 0.406), "default should be ImageNet mean"
    assert cfg.std == (0.229, 0.224, 0.225), "default should be ImageNet std"


def test_augmentation_config_defaults():
    from deepcontrail.data.config import AugmentationConfig

    cfg = AugmentationConfig()

    assert cfg.horizontal_flip == 0.5
    assert cfg.vertical_flip == 0.5
    assert cfg.random_rotate_90 == 0.5


def test_data_config_defaults():
    from deepcontrail.data.config import DataConfig, NormalizeConfig

    cfg = DataConfig(root="/tmp/fake")

    assert cfg.root == "/tmp/fake"
    assert cfg.split == "train"
    assert cfg.seed == 42
    assert cfg.image_size == 256
    assert cfg.batch_size == 16
    assert cfg.num_workers == 4
    assert cfg.augmentation is None
    assert cfg.normalize == NormalizeConfig()


def test_data_config_frozen():
    from deepcontrail.data.config import DataConfig

    cfg = DataConfig(root="/tmp/fake")

    with pytest.raises(ValidationError):
        cfg.split = "val"  # type: ignore[misc]


def test_augmentation_config_frozen():
    from deepcontrail.data.config import AugmentationConfig

    cfg = AugmentationConfig()

    with pytest.raises(ValidationError):
        cfg.horizontal_flip = 0.0  # type: ignore[misc]


def test_augmentation_config_rejects_invalid_probability():
    from deepcontrail.data.config import AugmentationConfig

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AugmentationConfig(horizontal_flip=-0.1)

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        AugmentationConfig(vertical_flip=1.5)


def test_data_config_accepts_test_split():
    from deepcontrail.data.config import DataConfig

    cfg = DataConfig(root="/tmp/fake", split="test")

    assert cfg.split == "test"


def test_data_config_rejects_invalid_split():
    from deepcontrail.data.config import DataConfig

    with pytest.raises(ValidationError, match="split"):
        DataConfig(root="/tmp/fake", split="other")


def test_data_config_from_dict():
    from deepcontrail.data.config import AugmentationConfig, DataConfig, NormalizeConfig

    d = {
        "root": "/data/test",
        "split": "val",
        "seed": 123,
        "image_size": 512,
        "batch_size": 8,
        "num_workers": 2,
        "augmentation": {
            "horizontal_flip": 0.3,
            "vertical_flip": 0.0,
            "random_rotate_90": 1.0,
        },
        "normalize": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.1, 0.1, 0.1],
        },
    }

    cfg = DataConfig.from_dict(d)

    assert cfg.root == "/data/test"
    assert cfg.split == "val"
    assert cfg.seed == 123
    assert cfg.image_size == 512
    assert cfg.augmentation == AugmentationConfig(
        horizontal_flip=0.3, vertical_flip=0.0, random_rotate_90=1.0
    )
    assert cfg.normalize == NormalizeConfig(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1))


def test_data_config_from_dict_minimal():
    from deepcontrail.data.config import DataConfig

    cfg = DataConfig.from_dict({"root": "/data/test"})

    assert cfg.root == "/data/test"
    assert cfg.split == "train"
    assert cfg.augmentation is None


def test_data_config_from_yaml(tmp_path: Path):
    from deepcontrail.data.config import DataConfig

    yaml_content = {
        "data": {
            "root": "/data/test",
            "split": "val",
            "seed": 7,
            "augmentation": {
                "horizontal_flip": 0.5,
                "vertical_flip": 0.5,
                "random_rotate_90": 0.5,
            },
        },
        "model": {"architecture": "Unet"},
    }
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(yaml.dump(yaml_content))

    cfg = DataConfig.from_yaml(yaml_path)

    assert cfg.root == "/data/test"
    assert cfg.split == "val"
    assert cfg.seed == 7
    assert cfg.augmentation is not None
    assert cfg.augmentation.horizontal_flip == 0.5


def test_data_config_model_dump_roundtrip():
    from deepcontrail.data.config import AugmentationConfig, DataConfig

    original = DataConfig(
        root="/data/test",
        split="val",
        seed=99,
        augmentation=AugmentationConfig(horizontal_flip=0.3),
    )

    d = original.model_dump()
    restored = DataConfig.from_dict(d)

    assert restored == original, "model_dump -> from_dict should round-trip"
