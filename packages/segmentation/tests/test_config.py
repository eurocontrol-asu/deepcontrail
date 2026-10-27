"""Tests for deepcontrail.segmentation.config — Pydantic config models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
import yaml
from pydantic import ValidationError


def test_normalize_config_defaults():
    from deepcontrail.segmentation.config import NormalizeConfig

    cfg = NormalizeConfig()

    assert cfg.mean == (0.485, 0.456, 0.406), "default should be ImageNet mean"
    assert cfg.std == (0.229, 0.224, 0.225), "default should be ImageNet std"


def test_augmentation_config_defaults():
    from deepcontrail.segmentation.config import AugmentationConfig

    cfg = AugmentationConfig()

    assert cfg.horizontal_flip == 0.5
    assert cfg.vertical_flip == 0.5
    assert cfg.random_rotate_90 == 0.5


def test_data_config_defaults():
    from deepcontrail.segmentation.config import NormalizeConfig, SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.root == "/tmp/fake"
    assert cfg.split == "train"
    assert cfg.mask_type == "binary"
    assert cfg.seed == 42
    assert cfg.image_size == 256
    assert cfg.batch_size == 16
    assert cfg.num_workers == 4
    assert cfg.augmentation is None
    assert cfg.normalize == NormalizeConfig()


def test_data_config_frozen():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    with pytest.raises(ValidationError):
        cfg.split = "val"  # type: ignore[misc]


def test_augmentation_config_frozen():
    from deepcontrail.segmentation.config import AugmentationConfig

    cfg = AugmentationConfig()

    with pytest.raises(ValidationError):
        cfg.horizontal_flip = 0.0  # type: ignore[misc]


def test_augmentation_config_rejects_invalid_probability():
    from deepcontrail.segmentation.config import AugmentationConfig

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AugmentationConfig(horizontal_flip=-0.1)

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        AugmentationConfig(vertical_flip=1.5)


def test_data_config_accepts_test_split():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake", split="test")

    assert cfg.split == "test"


def test_data_config_rejects_invalid_split():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="split"):
        SegmentationDataConfig(root="/tmp/fake", split="other")


def test_data_config_mask_type_defaults_to_binary():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.mask_type == "binary", "mask_type must default to binary"


def test_data_config_mask_type_accepts_probability():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake", mask_type="probability")

    assert cfg.mask_type == "probability"


def test_data_config_mask_type_rejects_invalid():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="mask_type"):
        SegmentationDataConfig(root="/tmp/fake", mask_type="soft")


def test_data_config_from_dict():
    from deepcontrail.segmentation.config import (
        AugmentationConfig,
        NormalizeConfig,
        SegmentationDataConfig,
    )

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

    cfg = SegmentationDataConfig.from_dict(d)

    assert cfg.root == "/data/test"
    assert cfg.split == "val"
    assert cfg.seed == 123
    assert cfg.image_size == 512
    assert cfg.augmentation == AugmentationConfig(
        horizontal_flip=0.3, vertical_flip=0.0, random_rotate_90=1.0
    )
    assert cfg.normalize == NormalizeConfig(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1))


def test_data_config_from_dict_minimal():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig.from_dict({"root": "/data/test"})

    assert cfg.root == "/data/test"
    assert cfg.split == "train"
    assert cfg.augmentation is None


def test_data_config_from_yaml(tmp_path: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig

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

    cfg = SegmentationDataConfig.from_yaml(yaml_path)

    assert cfg.root == "/data/test"
    assert cfg.split == "val"
    assert cfg.seed == 7
    assert cfg.augmentation is not None
    assert cfg.augmentation.horizontal_flip == 0.5


def test_data_config_model_dump_roundtrip():
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig

    original = SegmentationDataConfig(
        root="/data/test",
        split="val",
        seed=99,
        augmentation=AugmentationConfig(horizontal_flip=0.3),
    )

    d = original.model_dump()
    restored = SegmentationDataConfig.from_dict(d)

    assert restored == original, "model_dump -> from_dict should round-trip"


# --- k-fold cross-validation fields ---


def test_data_config_fold_defaults_none():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.fold is None, "CV disabled by default"
    assert cfg.num_folds == 4


def test_data_config_fold_within_range_ok():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake", fold=3, num_folds=4)

    assert cfg.fold == 3
    assert cfg.num_folds == 4


def test_data_config_fold_out_of_range_rejected():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="fold"):
        SegmentationDataConfig(root="/tmp/fake", fold=4, num_folds=4)

    with pytest.raises(ValidationError, match="fold"):
        SegmentationDataConfig(root="/tmp/fake", fold=-1, num_folds=4)


def test_data_config_num_folds_must_be_at_least_two():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError):
        SegmentationDataConfig(root="/tmp/fake", fold=0, num_folds=0)

    with pytest.raises(ValidationError):
        SegmentationDataConfig(root="/tmp/fake", fold=0, num_folds=1)


def test_data_config_val_fraction_zero_without_fold_rejected():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="val_fraction must be > 0"):
        SegmentationDataConfig(root="/tmp/fake", val_fraction=0.0)


def test_data_config_val_fraction_with_fold_allowed():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake", val_fraction=0.2, fold=0)

    assert cfg.fold == 0
    assert cfg.val_fraction == 0.2


# --- val_fraction ---


def test_data_config_val_fraction_defaults_to_0_1():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.val_fraction == 0.1, "val_fraction must default to 0.1"


def test_data_config_val_fraction_rejects_negative():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError):
        SegmentationDataConfig(root="/tmp/fake", val_fraction=-0.1)


def test_data_config_val_fraction_rejects_one_or_above():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError):
        SegmentationDataConfig(root="/tmp/fake", val_fraction=1.0)

    with pytest.raises(ValidationError):
        SegmentationDataConfig(root="/tmp/fake", val_fraction=1.5)


# --- image_source ---


def test_image_source_defaults_to_jpeg():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.image_source == "jpeg"


def test_image_source_accepts_spectral():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake", image_source="spectral")

    assert cfg.image_source == "spectral"


def test_image_source_rejects_invalid():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="image_source"):
        SegmentationDataConfig(root="/tmp/fake", image_source="raw")


# --- false_color ---

_LEGACY_CHANNELS = (
    (("data_12um", "data_11um"), (-4.0, 2.0)),
    (("data_11um", "data_10um"), (-4.0, 5.0)),
    (("data_11um",), (243.0, 303.0)),
)


def test_spectral_band_keys_match_converter_channel_order():
    from deepcontrail.core.bands import SPECTRAL_BAND_KEYS

    expected = (
        "data_10um",
        "data_11um",
        "data_12um",
        "brightness_temperature_difference",
        "cloud_top",
        "co2",
        "low_vapor",
        "mid_vapor",
        "upper_vapor",
        "ozone",
    )

    assert expected == SPECTRAL_BAND_KEYS, "channel order of spectral_bands/*.npy files on disk"


def test_false_color_defaults_to_band13_legacy_recipe():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake", image_source="spectral")

    got = tuple((ch.bands, ch.bounds) for ch in cfg.false_color.channels)
    assert got == _LEGACY_CHANNELS, (
        "default must stay the band13 legacy recipe so existing configs keep their inputs"
    )


def test_false_color_accepts_custom_bands_from_dict():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig.from_dict(
        {
            "root": "/tmp/fake",
            "false_color": {
                "channels": [
                    {"bands": ["data_12um", "data_11um"], "bounds": [-4, 2]},
                    {"bands": ["data_11um", "cloud_top"], "bounds": [-4, 5]},
                    {"bands": ["data_11um"], "bounds": [243, 303]},
                ]
            },
        }
    )

    assert cfg.false_color.channels[1].bands == ("data_11um", "cloud_top")


def _channels(**green: object) -> list[dict[str, object]]:
    g = {"bands": ["data_11um", "cloud_top"], "bounds": [-4, 5]} | green
    return [
        {"bands": ["data_12um", "data_11um"], "bounds": [-4, 2]},
        g,
        {"bands": ["data_11um"], "bounds": [243, 303]},
    ]


def test_false_color_rejects_unknown_band():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="bands"):
        SegmentationDataConfig.from_dict(
            {"root": "/tmp/fake", "false_color": {"channels": _channels(bands=["band_11"])}}
        )


def test_false_color_rejects_more_than_two_bands_per_channel():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="bands"):
        SegmentationDataConfig.from_dict(
            {
                "root": "/tmp/fake",
                "false_color": {
                    "channels": _channels(bands=["data_11um", "cloud_top", "data_10um"])
                },
            }
        )


def test_false_color_rejects_bounds_not_increasing():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="bounds"):
        SegmentationDataConfig.from_dict(
            {"root": "/tmp/fake", "false_color": {"channels": _channels(bounds=[5, -4])}}
        )


def test_false_color_rejects_channel_count_other_than_three():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="channels"):
        SegmentationDataConfig.from_dict(
            {"root": "/tmp/fake", "false_color": {"channels": _channels()[:2]}}
        )


# --- SPECTRAL_BAND_INFO ---

_VERIFIED_ABI_BANDS = {
    "upper_vapor": 8,
    "mid_vapor": 9,
    "low_vapor": 10,
    "cloud_top": 11,
    "ozone": 12,
    "data_10um": 13,
    "data_11um": 14,
    "data_12um": 15,
    "co2": 16,
}


def test_spectral_band_info_describes_every_key():
    from deepcontrail.core.bands import SPECTRAL_BAND_INFO, SPECTRAL_BAND_KEYS

    assert set(SPECTRAL_BAND_INFO) == set(SPECTRAL_BAND_KEYS), "every channel needs a description"


def test_spectral_band_info_maps_keys_to_verified_abi_bands():
    from deepcontrail.core.bands import SPECTRAL_BAND_INFO

    got = {k: v.abi_band for k, v in SPECTRAL_BAND_INFO.items() if v.abi_band is not None}

    assert got == _VERIFIED_ABI_BANDS, (
        "mapping verified against Kaggle band_XX files: data_10um is band 13, cloud_top band 11"
    )


def test_spectral_band_info_btd_is_derived_not_an_abi_band():
    from deepcontrail.core.bands import SPECTRAL_BAND_INFO

    btd = SPECTRAL_BAND_INFO["brightness_temperature_difference"]

    assert btd.abi_band is None
    assert btd.wavelength_um is None
    assert btd.derived_from == ("data_11um", "data_12um"), "BTD = band 14 - band 15"


# --- relation to the core DatasetConfig ---


def test_segmentation_data_config_extends_dataset_config():
    from deepcontrail.core.config import DatasetConfig
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/data", image_size=512)

    assert isinstance(cfg, DatasetConfig), "task configs must build on the core DatasetConfig"


def test_segmentation_data_config_from_yaml_returns_the_subclass(tmp_path: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({"data": {"root": "/data", "image_size": 512}}))

    cfg = SegmentationDataConfig.from_yaml(path)

    assert type(cfg) is SegmentationDataConfig
    assert cfg.image_size == 512, "task fields must be read too"
