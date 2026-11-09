"""Tests for deepcontrail.segmentation.models.config — ModelConfig Pydantic model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    from deepcontrail.segmentation.models.config import ModelConfig

    cfg = ModelConfig()

    assert cfg.architecture == "Unet"
    assert cfg.encoder_name == "efficientnet-b0"
    assert cfg.encoder_weights == "imagenet"
    assert cfg.in_channels == 3
    assert cfg.classes == 1
    assert cfg.decoder_attention_type is None
    assert cfg.decoder_use_norm == "batchnorm"
    assert cfg.extra_kwargs == {}


# ---------------------------------------------------------------------------
# Frozen
# ---------------------------------------------------------------------------


def test_model_config_frozen():
    from deepcontrail.segmentation.models.config import ModelConfig

    cfg = ModelConfig()

    with pytest.raises(ValidationError):
        cfg.architecture = "UnetPlusPlus"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_model_config_rejects_invalid_in_channels():
    from deepcontrail.segmentation.models.config import ModelConfig

    with pytest.raises(ValidationError, match="greater than 0"):
        ModelConfig(in_channels=0)


def test_model_config_rejects_invalid_classes():
    from deepcontrail.segmentation.models.config import ModelConfig

    with pytest.raises(ValidationError, match="greater than 0"):
        ModelConfig(classes=-1)


# ---------------------------------------------------------------------------
# from_dict / from_yaml / roundtrip
# ---------------------------------------------------------------------------


def test_model_config_from_dict():
    from deepcontrail.segmentation.models.config import ModelConfig

    d = {
        "architecture": "UnetPlusPlus",
        "encoder_name": "tu-coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 1,
        "decoder_attention_type": "scse",
        "decoder_use_norm": True,
    }

    cfg = ModelConfig.from_dict(d)

    assert cfg.architecture == "UnetPlusPlus"
    assert cfg.encoder_name == "tu-coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
    assert cfg.decoder_attention_type == "scse"
    assert cfg.decoder_use_norm is True


def test_model_config_from_dict_minimal():
    from deepcontrail.segmentation.models.config import ModelConfig

    cfg = ModelConfig.from_dict({})

    assert cfg.architecture == "Unet", "all fields have defaults"
    assert cfg.encoder_name == "efficientnet-b0"


def test_model_config_model_dump_roundtrip():
    from deepcontrail.segmentation.models.config import ModelConfig

    original = ModelConfig(
        architecture="UnetPlusPlus",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=9,
        classes=2,
        decoder_attention_type="scse",
        extra_kwargs={"encoder_depth": 3},
    )

    d = original.model_dump()
    restored = ModelConfig.from_dict(d)

    assert restored == original, "model_dump -> from_dict should round-trip"


def test_model_config_from_yaml(tmp_path: Path):
    from deepcontrail.segmentation.models.config import ModelConfig

    yaml_content = {
        "data": {"root": "/data/test"},
        "model": {
            "architecture": "UnetPlusPlus",
            "encoder_name": "resnet18",
            "encoder_weights": "imagenet",
        },
    }
    yaml_path = tmp_path / "experiment.yaml"
    yaml_path.write_text(yaml.dump(yaml_content))

    cfg = ModelConfig.from_yaml(yaml_path)

    assert cfg.architecture == "UnetPlusPlus"
    assert cfg.encoder_name == "resnet18"
    assert cfg.encoder_weights == "imagenet"


def test_model_config_extra_kwargs_preserved():
    from deepcontrail.segmentation.models.config import ModelConfig

    cfg = ModelConfig(extra_kwargs={"encoder_depth": 3, "decoder_channels": [256, 128]})

    assert cfg.extra_kwargs["encoder_depth"] == 3
    assert cfg.extra_kwargs["decoder_channels"] == [256, 128]

    d = cfg.model_dump()
    assert d["extra_kwargs"]["encoder_depth"] == 3


def test_model_config_rejects_an_unknown_field():
    from deepcontrail.segmentation.models.config import ModelConfig

    with pytest.raises(ValidationError, match="encoder"):
        ModelConfig.from_dict({"encoder": "resnet34"})  # typo of "encoder_name"
