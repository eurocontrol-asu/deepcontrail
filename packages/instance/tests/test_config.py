"""Tests for deepcontrail.instance.config — Mannstein config model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
import yaml
from pydantic import ValidationError


def test_mannstein_config_defaults_are_the_default_pipeline_settings():
    from deepcontrail.instance.config import InstancePipelineConfig, MannsteinConfig

    cfg = MannsteinConfig()

    assert cfg == InstancePipelineConfig().mannstein, "one set of tuned values, not two"
    assert cfg.line_filter is False


def test_mannstein_config_frozen_rejects_mutation():
    from deepcontrail.instance.config import MannsteinConfig

    cfg = MannsteinConfig()

    with pytest.raises(ValidationError):
        cfg.num_kernels = 30  # type: ignore[misc]


def test_mannstein_config_rejects_negative_min_pixels():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(min_pixels=-1)


def test_mannstein_config_rejects_even_kernel_size():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(kernel_size=8)


def test_mannstein_config_from_dict():
    from deepcontrail.instance.config import MannsteinConfig

    d = {
        "num_kernels": 40,
        "kernel_size": 9,
        "min_pixels": 10,
        "line_filter": True,
    }

    cfg = MannsteinConfig.from_dict(d)

    assert cfg.num_kernels == 40
    assert cfg.kernel_size == 9
    assert cfg.min_pixels == 10
    assert cfg.line_filter is True
    assert cfg.lookback == MannsteinConfig().lookback, "unset fields fall back to defaults"


def test_mannstein_config_from_dict_minimal():
    from deepcontrail.instance.config import MannsteinConfig

    cfg = MannsteinConfig.from_dict({})

    assert cfg == MannsteinConfig()


def test_mannstein_config_from_yaml(tmp_path: Path):
    from deepcontrail.instance.config import MannsteinConfig

    yaml_content = {
        "mannstein": {
            "num_kernels": 33,
            "kernel_size": 5,
            "min_pixels": 8,
        }
    }
    yaml_path = tmp_path / "instance.yaml"
    yaml_path.write_text(yaml.dump(yaml_content))

    cfg = MannsteinConfig.from_yaml(yaml_path)

    assert cfg.num_kernels == 33
    assert cfg.kernel_size == 5
    assert cfg.min_pixels == 8


def test_mannstein_config_model_dump_roundtrip():
    from deepcontrail.instance.config import MannsteinConfig

    original = MannsteinConfig(
        num_kernels=40,
        kernel_size=9,
        min_pixels=12,
        line_filter=True,
        filter_avg_size=None,
    )

    d = original.model_dump()
    restored = MannsteinConfig.from_dict(d)

    assert restored == original


def test_mannstein_config_rejects_nonpositive_kernels():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(num_kernels=0)


def test_mannstein_config_rejects_negative_sigma():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(sigma1=-0.5)


def test_mannstein_config_from_yaml_without_mannstein_section_is_refused(tmp_path: Path):
    from deepcontrail.instance.config import MannsteinConfig

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(yaml.dump({"pipeline": {"grow_steps": 0}}))

    with pytest.raises(ValueError, match="no mannstein"):
        MannsteinConfig.from_yaml(yaml_path)


def test_mannstein_config_rejects_r_score_out_of_range():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(min_r_score=1.5)


def test_mannstein_config_rejects_iou_out_of_range():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(iou_threshold=2.0)


def test_mannstein_config_filter_avg_size_accepts_none():
    from deepcontrail.instance.config import MannsteinConfig

    assert MannsteinConfig(filter_avg_size=None).filter_avg_size is None, (
        "None means: reuse fit_avg_size"
    )


def test_mannstein_config_rejects_nonpositive_filter_avg_size():
    from deepcontrail.instance.config import MannsteinConfig

    with pytest.raises(ValidationError):
        MannsteinConfig(filter_avg_size=0)
