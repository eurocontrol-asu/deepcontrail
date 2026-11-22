"""Tests for initialising a temporal feature-fusion model from a single-frame model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from deepcontrail.segmentation.models import ModelConfig


def _source_checkpoint(tmp_path, source) -> Path:
    """Write ``source``'s weights as a Lightning checkpoint and return the path."""
    import torch

    state_dict = {f"model.{key}": value for key, value in source.state_dict().items()}
    path = tmp_path / "source.ckpt"
    torch.save({"state_dict": state_dict}, path)
    return path


def _single_frame_cfg() -> ModelConfig:
    from deepcontrail.segmentation.models import ModelConfig

    return ModelConfig(
        architecture="Unet", encoder_name="resnet18", encoder_weights=None, in_channels=3
    )


def _features_cfg() -> ModelConfig:
    from deepcontrail.segmentation.models import ModelConfig

    return ModelConfig(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=9,
        temporal_fusion="features",
    )


def test_load_initial_weights_features_model_gives_the_source_output_on_the_centre_frame(
    tmp_path, full_precision_matmul
):
    import numpy as np
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(0)
    source = build_model(_single_frame_cfg())
    target = build_model(_features_cfg())
    path = _source_checkpoint(tmp_path, source)

    load_initial_weights(target, str(path))

    centre = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    stacked = torch.cat(
        [
            torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(1)),
            centre,
            torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3)),
        ],
        dim=1,
    )
    source.eval()
    target.eval()
    with torch.no_grad():
        from_source = torch.sigmoid(source(centre))
        from_target = torch.sigmoid(target(stacked))
    np.testing.assert_allclose(
        from_target.numpy(),
        from_source.numpy(),
        atol=1e-5,
        err_msg=(
            "a features-fusion model must match the single-frame source on the centre frame before training"
        ),
    )


def test_load_initial_weights_features_model_copies_every_source_weight(tmp_path):
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(0)
    source = build_model(_single_frame_cfg())
    target = build_model(_features_cfg())
    path = _source_checkpoint(tmp_path, source)

    load_initial_weights(target, str(path))

    after = target.state_dict()
    for key, tensor in source.state_dict().items():
        target_key = (
            "encoder.encoder." + key.removeprefix("encoder.") if key.startswith("encoder.") else key
        )
        assert torch.equal(after[target_key], tensor), f"weight {key} must come from the source"


def test_load_initial_weights_features_checkpoint_loads_as_is(tmp_path):
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(1)
    source = build_model(_features_cfg())
    torch.manual_seed(2)
    target = build_model(_features_cfg())
    path = _source_checkpoint(tmp_path, source)

    load_initial_weights(target, str(path))

    after = target.state_dict()
    for key, tensor in source.state_dict().items():
        assert torch.equal(after[key], tensor), f"weight {key} must load unchanged"


def test_load_initial_weights_input_temporal_source_into_features_model_raises(tmp_path):
    import pytest

    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    source = build_model(
        ModelConfig(
            architecture="Unet", encoder_name="resnet18", encoder_weights=None, in_channels=9
        )
    )
    target = build_model(_features_cfg())
    path = _source_checkpoint(tmp_path, source)

    with pytest.raises(RuntimeError, match=r"size mismatch|Unexpected key|Missing key"):
        load_initial_weights(target, str(path))
