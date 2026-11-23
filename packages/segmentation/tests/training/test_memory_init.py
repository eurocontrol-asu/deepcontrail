"""Tests for initialising a track-memory model from a model without memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def _source_checkpoint(tmp_path: Path, source: object) -> Path:
    """Write ``source``'s weights as a Lightning checkpoint and return the path."""
    import torch

    state_dict = {f"model.{key}": value for key, value in source.state_dict().items()}
    path = tmp_path / "source.ckpt"
    torch.save({"state_dict": state_dict}, path)
    return path


def _single_frame_cfg() -> object:
    from deepcontrail.segmentation.models import ModelConfig

    return ModelConfig(
        architecture="Unet", encoder_name="resnet18", encoder_weights=None, in_channels=3
    )


def _features_cfg() -> object:
    from deepcontrail.segmentation.models import ModelConfig

    return ModelConfig(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=9,
        temporal_fusion="features",
    )


def test_memory_model_from_a_single_frame_model_gives_its_output(tmp_path, full_precision_matmul):
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(0)
    source = build_model(_single_frame_cfg())
    target = build_model(_single_frame_cfg().model_copy(update={"track_memory": True}))
    path = _source_checkpoint(tmp_path, source)

    load_initial_weights(target, str(path))

    image = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    memory = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(1))
    source.eval()
    target.eval()
    with torch.no_grad():
        from_source = torch.sigmoid(source(image))
        from_target = torch.sigmoid(target(torch.cat([image, memory], dim=1)))
    assert torch.allclose(from_target, from_source, atol=1e-5), (
        "a memory model must match the single-frame source on the image, whatever the memory map"
    )


def test_temporal_memory_model_from_a_single_frame_model_gives_its_output_on_the_centre_frame(
    tmp_path, full_precision_matmul
):
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(0)
    source = build_model(_single_frame_cfg())
    target = build_model(
        _single_frame_cfg().model_copy(
            update={"in_channels": 9, "temporal_fusion": "features", "track_memory": True}
        )
    )
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
    memory = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(4))
    source.eval()
    target.eval()
    with torch.no_grad():
        from_source = torch.sigmoid(source(centre))
        from_target = torch.sigmoid(target(torch.cat([stacked, memory], dim=1)))
    assert torch.allclose(from_target, from_source, atol=1e-5), (
        "a temporal memory model must match the source on the centre frame before training"
    )


def test_temporal_memory_model_from_a_temporal_model_gives_its_output(
    tmp_path, full_precision_matmul
):
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(1)
    source = build_model(_features_cfg())
    target = build_model(_features_cfg().model_copy(update={"track_memory": True}))
    path = _source_checkpoint(tmp_path, source)

    load_initial_weights(target, str(path))

    stacked = torch.rand(1, 9, 32, 32, generator=torch.Generator().manual_seed(0))
    memory = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(2))
    source.eval()
    target.eval()
    with torch.no_grad():
        from_source = torch.sigmoid(source(stacked))
        from_target = torch.sigmoid(target(torch.cat([stacked, memory], dim=1)))
    assert torch.allclose(from_target, from_source, atol=1e-5), (
        "a memory temporal model must match its temporal source, whatever the memory map"
    )


def test_memory_checkpoint_loads_as_is(tmp_path):
    import torch

    from deepcontrail.segmentation.models import build_model
    from deepcontrail.segmentation.training.run import load_initial_weights

    torch.manual_seed(1)
    source = build_model(_single_frame_cfg().model_copy(update={"track_memory": True}))
    torch.manual_seed(2)
    target = build_model(_single_frame_cfg().model_copy(update={"track_memory": True}))
    path = _source_checkpoint(tmp_path, source)

    load_initial_weights(target, str(path))

    after = target.state_dict()
    for key, tensor in source.state_dict().items():
        assert torch.equal(after[key], tensor), f"weight {key} must load unchanged"
