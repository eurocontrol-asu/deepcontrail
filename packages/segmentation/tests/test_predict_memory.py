"""Tests for ``predict``'s track-memory channel — the previous frame's map as the last input.

A track-memory model takes the image channels plus one last channel: a contrail probability
map of the previous frame in ``[0, 1]``. These tests build a tiny such model with random
weights (``encoder_weights=None``) and check the memory is appended, reaches the output, and
is validated; a model without track memory must reject it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from deepcontrail.segmentation.pretrained import Pretrained

RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


def _memory_pretrained(
    tmp_path: Path, *, inject: float = 0.0, track_memory: bool = True
) -> Pretrained:
    """Build a tiny track-memory bundle (native_size 32) with random weights; ``inject`` lights it.

    ``inject`` sets the last conv weight of every memory projection to a non-zero value: those
    weights start at zero (so a fresh memory changes nothing), and a value such as 0.05 makes the
    memory channel actually move the output, so the "memory reaches it" checks can see a difference.
    """
    from deepcontrail.core.false_color import FalseColorConfig
    from deepcontrail.segmentation.models import ModelConfig, build_model
    from deepcontrail.segmentation.pretrained import (
        PretrainedConfig,
        load_pretrained,
        save_pretrained,
    )

    cfg = ModelConfig(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        track_memory=track_memory,
    )
    torch.manual_seed(0)
    model = build_model(cfg)
    if inject:
        with torch.no_grad():
            for seq in model.encoder.inject:
                seq[-1].weight.copy_(torch.full_like(seq[-1].weight, inject))
    config = PretrainedConfig(
        format_version=1,
        model=cfg,
        false_color=FalseColorConfig(**RECIPE),
        image_size=32,
        native_size=32,
        tta=False,
    )
    bundle = tmp_path / "bundle"
    save_pretrained(model, config, bundle)
    return load_pretrained(bundle)


def _composite(shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 1, shape).astype(np.float32)


# --- the memory channel -----------------------------------------------------------------


def test_predict_without_memory_equals_an_empty_memory(tmp_path: Path):
    """``memory=None`` (the start of a sequence) is exactly an all-zero memory."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path)
    composite = _composite((32, 32, 3))

    np.testing.assert_array_equal(
        predict(pretrained, composite),
        predict(pretrained, composite, memory=np.zeros((32, 32), np.float32)),
        err_msg="no memory and a zero memory are the same input to the model",
    )


def test_predict_memory_reaches_the_output(tmp_path: Path):
    """With the memory projection lit up, a memory of ones differs from a memory of zeros."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path, inject=0.05)
    composite = _composite((32, 32, 3))

    with_memory = predict(pretrained, composite, memory=np.ones((32, 32), np.float32), tta=False)
    without = predict(pretrained, composite, memory=np.zeros((32, 32), np.float32), tta=False)

    assert not np.allclose(with_memory, without), "the previous frame's map must reach the output"


def test_predict_memory_for_a_model_without_memory_raises(tmp_path: Path):
    """A model without track memory rejects a memory map outright."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path, track_memory=False)
    composite = _composite((32, 32, 3))

    with pytest.raises(ValueError, match="no track memory"):
        predict(pretrained, composite, memory=np.ones((32, 32), np.float32), tta=False)


def test_predict_memory_with_the_wrong_shape_raises(tmp_path: Path):
    """A memory that does not match the composite's H x W is rejected before prediction."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path)
    composite = _composite((32, 32, 3))

    with pytest.raises(ValueError, match="memory shape"):
        predict(pretrained, composite, memory=np.zeros((16, 16), np.float32), tta=False)


def test_predict_memory_outside_the_unit_range_raises(tmp_path: Path):
    """A memory with values outside ``[0, 1]`` is rejected."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path)
    composite = _composite((32, 32, 3))

    with pytest.raises(ValueError, match="memory values"):
        predict(pretrained, composite, memory=np.full((32, 32), 1.5, np.float32), tta=False)


def test_predict_memory_on_a_larger_scene_is_tiled(tmp_path: Path):
    """A scene larger than native size tiles the memory like the image; None equals zeros."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path, inject=0.05)
    composite = _composite((64, 64, 3))
    ones = np.ones((64, 64), np.float32)
    zeros = np.zeros((64, 64), np.float32)

    none_out = predict(pretrained, composite, tta=False)
    zeros_out = predict(pretrained, composite, memory=zeros, tta=False)
    ones_out = predict(pretrained, composite, memory=ones, tta=False)

    assert ones_out.shape == (64, 64), "a larger scene comes back at its own size"
    np.testing.assert_array_equal(
        none_out, zeros_out, err_msg="an empty memory is zeros everywhere"
    )
    assert not np.allclose(ones_out, zeros_out), "the memory changes the tiled output too"


def test_predict_memory_with_tta(tmp_path: Path):
    """Under D4 TTA the memory channel is transformed with the image and still matters."""
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path, inject=0.05)
    composite = _composite((32, 32, 3))

    out = predict(pretrained, composite, memory=np.ones((32, 32), np.float32), tta=True)

    assert out.shape == (32, 32)
    assert out.min() >= 0.0, "sigmoid probabilities stay >= 0"
    assert out.max() <= 1.0, "sigmoid probabilities stay <= 1"
    assert not np.allclose(
        out, predict(pretrained, composite, memory=np.zeros((32, 32), np.float32), tta=True)
    ), "the memory must matter under TTA too"


def test_predict_batch_with_memory(tmp_path: Path, full_precision_matmul):
    """A batch with a memory batch matches the single predictions with their own memories.

    Batching a convolution rounds a few pixels differently from a single frame (the extra memory
    projections add a little), so the check is at 1e-5, not bit-exact.
    """
    from deepcontrail.segmentation.pretrained import predict

    pretrained = _memory_pretrained(tmp_path, inject=0.05)
    composites = _composite((2, 32, 32, 3))
    memories = np.random.default_rng(1).uniform(0, 1, (2, 32, 32)).astype(np.float32)

    batch_out = predict(pretrained, composites, memory=memories, tta=False)
    single_out = np.stack(
        [predict(pretrained, composites[i], memory=memories[i], tta=False) for i in range(2)]
    )

    np.testing.assert_allclose(
        batch_out, single_out, atol=1e-5, err_msg="per-frame memory in a batch"
    )
