"""Tests for deepcontrail.segmentation.tensors — tensor conversion and normalization."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# to_tensors
# ---------------------------------------------------------------------------


def test_to_tensors_shapes_and_dtypes():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import to_tensors

    image = np.random.RandomState(0).randint(0, 256, (8, 8, 3), dtype=np.uint8)
    mask = np.array([[0, 1], [1, 0]] * 16, dtype=np.uint8).reshape(8, 8)

    img_t, mask_t = to_tensors(image, mask, NormalizeConfig())

    assert isinstance(img_t, torch.Tensor)
    assert img_t.shape == (3, 8, 8), "image should be CHW"
    assert img_t.dtype == torch.float32
    assert isinstance(mask_t, torch.Tensor)
    assert mask_t.shape == (8, 8)
    assert mask_t.dtype == torch.long


def test_to_tensors_applies_normalization():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import to_tensors

    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    raw_scaled = 128.0 / 255.0

    img_t, _ = to_tensors(image, mask, NormalizeConfig())

    assert not torch.allclose(img_t, torch.tensor(raw_scaled)), (
        "values should differ from raw [0,1] scaling after normalization"
    )


def test_norm_tensors_cached_same_object():
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import _norm_tensors

    cfg = NormalizeConfig()
    first = _norm_tensors(cfg.mean, cfg.std)
    second = _norm_tensors(cfg.mean, cfg.std)

    assert first is second, "identical mean/std tuples must return the same cached tensors"
    assert _norm_tensors.cache_info().hits >= 1, "second identical call should be a cache hit"


def test_norm_tensors_returns_mean_std_shapes():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.tensors import _norm_tensors

    mean, std = _norm_tensors((1.0, 2.0, 3.0), (0.1, 0.2, 0.3))

    assert mean.shape == (3, 1, 1)
    assert std.shape == (3, 1, 1)
    assert mean.dtype == torch.float32
    assert std.dtype == torch.float32


# ---------------------------------------------------------------------------
# to_tensors / to_numpy with float32 input
# ---------------------------------------------------------------------------


def test_to_tensors_accepts_float32_image():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import to_tensors

    image = np.random.RandomState(0).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    mask = np.array([[0, 1], [1, 0]] * 16, dtype=np.uint8).reshape(8, 8)

    img_t, _mask_t = to_tensors(image, mask, NormalizeConfig())

    assert img_t.shape == (3, 8, 8)
    assert img_t.dtype == torch.float32
    assert img_t.abs().max() > 0.1, "float32 input must not be divided by 255"


def test_to_tensors_soft_accepts_float32_image():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import to_tensors_soft

    image = np.random.RandomState(0).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    mask = np.random.RandomState(1).uniform(0, 1, (8, 8)).astype(np.float32)

    img_t, _mask_t = to_tensors_soft(image, mask, NormalizeConfig())

    assert img_t.shape == (3, 8, 8)
    assert img_t.dtype == torch.float32
    assert img_t.abs().max() > 0.1, "float32 input must not be divided by 255"


# ---------------------------------------------------------------------------
# stacked temporal images: 3*(2k+1) channels
# ---------------------------------------------------------------------------


def test_to_tensors_tiles_mean_std_for_nine_channels():
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import to_tensors

    cfg = NormalizeConfig()
    # Nine channels, each a distinct constant, so per-channel normalization is identifiable.
    per_channel = np.arange(9, dtype=np.float32).reshape(1, 1, 9)
    image = np.broadcast_to(per_channel, (8, 8, 9)).astype(np.uint8)
    mask = np.zeros((8, 8), np.uint8)

    img_t, _ = to_tensors(image, mask, cfg)

    assert img_t.shape == (9, 8, 8), "the nine stacked channels are kept, not collapsed"
    mean = np.array(cfg.mean + cfg.mean + cfg.mean)
    std = np.array(cfg.std + cfg.std + cfg.std)
    raw = np.broadcast_to(per_channel / 255.0, (8, 8, 9))
    expected = torch.tensor((raw - mean) / std, dtype=torch.float32).permute(2, 0, 1)
    # expected is built with the mean/std tiled once per composite of three channels, so matching
    # it proves the tiling: channels (0,1,2), (3,4,5) and (6,7,8) each use the base (mean, std).
    assert torch.allclose(img_t, expected), "mean/std are tiled once per composite of three"


def test_to_tensors_raises_for_a_non_multiple_of_three_channels():
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import NormalizeConfig
    from deepcontrail.segmentation.tensors import to_tensors, to_tensors_soft

    image = np.zeros((8, 8, 5), np.uint8)
    mask = np.zeros((8, 8), np.uint8)

    with pytest.raises(ValueError, match="multiple of 3"):
        to_tensors(image, mask, NormalizeConfig())
    with pytest.raises(ValueError, match="multiple of 3"):
        to_tensors_soft(image, np.zeros((8, 8), np.float32), NormalizeConfig())
