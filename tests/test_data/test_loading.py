"""Tests for deepcontrail.data._loading — shared loading utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


def test_load_image_returns_uint8_rgb(tmp_path: Path):
    from deepcontrail.data._loading import load_image

    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    path = tmp_path / "img.jpg"
    Image.fromarray(arr, mode="RGB").save(path)

    result = load_image(path)

    assert result.dtype == np.uint8
    assert result.shape == (8, 8, 3)


def test_load_image_converts_grayscale_to_rgb(tmp_path: Path):
    from deepcontrail.data._loading import load_image

    arr = np.full((8, 8), 128, dtype=np.uint8)
    path = tmp_path / "gray.png"
    Image.fromarray(arr, mode="L").save(path)

    result = load_image(path)

    assert result.shape == (8, 8, 3), "grayscale should be converted to RGB"


def test_load_image_raises_on_missing_file(tmp_path: Path):
    from deepcontrail.data._loading import load_image

    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nonexistent.jpg")


# ---------------------------------------------------------------------------
# load_mask
# ---------------------------------------------------------------------------


def test_load_mask_returns_uint8_hw(tmp_path: Path):
    from deepcontrail.data._loading import load_mask

    arr = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    path = tmp_path / "mask.png"
    Image.fromarray(arr, mode="L").save(path)

    result = load_mask(path)

    assert result.dtype == np.uint8
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# binarize_mask
# ---------------------------------------------------------------------------


def test_binarize_mask_converts_255_to_1():
    from deepcontrail.data._loading import binarize_mask

    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    result = binarize_mask(mask)

    np.testing.assert_array_equal(result, [[0, 1], [1, 0]])
    assert result.dtype == np.uint8


def test_binarize_mask_custom_threshold():
    from deepcontrail.data._loading import binarize_mask

    mask = np.array([0, 64, 128, 200, 255], dtype=np.uint8)

    result = binarize_mask(mask, threshold=128)

    np.testing.assert_array_equal(result, [0, 0, 0, 1, 1])


def test_binarize_mask_default_threshold_any_nonzero():
    from deepcontrail.data._loading import binarize_mask

    mask = np.array([0, 1, 64, 128, 255], dtype=np.uint8)

    result = binarize_mask(mask)

    np.testing.assert_array_equal(
        result, [0, 1, 1, 1, 1], err_msg="default threshold=0: any nonzero → 1"
    )


def test_binarize_mask_allzero():
    from deepcontrail.data._loading import binarize_mask

    mask = np.zeros((4, 4), dtype=np.uint8)

    result = binarize_mask(mask)

    assert result.sum() == 0


# ---------------------------------------------------------------------------
# validate_binary_mask
# ---------------------------------------------------------------------------


def test_validate_binary_mask_accepts_valid():
    from deepcontrail.data._loading import validate_binary_mask

    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    validate_binary_mask(mask)


def test_validate_binary_mask_accepts_allzero():
    from deepcontrail.data._loading import validate_binary_mask

    validate_binary_mask(np.zeros((4, 4), dtype=np.uint8))


def test_validate_binary_mask_rejects_255():
    from deepcontrail.data._loading import validate_binary_mask

    mask = np.array([[0, 255]], dtype=np.uint8)

    with pytest.raises(AssertionError, match="binary"):
        validate_binary_mask(mask)


# ---------------------------------------------------------------------------
# to_tensors
# ---------------------------------------------------------------------------


def test_to_tensors_shapes_and_dtypes():
    torch = pytest.importorskip("torch")
    from deepcontrail.data._loading import to_tensors
    from deepcontrail.data.config import NormalizeConfig

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
    from deepcontrail.data._loading import to_tensors
    from deepcontrail.data.config import NormalizeConfig

    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    raw_scaled = 128.0 / 255.0

    img_t, _ = to_tensors(image, mask, NormalizeConfig())

    assert not torch.allclose(img_t, torch.tensor(raw_scaled)), (
        "values should differ from raw [0,1] scaling after normalization"
    )


def test_norm_tensors_cached_same_object():
    pytest.importorskip("torch")
    from deepcontrail.data._loading import _norm_tensors
    from deepcontrail.data.config import NormalizeConfig

    cfg = NormalizeConfig()
    first = _norm_tensors(cfg.mean, cfg.std)
    second = _norm_tensors(cfg.mean, cfg.std)

    assert first is second, "identical mean/std tuples must return the same cached tensors"
    assert _norm_tensors.cache_info().hits >= 1, "second identical call should be a cache hit"


def test_norm_tensors_returns_mean_std_shapes():
    torch = pytest.importorskip("torch")
    from deepcontrail.data._loading import _norm_tensors

    mean, std = _norm_tensors((1.0, 2.0, 3.0), (0.1, 0.2, 0.3))

    assert mean.shape == (3, 1, 1)
    assert std.shape == (3, 1, 1)
    assert mean.dtype == torch.float32
    assert std.dtype == torch.float32


# ---------------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------------


def test_to_numpy_shapes_and_dtypes():
    from deepcontrail.data._loading import to_numpy

    image = np.random.RandomState(0).randint(0, 256, (8, 8, 3), dtype=np.uint8)
    mask = np.array([[0, 1], [1, 0]] * 16, dtype=np.uint8).reshape(8, 8)

    img_f, mask_out = to_numpy(image, mask)

    assert isinstance(img_f, np.ndarray)
    assert img_f.dtype == np.float32
    assert img_f.shape == (8, 8, 3), "numpy image should stay HWC"
    assert img_f.min() >= 0.0
    assert img_f.max() <= 1.0
    assert isinstance(mask_out, np.ndarray)
    assert mask_out.dtype == np.uint8


def test_to_numpy_does_not_normalize():
    from deepcontrail.data._loading import to_numpy

    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    img_f, _ = to_numpy(image, mask)

    np.testing.assert_allclose(img_f, 1.0, err_msg="[0,1] scaling, no mean/std normalization")


# ---------------------------------------------------------------------------
# shift_image
# ---------------------------------------------------------------------------


def test_shift_image_zero_is_identity():
    from deepcontrail.data._loading import shift_image

    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)

    result = shift_image(image, 0.0)

    np.testing.assert_array_equal(result, image, err_msg="zero shift must be identity")


def test_shift_image_half_pixel_changes_values():
    from deepcontrail.data._loading import shift_image

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = [255, 255, 255]

    result = shift_image(image, 0.5)

    assert not np.array_equal(result, image), "0.5px shift must change pixel values"


def test_shift_image_preserves_shape_and_dtype():
    from deepcontrail.data._loading import shift_image

    rng = np.random.RandomState(7)
    image = rng.randint(0, 256, (16, 12, 3), dtype=np.uint8)

    result = shift_image(image, 0.5)

    assert result.shape == image.shape, "shape must be preserved"
    assert result.dtype == image.dtype, "dtype must be preserved"


def test_shift_image_float32_preserves_dtype():
    from deepcontrail.data._loading import shift_image

    image = np.random.RandomState(42).uniform(0, 1, (8, 8, 3)).astype(np.float32)

    result = shift_image(image, 0.5)

    assert result.dtype == np.float32, "float32 input must return float32"
    assert result.shape == image.shape


def test_shift_image_float32_zero_is_identity():
    from deepcontrail.data._loading import shift_image

    image = np.random.RandomState(42).uniform(0, 1, (8, 8, 3)).astype(np.float32)

    result = shift_image(image, 0.0)

    np.testing.assert_array_equal(result, image, err_msg="zero shift must be identity for float32")


# ---------------------------------------------------------------------------
# ash_false_color
# ---------------------------------------------------------------------------


def test_ash_false_color_shape_and_dtype():
    from deepcontrail.data._loading import ash_false_color

    b10 = np.full((8, 8), 270.0, dtype=np.float32)
    b11 = np.full((8, 8), 273.0, dtype=np.float32)
    b12 = np.full((8, 8), 271.0, dtype=np.float32)

    result = ash_false_color(b10, b11, b12)

    assert result.shape == (8, 8, 3), f"expected (8, 8, 3), got {result.shape}"
    assert result.dtype == np.float32


def test_ash_false_color_values_in_unit_range():
    from deepcontrail.data._loading import ash_false_color

    rng = np.random.RandomState(42)
    b10 = rng.uniform(220, 320, (16, 16)).astype(np.float32)
    b11 = rng.uniform(220, 320, (16, 16)).astype(np.float32)
    b12 = rng.uniform(220, 320, (16, 16)).astype(np.float32)

    result = ash_false_color(b10, b11, b12)

    assert result.min() >= 0.0, f"min {result.min()} is below 0"
    assert result.max() <= 1.0, f"max {result.max()} is above 1"


def test_ash_false_color_known_values():
    from deepcontrail.data._loading import ash_false_color

    b10 = np.array([[270.0]], dtype=np.float32)
    b11 = np.array([[273.0]], dtype=np.float32)
    b12 = np.array([[271.0]], dtype=np.float32)

    result = ash_false_color(b10, b11, b12)

    # R = clip((b12 - b11 + 4) / 6, 0, 1) = (271 - 273 + 4) / 6 = 1/3
    # G = clip((b11 - b10 + 4) / 9, 0, 1) = (273 - 270 + 4) / 9 = 7/9
    # B = clip((b11 - 243) / 60, 0, 1)    = (273 - 243) / 60    = 0.5
    np.testing.assert_allclose(result[0, 0, 0], 1.0 / 3.0, atol=1e-6, err_msg="R channel")
    np.testing.assert_allclose(result[0, 0, 1], 7.0 / 9.0, atol=1e-6, err_msg="G channel")
    np.testing.assert_allclose(result[0, 0, 2], 0.5, atol=1e-6, err_msg="B channel")


def test_ash_false_color_clips_extremes():
    from deepcontrail.data._loading import ash_false_color

    b10 = np.array([[200.0]], dtype=np.float32)
    b11 = np.array([[200.0]], dtype=np.float32)
    b12 = np.array([[300.0]], dtype=np.float32)

    result = ash_false_color(b10, b11, b12)

    assert result[0, 0, 0] == 1.0, "R should be clipped to 1.0 for extreme difference"


# ---------------------------------------------------------------------------
# load_false_color
# ---------------------------------------------------------------------------


def test_load_false_color_returns_float32_hwc(tmp_path: Path):
    from deepcontrail.data._loading import load_false_color

    rng = np.random.RandomState(0)
    fc = rng.uniform(0, 1, (8, 8, 3)).astype(np.float32)
    np.save(tmp_path / "fc.npy", fc)

    result = load_false_color(tmp_path / "fc.npy")

    assert result.dtype == np.float32
    assert result.shape == (8, 8, 3)
    np.testing.assert_array_equal(result, fc)


def test_load_false_color_preserves_values(tmp_path: Path):
    from deepcontrail.data._loading import load_false_color

    fc = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    np.save(tmp_path / "fc.npy", fc)

    result = load_false_color(tmp_path / "fc.npy")

    np.testing.assert_array_equal(result, fc, err_msg="must return pre-computed values as-is")


# ---------------------------------------------------------------------------
# to_tensors / to_numpy with float32 input
# ---------------------------------------------------------------------------


def test_to_tensors_accepts_float32_image():
    torch = pytest.importorskip("torch")
    from deepcontrail.data._loading import to_tensors
    from deepcontrail.data.config import NormalizeConfig

    image = np.random.RandomState(0).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    mask = np.array([[0, 1], [1, 0]] * 16, dtype=np.uint8).reshape(8, 8)

    img_t, _mask_t = to_tensors(image, mask, NormalizeConfig())

    assert img_t.shape == (3, 8, 8)
    assert img_t.dtype == torch.float32
    assert img_t.abs().max() > 0.1, "float32 input must not be divided by 255"


def test_to_tensors_soft_accepts_float32_image():
    torch = pytest.importorskip("torch")
    from deepcontrail.data._loading import to_tensors_soft
    from deepcontrail.data.config import NormalizeConfig

    image = np.random.RandomState(0).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    mask = np.random.RandomState(1).uniform(0, 1, (8, 8)).astype(np.float32)

    img_t, _mask_t = to_tensors_soft(image, mask, NormalizeConfig())

    assert img_t.shape == (3, 8, 8)
    assert img_t.dtype == torch.float32
    assert img_t.abs().max() > 0.1, "float32 input must not be divided by 255"


def test_to_numpy_accepts_float32_image():
    from deepcontrail.data._loading import to_numpy

    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)

    img_f, _ = to_numpy(image, mask)

    np.testing.assert_allclose(img_f, 0.5, err_msg="float32 input should not be divided by 255")
