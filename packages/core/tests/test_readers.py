"""Tests for deepcontrail.core.readers — image/mask I/O, binarization, pixel shift."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# load_mask
# ---------------------------------------------------------------------------


def test_load_mask_returns_uint8_hw(tmp_path: Path):
    from deepcontrail.core.readers import load_mask

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
    from deepcontrail.core.readers import binarize_mask

    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    result = binarize_mask(mask)

    np.testing.assert_array_equal(result, [[0, 1], [1, 0]])
    assert result.dtype == np.uint8


def test_binarize_mask_custom_threshold():
    from deepcontrail.core.readers import binarize_mask

    mask = np.array([0, 64, 128, 200, 255], dtype=np.uint8)

    result = binarize_mask(mask, threshold=128)

    np.testing.assert_array_equal(result, [0, 0, 0, 1, 1])


def test_binarize_mask_default_threshold_any_nonzero():
    from deepcontrail.core.readers import binarize_mask

    mask = np.array([0, 1, 64, 128, 255], dtype=np.uint8)

    result = binarize_mask(mask)

    np.testing.assert_array_equal(
        result, [0, 1, 1, 1, 1], err_msg="default threshold=0: any nonzero → 1"
    )


def test_binarize_mask_allzero():
    from deepcontrail.core.readers import binarize_mask

    mask = np.zeros((4, 4), dtype=np.uint8)

    result = binarize_mask(mask)

    assert result.sum() == 0


# ---------------------------------------------------------------------------
# validate_binary_mask
# ---------------------------------------------------------------------------


def test_validate_binary_mask_accepts_valid():
    from deepcontrail.core.readers import validate_binary_mask

    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    validate_binary_mask(mask)


def test_validate_binary_mask_accepts_allzero():
    from deepcontrail.core.readers import validate_binary_mask

    validate_binary_mask(np.zeros((4, 4), dtype=np.uint8))


def test_validate_binary_mask_rejects_255():
    from deepcontrail.core.readers import validate_binary_mask

    mask = np.array([[0, 255]], dtype=np.uint8)

    with pytest.raises(AssertionError, match="binary"):
        validate_binary_mask(mask)


# ---------------------------------------------------------------------------
# to_numpy
# ---------------------------------------------------------------------------


def test_to_numpy_shapes_and_dtypes():
    from deepcontrail.core.readers import to_numpy

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
    from deepcontrail.core.readers import to_numpy

    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)

    img_f, _ = to_numpy(image, mask)

    np.testing.assert_allclose(img_f, 1.0, err_msg="[0,1] scaling, no mean/std normalization")


# ---------------------------------------------------------------------------
# shift_image
# ---------------------------------------------------------------------------


def test_shift_image_zero_is_identity():
    from deepcontrail.core.readers import shift_image

    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)

    result = shift_image(image, 0.0)

    np.testing.assert_array_equal(result, image, err_msg="zero shift must be identity")


def test_shift_image_half_pixel_changes_values():
    from deepcontrail.core.readers import shift_image

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = [255, 255, 255]

    result = shift_image(image, 0.5)

    assert not np.array_equal(result, image), "0.5px shift must change pixel values"


def test_shift_image_preserves_shape_and_dtype():
    from deepcontrail.core.readers import shift_image

    rng = np.random.RandomState(7)
    image = rng.randint(0, 256, (16, 12, 3), dtype=np.uint8)

    result = shift_image(image, 0.5)

    assert result.shape == image.shape, "shape must be preserved"
    assert result.dtype == image.dtype, "dtype must be preserved"


def test_shift_image_float32_preserves_dtype():
    from deepcontrail.core.readers import shift_image

    image = np.random.RandomState(42).uniform(0, 1, (8, 8, 3)).astype(np.float32)

    result = shift_image(image, 0.5)

    assert result.dtype == np.float32, "float32 input must return float32"
    assert result.shape == image.shape


def test_shift_image_float32_zero_is_identity():
    from deepcontrail.core.readers import shift_image

    image = np.random.RandomState(42).uniform(0, 1, (8, 8, 3)).astype(np.float32)

    result = shift_image(image, 0.0)

    np.testing.assert_array_equal(result, image, err_msg="zero shift must be identity for float32")


def test_shift_image_by_one_pixel_moves_content_right_and_down():
    from deepcontrail.core.readers import shift_image

    # Single bright pixel at (2, 2) in a float image
    image = np.zeros((5, 5, 3), dtype=np.float32)
    image[2, 2, :] = 1.0

    result = shift_image(image, 1.0)

    # After shifting by 1 pixel right and down, the bright pixel should move to (3, 3)
    assert result[3, 3, 0] > 0.9, "shift of 1.0 moves content right and down by 1 pixel"
    assert result[2, 2, 0] < 0.1, "original position should be near zero after shift"


def test_shift_image_by_half_a_pixel_spreads_a_quarter_into_four_pixels():
    from deepcontrail.core.readers import shift_image

    # Single bright pixel at (2, 2) in a float image
    image = np.zeros((5, 5, 3), dtype=np.float32)
    image[2, 2, 0] = 1.0

    result = shift_image(image, 0.5)

    # After shifting by 0.5 pixels right and down, the intensity spreads into four pixels
    # Bilinear interpolation: 0.25 into each of (2,2), (2,3), (3,2), (3,3)
    assert 0.2 < result[2, 2, 0] < 0.3, "quarter of intensity at (2,2)"
    assert 0.2 < result[2, 3, 0] < 0.3, "quarter of intensity at (2,3)"
    assert 0.2 < result[3, 2, 0] < 0.3, "quarter of intensity at (3,2)"
    assert 0.2 < result[3, 3, 0] < 0.3, "quarter of intensity at (3,3)"
    assert result[1, 1, 0] < 0.01, "no intensity should leak to (1,1)"


def test_to_numpy_accepts_float32_image():
    from deepcontrail.core.readers import to_numpy

    image = np.full((4, 4, 3), 0.5, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)

    img_f, _ = to_numpy(image, mask)

    np.testing.assert_allclose(img_f, 0.5, err_msg="float32 input should not be divided by 255")
