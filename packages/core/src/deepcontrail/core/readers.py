"""Shared readers for scenes on disk: images, masks, and the pixel shift.

Plain numpy/Pillow functions; no torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def shift_image(image: np.ndarray, pixel_shift: float) -> np.ndarray:
    """Apply a sub-pixel translation (right and down) to an HWC image.

    Corrects a known labeling misalignment in the GOES-16 dataset by shifting
    the image content by ``pixel_shift`` pixels to the right and down using
    bilinear interpolation.  Accepts both uint8 and float32 images.

    Parameters
    ----------
    image
        Input array with shape ``(H, W, 3)``, dtype ``uint8`` or ``float32``.
    pixel_shift
        Translation amount in pixels (e.g. 0.5).

    Returns
    -------
    np.ndarray
        Shifted array, same shape and dtype as input.
    """
    if pixel_shift == 0.0:
        return image
    h, w = image.shape[:2]
    is_float = np.issubdtype(image.dtype, np.floating)
    coeffs = (1, 0, -pixel_shift, 0, 1, -pixel_shift)
    if is_float:
        channels = []
        for c in range(image.shape[2]):
            ch = Image.fromarray(image[:, :, c], mode="F")
            shifted_ch = ch.transform(
                (w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR
            )
            channels.append(np.array(shifted_ch, dtype=np.float32))
        return np.stack(channels, axis=-1)
    pil_img = Image.fromarray(image)
    shifted = pil_img.transform(
        (w, h), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR
    )
    return np.array(shifted, dtype=np.uint8)


def load_mask(path: str | Path) -> np.ndarray:
    """Open a grayscale mask file and return a uint8 numpy array.

    Parameters
    ----------
    path
        Path to the mask file (PNG).

    Returns
    -------
    np.ndarray
        Array with shape ``(H, W)`` and dtype ``uint8``.
    """
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.uint8)


def binarize_mask(mask: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Convert a grayscale mask to binary {0, 1}.

    Parameters
    ----------
    mask
        Grayscale mask array (e.g. values in {0, 255}).
    threshold
        Pixels strictly above this value become 1. Default 0 means any
        non-zero pixel is foreground.

    Returns
    -------
    np.ndarray
        Binary uint8 array with values in {0, 1}.
    """
    return (mask > threshold).astype(np.uint8)


def validate_binary_mask(mask: np.ndarray) -> None:
    """Assert that a mask contains only binary values {0, 1}.

    Parameters
    ----------
    mask
        Mask array to validate.

    Raises
    ------
    AssertionError
        If the mask contains values outside {0, 1}.
    """
    unique = set(np.unique(mask).tolist())
    assert unique <= {0, 1}, f"mask must be binary {{0, 1}}, got {unique}"


def to_numpy(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert image to float32 [0,1] without normalization (exploration fallback).

    Parameters
    ----------
    image
        uint8 HWC image array.
    mask
        uint8 HW binary mask array.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(image, mask)`` where image is float32 HWC in [0, 1] and
        mask is uint8 HW in {0, 1}.
    """
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32), mask
    return image.astype(np.float32) / 255.0, mask
