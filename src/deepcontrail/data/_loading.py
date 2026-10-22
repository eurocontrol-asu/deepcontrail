"""Shared loading utilities for all dataset implementations.

Plain functions for image/mask I/O, validation, and tensor conversion.
Future dataset classes import what they need — composition over inheritance.

This is a private module (leading underscore). It is NOT exported from
``deepcontrail.data.__init__``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from deepcontrail.data.config import SPECTRAL_BAND_KEYS

if TYPE_CHECKING:
    from pathlib import Path

    from deepcontrail.data.config import FalseColorConfig, NormalizeConfig


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


_BAND_INDEX = {key: i for i, key in enumerate(SPECTRAL_BAND_KEYS)}


def false_color_from_spectral(bands: np.ndarray, config: FalseColorConfig) -> np.ndarray:
    """Build the 3-channel composite from a 10-channel spectral array.

    Parameters
    ----------
    bands
        Brightness temperatures with shape ``(H, W, 10)``, channels in
        ``SPECTRAL_BAND_KEYS`` order (the ``spectral_bands/*.npy`` layout).
    config
        Bands and bounds of each output channel.

    Returns
    -------
    np.ndarray
        Float32 ``(H, W, 3)`` array with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``bands`` is not ``(H, W, 10)``.
    """
    if bands.ndim != 3 or bands.shape[-1] != len(SPECTRAL_BAND_KEYS):
        n = len(SPECTRAL_BAND_KEYS)
        msg = f"expected spectral bands of shape (H, W, {n}), got {bands.shape}"
        raise ValueError(msg)

    channels = []
    for channel in config.channels:
        value = bands[..., _BAND_INDEX[channel.bands[0]]]
        if len(channel.bands) == 2:
            value = value - bands[..., _BAND_INDEX[channel.bands[1]]]
        lo, hi = channel.bounds
        channels.append((value - lo) / (hi - lo))
    return np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)


def load_spectral_false_color(path: str | Path, config: FalseColorConfig) -> np.ndarray:
    """Load a ``spectral_bands/*.npy`` file and build its composite with ``config``."""
    return false_color_from_spectral(np.load(path), config)


def load_false_color(path: str | Path) -> np.ndarray:
    """Load a pre-computed ash false-color composite.

    Parameters
    ----------
    path
        Path to a ``false_color/*.npy`` file with shape ``(H, W, 3)``,
        float32 values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Float32 ``(H, W, 3)`` array.
    """
    return np.load(path).astype(np.float32)


def load_image(path: str | Path) -> np.ndarray:
    """Open an image file and return a uint8 RGB numpy array.

    Parameters
    ----------
    path
        Path to the image file (JPEG, PNG, etc.).

    Returns
    -------
    np.ndarray
        Array with shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


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


@functools.lru_cache(maxsize=4)
def _norm_tensors(mean: tuple[float, ...], std: tuple[float, ...]) -> tuple[Any, Any]:
    """Build (mean, std) normalization tensors, cached per unique pair.

    Called once per ``__getitem__``, so without a cache two tensors are
    rebuilt from their tuples on every sample. The pairs are hashable and
    the values never change, so a module-level cache is a safe, cheap
    win. Cached tensors are CPU tensors consumed by a CPU computation.
    """
    import torch

    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return mean_t, std_t


def to_tensors(
    image: np.ndarray,
    mask: np.ndarray,
    normalize: NormalizeConfig,
) -> tuple[Any, Any]:
    """Convert image and mask numpy arrays to torch tensors.

    Parameters
    ----------
    image
        uint8 HWC image array.
    mask
        uint8 HW binary mask array.
    normalize
        Channel-wise mean/std for normalization.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(image, mask)`` where image is float32 CHW (normalized) and
        mask is long HW.
    """
    import torch

    if np.issubdtype(image.dtype, np.floating):
        image_f = image.astype(np.float32)
    else:
        image_f = image.astype(np.float32) / 255.0
    image_t = torch.from_numpy(image_f).permute(2, 0, 1)

    mean, std = _norm_tensors(normalize.mean, normalize.std)
    image_t = (image_t - mean) / std

    mask_t = torch.from_numpy(mask).long()
    return image_t, mask_t


def to_tensors_soft(
    image: np.ndarray,
    mask: np.ndarray,
    normalize: NormalizeConfig,
) -> tuple[Any, Any]:
    """Convert image and soft mask to torch tensors.

    Like ``to_tensors`` but returns the mask as ``float32`` rather than ``long``,
    preserving the continuous probability values used for soft-label training.

    Parameters
    ----------
    image
        uint8 HWC image array.
    mask
        float32 HW soft-mask array with values in ``[0, 1]``.
    normalize
        Channel-wise mean/std for normalization.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(image, mask)`` where image is float32 CHW (normalized) and
        mask is float32 HW.
    """
    import torch

    if np.issubdtype(image.dtype, np.floating):
        image_f = image.astype(np.float32)
    else:
        image_f = image.astype(np.float32) / 255.0
    image_t = torch.from_numpy(image_f).permute(2, 0, 1)

    mean, std = _norm_tensors(normalize.mean, normalize.std)
    image_t = (image_t - mean) / std

    mask_t = torch.from_numpy(mask.astype(np.float32))
    return image_t, mask_t


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
