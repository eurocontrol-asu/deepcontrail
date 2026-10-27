"""Conversion of loaded scenes to normalized torch tensors (torch imported lazily)."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from deepcontrail.segmentation.config import NormalizeConfig


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
