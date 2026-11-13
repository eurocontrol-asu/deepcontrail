"""Oriented difference-of-Gaussians line kernels and oriented line detection.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

A contrail is a thin, long, bright line. For each of ``num_kernels`` evenly spaced
orientations a difference-of-Gaussians kernel responds strongly where a line of that
orientation runs through the pixel. This is the line filter of the Mannstein-style method
(Mannstein, Meyer & Wendling, 1999, Int. J. Remote Sensing 20(8), 1641-1660,
doi:10.1080/014311699212650); it is optional and off by default.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import oaconvolve

# A kernel is a difference of two Gaussians of the signed distance ``d`` to its axis. The two
# sigmas, on the kernel's [-1, 1] grid, are sigma1/size and (sigma1 + sigma_delta)/size.


def line_orientations(num_kernels: int) -> np.ndarray:
    """Return the ``num_kernels`` evenly spaced orientations in ``[0, 180)`` degrees.

    Parameters
    ----------
    num_kernels
        Number of orientations.

    Returns
    -------
    np.ndarray
        Orientations in degrees, ``180 * k / num_kernels`` for ``k = 0 .. num_kernels - 1``.
    """
    return np.linspace(0.0, 180.0, num_kernels, endpoint=False)


def _balance(diff: np.ndarray) -> np.ndarray:
    """Scale a zero-sum kernel so its positive lobe sums to +1 and its negative lobe to -1.

    Each weight is divided by the sum of its own lobe: positives by the positive total, the rest
    by the (positive) magnitude of the negative total. A kernel with no negative (or no positive)
    weights is returned unchanged.

    Parameters
    ----------
    diff
        2D array of the kernel's raw weights.

    Returns
    -------
    np.ndarray
        Rescaled kernel, same shape as ``diff``.
    """
    positive_total = float(np.sum(np.where(diff > 0.0, diff, 0.0)))
    negative_total = float(np.sum(np.where(diff < 0.0, -diff, 0.0)))
    if positive_total == 0.0 or negative_total == 0.0:
        return diff
    normaliser = np.where(diff > 0.0, positive_total, negative_total)
    return diff / normaliser


def _kernel_cells(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Column and row coordinates of a ``size`` x ``size`` grid, meshgrid 'xy' indexing."""
    axis = np.linspace(-1.0, 1.0, size)
    cols, rows = np.meshgrid(axis, axis)  # cols vary along axis 1, rows along axis 0
    return cols, rows


@lru_cache(maxsize=16)
def build_kernels(
    num_kernels: int, kernel_size: int, sigma1: float, sigma_delta: float
) -> np.ndarray:
    """Return the oriented line kernels of a configuration, shape ``(num_kernels, size, size)``.

    Built once per configuration and cached, so the same array is shared by every call; it is
    returned read-only so no caller can change the kernels of later calls.

    Parameters
    ----------
    num_kernels
        Number of evenly spaced orientations.
    kernel_size
        Kernel spatial size in pixels.
    sigma1
        Inner Gaussian width (see MannsteinConfig.sigma1: about half of it in pixels).
    sigma_delta
        Outer minus inner Gaussian width, in the same units.

    Returns
    -------
    np.ndarray
        Read-only array of balanced difference-of-Gaussians kernels.
    """
    angles = np.radians(line_orientations(num_kernels))
    cos = np.cos(angles)[:, None, None]
    sin = np.sin(angles)[:, None, None]
    cols, rows = _kernel_cells(kernel_size)
    signed = cos * cols[None] - sin * rows[None]  # (num_kernels, size, size)

    s = float(kernel_size)
    sigma_a = sigma1 / s
    sigma_b = (sigma1 + sigma_delta) / s
    norm_a = 1.0 / (sigma_a * np.sqrt(2.0 * np.pi))
    norm_b = 1.0 / (sigma_b * np.sqrt(2.0 * np.pi))
    gaussian_a = norm_a * np.exp(-((signed / sigma_a) ** 2) / 2.0)
    gaussian_b = norm_b * np.exp(-((signed / sigma_b) ** 2) / 2.0)
    diff = gaussian_a - gaussian_b
    kernels = np.stack([_balance(d) for d in diff])
    kernels.setflags(write=False)
    return kernels


def detect_line_pixels(
    mask: np.ndarray,
    num_kernels: int,
    kernel_size: int,
    sigma1: float,
    sigma_delta: float,
    threshold: float,
) -> np.ndarray:
    """Detect linear features by convolving with oriented difference-of-Gaussians kernels.

    For each orientation the mask is convolved with the oriented kernel; a pixel is detected
    when, for at least one orientation, the response and the mask both exceed ``threshold`` and
    the response is finite. The result is the union over orientations.

    Parameters
    ----------
    mask
        2D array ``(H, W)``, values in ``{0, 1}`` or float ``[0, 1]``.
    num_kernels
        Number of oriented kernels.
    kernel_size
        Kernel spatial size in pixels (odd).
    sigma1
        Inner Gaussian width (see MannsteinConfig.sigma1: about half of it in pixels).
    sigma_delta
        Outer minus inner Gaussian width, in the same units.
    threshold
        Response / mask value threshold for detection.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` mask of detected pixels.
    """
    if mask.size == 0:
        return np.zeros(mask.shape, dtype=bool)

    kernels = build_kernels(num_kernels, kernel_size, sigma1, sigma_delta)
    above = mask > threshold
    detected = np.zeros(mask.shape, dtype=bool)
    for kernel in kernels:
        response = oaconvolve(mask, kernel, mode="same")
        detected |= (response > threshold) & above & np.isfinite(response)
    return detected
