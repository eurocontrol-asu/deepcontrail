"""Difference-of-Gaussians line kernels and oriented line detection.

Private module (leading underscore); not exported from ``deepcontrail.instance``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import oaconvolve


def zero_sum_normalize(kernel: np.ndarray) -> np.ndarray:
    """Normalize a kernel so positive weights sum to +1 and negative to -1.

    A kernel with no negative (or no positive) weights is returned unchanged.

    Parameters
    ----------
    kernel
        2D array of kernel weights.

    Returns
    -------
    np.ndarray
        Rescaled kernel, same shape as input.
    """
    pos_sum = np.sum(np.where(kernel > 0, kernel, 0.0))
    neg_sum = -np.sum(np.where(kernel <= 0, kernel, 0.0))
    if pos_sum == 0.0 or neg_sum == 0.0:
        return kernel
    return np.where(kernel > 0, kernel / pos_sum, kernel / neg_sum)


def synthesize_line_kernel(
    degrees: float, size: int, sigma1: float, sigma_delta: float
) -> np.ndarray:
    """Build a DoG line-detection kernel at the given orientation.

    Parameters
    ----------
    degrees
        Kernel orientation in degrees.
    size
        Kernel spatial size (``size`` x ``size``).
    sigma1
        Inner Gaussian sigma in pixels.
    sigma_delta
        Difference between the outer and inner Gaussian sigma, in pixels.

    Returns
    -------
    np.ndarray
        Zero-sum-normalized 2D kernel.
    """
    radians = degrees * (np.pi / 180.0)
    n1, n2 = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = (n1 * np.cos(radians)) - (n2 * np.sin(radians))

    sigma1_u = sigma1 / size
    sigma2_u = (sigma1 + sigma_delta) / size
    gaussian1 = (1.0 / (sigma1_u * np.sqrt(2.0 * np.pi))) * np.exp(-((d / sigma1_u) ** 2) / 2.0)
    gaussian2 = (1.0 / (sigma2_u * np.sqrt(2.0 * np.pi))) * np.exp(-((d / sigma2_u) ** 2) / 2.0)

    return zero_sum_normalize(gaussian1 - gaussian2)


def get_angles(num_kernels: int) -> np.ndarray:
    """Return evenly spaced angles in [0, 180) degrees.

    Parameters
    ----------
    num_kernels
        Number of angles (kernels) to generate.

    Returns
    -------
    np.ndarray
        ``num_kernels`` angles in degrees, spanning ``[0, 180)``.
    """
    return np.linspace(0, 180, num_kernels, endpoint=False)


def detect_lines(
    mask: np.ndarray,
    num_kernels: int,
    kernel_size: int,
    sigma1: float,
    sigma_delta: float,
    threshold: float,
) -> np.ndarray:
    """Detect linear features by convolving with oriented DoG kernels.

    Convolves the 2D binary mask with each oriented kernel, detects pixels
    where both the response and the mask exceed the threshold, and returns
    the union of all detections as a boolean mask.

    Parameters
    ----------
    mask
        2D array ``(H, W)``, values in ``{0, 1}`` or float ``[0, 1]``.
    num_kernels
        Number of oriented kernels to use.
    kernel_size
        Kernel spatial size in pixels.
    sigma1
        Inner Gaussian sigma in pixels.
    sigma_delta
        Difference between the outer and inner Gaussian sigma, in pixels.
    threshold
        Response / mask value threshold for detection.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` mask of detected feature pixels.
    """
    if mask.size == 0:
        return np.zeros(mask.shape, dtype=bool)

    angles = get_angles(num_kernels)
    detected = np.zeros(mask.shape, dtype=bool)
    for deg in angles:
        kernel = synthesize_line_kernel(deg, kernel_size, sigma1, sigma_delta)
        response = oaconvolve(mask, kernel, mode="same")
        hits = (response > threshold) & (mask > threshold) & np.isfinite(response)
        detected |= hits
    return detected
