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


def instance_mask_problem(raw: np.ndarray) -> str | None:
    """Describe why an array is not an instance-mask array, or return ``None``.

    The format is ``(N_instances, H, W, N_annotators)`` with values in ``{-1, 0, 1}``: 1 marks
    the pixels of instance n for annotator a, 0 the rest, and -1 an instance that annotator
    did not draw. A scene without contrails has ``N_instances == 0``.
    """
    if raw.ndim != 4:
        return f"expected (instances, H, W, annotators), got shape {raw.shape}"
    if not set(np.unique(raw).tolist()) <= {-1, 0, 1}:
        return "values must be -1, 0 or 1"
    return None


def load_instance_masks(path: str | Path) -> np.ndarray:
    """Load an instance-mask array from an ``instance_masks/<split>/<id>.npy`` file.

    Parameters
    ----------
    path
        Path to the ``.npy`` file.

    Returns
    -------
    np.ndarray
        Array of shape ``(N_instances, H, W, N_annotators)`` with values in ``{-1, 0, 1}``,
        as stored (``int8`` in the DeepContrail format).

    Raises
    ------
    ValueError
        If the file does not hold an array in that format.
    """
    raw = np.load(path)
    problem = instance_mask_problem(raw)
    if problem is not None:
        raise ValueError(f"{path}: {problem}")
    return raw


def extract_annotator_instances(raw: np.ndarray, annotator: int) -> list[np.ndarray]:
    """Return one annotator's instances as boolean masks.

    Instances where the annotator marked no pixel (all -1 or all 0) are dropped.

    Parameters
    ----------
    raw
        Instance-mask array ``(N_instances, H, W, N_annotators)``.
    annotator
        Annotator index (0-based).

    Returns
    -------
    list[np.ndarray]
        Boolean ``(H, W)`` masks, one per instance the annotator drew.
    """
    masks: list[np.ndarray] = []
    for i in range(raw.shape[0]):
        annotator_slice = raw[i, :, :, annotator]
        if np.any(annotator_slice == 1):
            masks.append(annotator_slice == 1)
    return masks


def extract_all_annotator_instances(raw: np.ndarray) -> list[list[np.ndarray]]:
    """Return every annotator's instances: ``result[a]`` is annotator ``a``'s boolean masks.

    Parameters
    ----------
    raw
        Instance-mask array ``(N_instances, H, W, N_annotators)``.

    Returns
    -------
    list[list[np.ndarray]]
        One list per annotator, in annotator order.
    """
    return [extract_annotator_instances(raw, a) for a in range(raw.shape[-1])]
