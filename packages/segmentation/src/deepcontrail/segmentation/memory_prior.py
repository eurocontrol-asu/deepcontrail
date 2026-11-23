"""Track-memory channel of the training data: the previous label, corrupted like a prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from deepcontrail.segmentation.config import MemoryCorruptionConfig


def corrupt_memory(
    memory: np.ndarray, config: MemoryCorruptionConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return a corrupted copy of a ``(H, W)`` memory in [0, 1] (see ``MemoryCorruptionConfig``).

    In order: the whole map is zeroed with probability ``empty_rate`` (and returned); each
    connected region above zero is removed with probability ``drop_rate``; a false copy shifted by
    ``ghost_shift`` pixels is added with probability ``ghost_rate``; the contrails are widened by a
    radius drawn in ``[0, max_dilation]``; the map moves by a
    random whole offset in ``[-max_shift, max_shift]`` per axis (zeros enter); it is blurred with a
    sigma drawn in ``[0, max_blur_sigma]`` and multiplied by a factor drawn in ``[min_scale, 1]``.
    The result is float32 clipped to [0, 1]; the input is not modified.
    """
    if rng.random() < config.empty_rate:
        return np.zeros(memory.shape, np.float32)
    out = np.array(memory, dtype=np.float32, copy=True)
    if config.drop_rate > 0:
        from scipy.ndimage import label

        # scipy's stub types label()'s return oddly; the values are the map and its region count.
        regions, count = label(out > 0)  # pyright: ignore[reportGeneralTypeIssues]
        if count:
            dropped = np.flatnonzero(rng.random(count) < config.drop_rate) + 1
            out[np.isin(regions, dropped)] = 0.0
    if config.ghost_rate > 0 and rng.random() < config.ghost_rate:
        # A false copy where the image shows no contrail: the memory must be checked, not copied.
        angle = rng.uniform(0.0, 2.0 * np.pi)
        distance = int(rng.integers(config.ghost_shift[0], config.ghost_shift[1] + 1))
        gy, gx = round(distance * np.sin(angle)), round(distance * np.cos(angle))
        out = np.maximum(out, _shift(out, int(gy), int(gx)))
    if config.max_dilation > 0:
        radius = int(rng.integers(0, config.max_dilation + 1))
        if radius:
            from scipy.ndimage import maximum_filter

            out = maximum_filter(out, size=2 * radius + 1)
    if config.max_shift > 0:
        dy, dx = (int(v) for v in rng.integers(-config.max_shift, config.max_shift + 1, size=2))
        out = _shift(out, dy, dx)
    if config.max_blur_sigma > 0:
        sigma = float(rng.uniform(0.0, config.max_blur_sigma))
        if sigma > 0:
            from scipy.ndimage import gaussian_filter

            out = gaussian_filter(out, sigma)
    out = out * float(rng.uniform(config.min_scale, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _shift(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2-D array by whole pixels (down by ``dy``, right by ``dx``), zeros entering."""
    out = np.zeros_like(image)
    height, width = image.shape
    if abs(dy) >= height or abs(dx) >= width:
        return out
    out[max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)] = image[
        max(-dy, 0) : height + min(-dy, 0), max(-dx, 0) : width + min(-dx, 0)
    ]
    return out
