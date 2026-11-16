"""Pure-numpy raster helpers: polygons to masks, and block-mean downscaling of images.

Neither function needs matplotlib, scipy, skimage or Pillow, so the converter can run in a
minimal environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def rasterize(
    polygons: Iterable[Iterable[tuple[float, float]]],
    size: int,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Rasterise polygons into a boolean ``(size, size)`` mask of covered pixel centres.

    Parameters
    ----------
    polygons
        Iterable of polygons; each an iterable of ``(x, y)`` vertices, normalised
        (``x`` = column, ``y`` = row, ``0..1`` from the top-left image corner).
    size
        The grid is ``size`` x ``size`` pixels; vertices are scaled by ``size``.
    offset
        ``(dx, dy)`` added to every vertex, in pixels of this grid.

    Returns
    -------
    np.ndarray
        Boolean ``(size, size)`` mask. A pixel ``(row r, column c)`` has its centre at
        ``(c + 0.5, r + 0.5)``; a pixel is set when its centre is inside any polygon by the
        even-odd (crossing-number) rule. A vertex exactly on an edge may go either way, so
        tests keep centres off edges. Polygons with fewer than three vertices are ignored.

    Notes
    -----
    For each polygon only the pixel centres of its clipped bounding box are tested, vectorised
    over the polygon's edges.
    """
    mask = np.zeros((size, size), dtype=bool)
    dx, dy = offset
    for polygon in polygons:
        verts = np.asarray(polygon, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            continue
        xy = verts * size + (dx, dy)
        x = xy[:, 0]
        y = xy[:, 1]
        c_min = max(0, int(np.floor(x.min() - 0.5)))
        c_max = min(size - 1, int(np.floor(x.max() + 0.5)))
        r_min = max(0, int(np.floor(y.min() - 0.5)))
        r_max = min(size - 1, int(np.floor(y.max() + 0.5)))
        if c_max < c_min or r_max < r_min:
            continue
        rows = np.arange(r_min, r_max + 1, dtype=np.float64) + 0.5
        cols = np.arange(c_min, c_max + 1, dtype=np.float64) + 0.5
        # query centres on a (R, C) grid, edges on an (E,) axis -> crossings over (E, R, C)
        qy = rows[None, :, None]  # (1, R, 1)
        qx = cols[None, None, :]  # (1, 1, C)
        x1 = x[:, None, None]
        y1 = y[:, None, None]
        x2 = np.roll(x, -1)[:, None, None]
        y2 = np.roll(y, -1)[:, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = ((y1 > qy) != (y2 > qy)) & (x1 + (qy - y1) * (x2 - x1) / (y2 - y1) > qx)
        crossings = crosses.sum(axis=0)
        mask[r_min : r_max + 1, c_min : c_max + 1] |= crossings % 2 == 1
    return mask


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Downscale an ``(H, W, C)`` image to ``(H / factor, W / factor, C)`` by block mean.

    Parameters
    ----------
    image
        ``(H, W, C)`` array; ``NaN`` marks no-data and is ignored in each block's mean.
    factor
        Side of the square block to average over. ``H`` and ``W`` must be divisible by it.

    Returns
    -------
    np.ndarray
        ``float32`` ``(H / factor, W / factor, C)`` array. A block that is entirely ``NaN``
        stays ``NaN``.

    Raises
    ------
    ValueError
        If ``image`` has no three dimensions, or ``H``/``W`` is not divisible by ``factor``.
    """
    if image.ndim != 3:
        msg = f"image must be (H, W, C), got shape {image.shape}"
        raise ValueError(msg)
    height, width = image.shape[:2]
    if factor < 1 or height % factor or width % factor:
        msg = f"image shape {image.shape} must be divisible by block factor {factor}"
        raise ValueError(msg)
    blocks = image.reshape(height // factor, factor, width // factor, factor, -1)
    count = np.sum(~np.isnan(blocks), axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return mean.astype(np.float32)
