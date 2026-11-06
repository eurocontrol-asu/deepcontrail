"""Direction-following region growing that splits a binary mask into contrail regions.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

Seeds walk the mask in row-major order. From each seed not already claimed by an earlier
region, a breadth-first flood grows along 8-connected neighbours that stay aligned with the
region's local direction, until the region ends. Each region is then tested for minimum size
and linearity (:func:`deepcontrail.instance._shape.is_long_and_straight`). A pixel may belong
to more than one region where contrails cross; only seeds are skipped once claimed.

This is the region-growing step of the Mannstein-style method (Mannstein, Meyer & Wendling,
1999, Int. J. Remote Sensing 20(8), 1641-1660, doi:10.1080/014311699212650).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from deepcontrail.instance._shape import _ols, is_long_and_straight

# 8-neighbour offsets examined in this exact order for every dequeued pixel.
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _fit_window(rows: list, cols: list, lookback: int) -> tuple[int, float, float]:
    """Local direction of the last ``lookback`` region pixels as ``(branch, slope, intercept)``.

    ``branch`` 0 fits rows on cols (columns vary more), ``branch`` 1 fits cols on rows. The
    fit is the closed-form least squares of :func:`deepcontrail.instance._shape._ols`.
    """
    local_rows = np.asarray(rows[-lookback:], dtype=float)
    local_cols = np.asarray(cols[-lookback:], dtype=float)
    if np.std(local_cols) > np.std(local_rows):
        _, slope, intercept = _ols(local_cols, local_rows)
        return 0, slope, intercept
    _, slope, intercept = _ols(local_rows, local_cols)
    return 1, slope, intercept


def pixel_aligns(
    rows: list,
    cols: list,
    new_row: int,
    new_col: int,
    lookback: int,
    threshold: float,
) -> bool:
    """Whether ``new_row, new_col`` continues the local direction of ``rows, cols``.

    Fewer than five accumulated pixels are always accepted (bootstrap). Otherwise the last
    ``lookback`` pixels set the local direction: if their columns vary more than their rows,
    rows are regressed on columns and the candidate is accepted when its row is within
    ``threshold`` of the prediction, and the other way round otherwise.

    Parameters
    ----------
    rows
        Row coordinates accumulated so far, in acceptance order.
    cols
        Column coordinates accumulated so far, in acceptance order.
    new_row
        Row coordinate of the candidate pixel.
    new_col
        Column coordinate of the candidate pixel.
    lookback
        Trailing window used for the local regression.
    threshold
        Maximum allowed deviation (in pixels) for alignment.

    Returns
    -------
    bool
        True if the pixel is locally aligned with the region.
    """
    if len(rows) < 5:
        return True
    branch, slope, intercept = _fit_window(rows, cols, lookback)
    if branch == 0:
        return abs(slope * new_col + intercept - new_row) < threshold
    return abs(slope * new_row + intercept - new_col) < threshold


def grow_regions(
    binary_mask: np.ndarray,
    min_pixels: int,
    min_length: float,
    min_r_score: float,
    lookback: int,
    alignment_threshold: float,
    avg_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grow aligned regions out of a binary mask and keep the straight, long ones.

    Parameters
    ----------
    binary_mask
        2D boolean or ``{0, 1}`` array of shape ``(H, W)``.
    min_pixels
        Minimum pixel count to keep a region.
    min_length
        Minimum region length for the linearity check.
    min_r_score
        Straightness threshold, squared before the comparison.
    lookback
        Trailing window used for the local alignment regression.
    alignment_threshold
        Maximum allowed deviation for a neighbour to join a growing region.
    avg_size
        Endpoint averaging window for the length check.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``(rows, cols)`` coordinate arrays of the kept regions, in seed order and pixels in
        acceptance order.
    """
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    contrails: list[tuple[np.ndarray, np.ndarray]] = []
    seed_rows, seed_cols = np.where(binary_mask > 0)

    for seed_r, seed_c in zip(seed_rows, seed_cols, strict=True):
        if visited[seed_r, seed_c]:
            continue

        region = {(seed_r, seed_c)}
        blob_r = [seed_r]
        blob_c = [seed_c]
        queue: deque[tuple[int, int]] = deque(((seed_r, seed_c),))
        # cached local direction; recomputed only when the region last changed (see ``dirty``)
        fit = _fit_window(blob_r, blob_c, lookback)
        dirty = True

        region_add = region.add
        blob_r_append = blob_r.append
        blob_c_append = blob_c.append
        queue_push = queue.append

        while queue:
            cur_r, cur_c = queue.popleft()
            for d_r, d_c in _OFFSETS:
                n_r = cur_r + d_r
                n_c = cur_c + d_c
                if not (0 <= n_r < h and 0 <= n_c < w):
                    continue
                if not binary_mask[n_r, n_c]:
                    continue
                if (n_r, n_c) in region:
                    continue
                if len(blob_r) >= 5:
                    if dirty:
                        fit = _fit_window(blob_r, blob_c, lookback)
                        dirty = False
                    branch, slope, intercept = fit
                    if branch == 0:
                        aligned = abs(slope * n_c + intercept - n_r) < alignment_threshold
                    else:
                        aligned = abs(slope * n_r + intercept - n_c) < alignment_threshold
                    if not aligned:
                        continue
                region_add((n_r, n_c))
                blob_r_append(n_r)
                blob_c_append(n_c)
                queue_push((n_r, n_c))
                dirty = True

        rows_arr = np.array(blob_r)
        cols_arr = np.array(blob_c)
        for r, c in region:
            visited[r, c] = True

        if len(blob_r) < min_pixels:
            continue
        if is_long_and_straight(rows_arr, cols_arr, min_r_score, min_length, avg_size):
            contrails.append((rows_arr, cols_arr))

    return contrails
