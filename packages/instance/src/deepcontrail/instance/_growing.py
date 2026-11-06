"""Direction-following region growing that splits a binary mask into contrail regions.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

Seeds walk the mask in row-major order. From each seed not already claimed by an earlier
region, a breadth-first flood grows along 8-connected neighbours that stay aligned with the
region's local direction, until the region ends. Each region is then tested for minimum size
and linearity (:func:`deepcontrail.instance._shape.is_long_and_straight`). A pixel may belong
to more than one region where contrails cross; only seeds are skipped once claimed.

The mask is padded by one background pixel on every side and worked on as flat indices, so the
eight neighbours are constant offsets with no bounds check. Region membership is an int32
"owner stamp" per pixel, and the local alignment direction is kept as exact integer running
sums over the trailing ``lookback`` window, updated in constant time as the region grows.

This is the region-growing step of the Mannstein-style method (Mannstein, Meyer & Wendling,
1999, Int. J. Remote Sensing 20(8), 1641-1660, doi:10.1080/014311699212650).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from deepcontrail.instance._shape import is_long_and_straight

# 8-neighbour directions examined in this exact order for every dequeued pixel: (dr, dc).
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


class _Region:
    """A growing region: its accepted pixels and a running alignment-window fit.

    Pixels are stored as flat indices into the padded mask. The alignment window keeps exact
    integer running sums (n, Σr, Σc, Σr², Σc², Σrc) over the last ``lookback`` accepted pixels,
    so the local least-squares direction is recomputed in constant time when the region grows.
    """

    __slots__ = (
        "branch",
        "dirty",
        "flat",
        "intercept",
        "lookback",
        "size",
        "slope",
        "sum_c",
        "sum_c2",
        "sum_r",
        "sum_r2",
        "sum_rc",
        "width",
        "window",
        "window_n",
    )

    def __init__(self, flat0: int, row0: int, col0: int, lookback: int, width: int) -> None:
        self.flat = [flat0]
        self.size = 1
        self.sum_r = row0
        self.sum_c = col0
        self.sum_r2 = row0 * row0
        self.sum_c2 = col0 * col0
        self.sum_rc = row0 * col0
        self.window = deque((flat0,))
        self.window_n = 1
        self.branch = 0
        self.slope = 0.0
        self.intercept = 0.0
        self.dirty = True
        self.lookback = lookback
        self.width = width

    def add(self, flat: int, row: int, col: int) -> None:
        """Accept a pixel and fold it into the running alignment window."""
        self.flat.append(flat)
        self.size += 1
        self.window.append(flat)
        self.sum_r += row
        self.sum_c += col
        self.sum_r2 += row * row
        self.sum_c2 += col * col
        self.sum_rc += row * col
        self.window_n += 1
        if self.window_n > self.lookback:
            oldest = self.window.popleft()
            oldest_r = oldest // self.width - 1
            oldest_c = oldest % self.width - 1
            self.sum_r -= oldest_r
            self.sum_c -= oldest_c
            self.sum_r2 -= oldest_r * oldest_r
            self.sum_c2 -= oldest_c * oldest_c
            self.sum_rc -= oldest_r * oldest_c
            self.window_n -= 1
        self.dirty = True

    def refresh_fit(self) -> None:
        """Recompute the cached local direction from the running window sums.

        Branch 0 fits rows on columns (columns vary more), branch 1 fits columns on rows. With
        no spread in the regressor the slope is ``0.0`` and the intercept is the regressand mean.
        """
        n = self.window_n
        var_c = n * self.sum_c2 - self.sum_c * self.sum_c
        var_r = n * self.sum_r2 - self.sum_r * self.sum_r
        fit_rows_on_cols = var_c > var_r if var_c != var_r else self._tie_prefers_cols()
        if fit_rows_on_cols:
            self.branch = 0
            slope = (n * self.sum_rc - self.sum_r * self.sum_c) / var_c
            self.slope = slope
            self.intercept = (self.sum_r - slope * self.sum_c) / n
        else:
            self.branch = 1
            if var_r == 0:
                self.slope = 0.0
                self.intercept = self.sum_c / n
            else:
                slope = (n * self.sum_rc - self.sum_r * self.sum_c) / var_r
                self.slope = slope
                self.intercept = (self.sum_c - slope * self.sum_r) / n
        self.dirty = False

    def _tie_prefers_cols(self) -> bool:
        """Break an exact spread tie between the two axes.

        When the window's row and column spreads are exactly equal, either axis fits equally
        well. The choice follows the float comparison ``std(cols) > std(rows)`` over the window,
        so that results stay identical to the published v0.1.0 instance step and the score model
        trained on its output. Exact ties are rare, so this costs nothing measurable.
        """
        window = np.asarray(self.window, dtype=np.int64)
        rows = (window // self.width - 1).astype(float)
        cols = (window % self.width - 1).astype(float)
        return bool(np.std(cols) > np.std(rows))

    def accepts(self, row: int, col: int, threshold: float) -> bool:
        """Whether a candidate pixel continues the region's local direction.

        Fewer than five accumulated pixels are always accepted (bootstrap). Otherwise the cached
        local direction predicts the candidate's coordinate along the more-variant axis and it is
        accepted when the deviation is within ``threshold``.
        """
        if self.size < 5:
            return True
        if self.dirty:
            self.refresh_fit()
        if self.branch == 0:
            return abs(self.slope * col + self.intercept - row) < threshold
        return abs(self.slope * row + self.intercept - col) < threshold

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """(rows, cols) arrays of the accepted pixels in acceptance order."""
        flat = np.asarray(self.flat, dtype=np.int64)
        return flat // self.width - 1, flat % self.width - 1


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
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = binary_mask
    mask = padded.ravel()
    width = w + 2
    stamp = np.zeros(mask.size, dtype=np.int32)
    # (flat offset, dr, dc) per neighbour, in the required examination order.
    neighbors = tuple((dr * width + dc, dr, dc) for dr, dc in _OFFSETS)

    contrails: list[tuple[np.ndarray, np.ndarray]] = []
    region_num = 0
    for seed in np.nonzero(mask)[0]:
        seed = int(seed)
        if stamp[seed] != 0:
            continue
        region_num += 1
        region = _Region(seed, seed // width - 1, seed % width - 1, lookback, width)
        stamp[seed] = region_num
        queue: deque[int] = deque((seed,))

        while queue:
            cur = int(queue.popleft())
            cur_r = cur // width - 1
            cur_c = cur % width - 1
            for offset, d_r, d_c in neighbors:
                flat = cur + offset
                if not mask[flat]:
                    continue
                if stamp[flat] == region_num:
                    continue
                n_r = cur_r + d_r
                n_c = cur_c + d_c
                if not region.accepts(n_r, n_c, alignment_threshold):
                    continue
                stamp[flat] = region_num
                region.add(flat, n_r, n_c)
                queue.append(flat)

        if region.size < min_pixels:
            continue
        rows_arr, cols_arr = region.coords()
        if is_long_and_straight(rows_arr, cols_arr, min_r_score, min_length, avg_size):
            contrails.append((rows_arr, cols_arr))

    return contrails
