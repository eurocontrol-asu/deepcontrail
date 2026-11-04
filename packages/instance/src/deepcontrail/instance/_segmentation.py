"""Alignment-aware BFS extraction of contrail segments from a binary mask.

Private module (leading underscore); not exported from ``deepcontrail.instance``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from deepcontrail.instance._geometry import is_linear_and_long, is_pixel_aligned


def segment_contrails(
    binary_mask: np.ndarray,
    min_pixels: int,
    min_length: float,
    min_r_score: float,
    lookback: int,
    alignment_threshold: float,
    avg_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract contrail segments from a binary mask via alignment-aware BFS.

    Uses 8-connected flood fill with local alignment constraints. Each candidate
    blob is validated for minimum size and linearity.

    Parameters
    ----------
    binary_mask
        2D boolean or ``{0, 1}`` array of shape ``(H, W)``.
    min_pixels
        Minimum pixel count to keep a segment.
    min_length
        Minimum pixel length for the linearity check.
    min_r_score
        Straightness threshold, squared before the comparison.
    lookback
        Local regression window for alignment checking.
    alignment_threshold
        Maximum allowed deviation for pixel alignment.
    avg_size
        Endpoint averaging window for the linearity check.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of ``(rows, cols)`` coordinate arrays for valid contrail segments.
    """
    h, w = binary_mask.shape
    global_visited = np.zeros((h, w), dtype=bool)
    contrails: list[tuple[np.ndarray, np.ndarray]] = []
    candidate_y, candidate_x = np.where(binary_mask > 0)

    for seed_y, seed_x in zip(candidate_y, candidate_x, strict=True):
        if global_visited[seed_y, seed_x]:
            continue

        current_pixels = {(seed_y, seed_x)}
        blob_ys = [seed_y]
        blob_xs = [seed_x]
        queue = deque([(seed_y, seed_x)])

        while queue:
            cy, cx = queue.popleft()
            neighbors = [
                (cy - 1, cx - 1),
                (cy - 1, cx),
                (cy - 1, cx + 1),
                (cy, cx - 1),
                (cy, cx + 1),
                (cy + 1, cx - 1),
                (cy + 1, cx),
                (cy + 1, cx + 1),
            ]
            for ny, nx in neighbors:
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                if binary_mask[ny, nx] == 0:
                    continue
                if (ny, nx) in current_pixels:
                    continue
                if is_pixel_aligned(blob_ys, blob_xs, ny, nx, lookback, alignment_threshold):
                    current_pixels.add((ny, nx))
                    blob_ys.append(ny)
                    blob_xs.append(nx)
                    queue.append((ny, nx))

        ys_arr = np.array(blob_ys)
        xs_arr = np.array(blob_xs)
        for y, x in current_pixels:
            global_visited[y, x] = True

        if len(blob_ys) < min_pixels:
            continue
        if is_linear_and_long(ys_arr, xs_arr, min_r_score, min_length, avg_size):
            contrails.append((ys_arr, xs_arr))

    return contrails
