"""The shape tests a growing region must pass: straightness, length and quality score.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

The direction-independent straightness and the length test follow the Mannstein-style detection of
Mannstein, Meyer & Wendling (1999), "Operational detection of contrails from NOAA-AVHRR
data", Int. J. Remote Sensing 20(8), 1641-1660, doi:10.1080/014311699212650: a candidate
region is kept only when it is both straight and long.
"""

from __future__ import annotations

import numpy as np


def _dominant_fit(
    sum_cross: int, sum_dep: int, sum_indep: int, sum_indep_sq: int, n: int
) -> tuple[float, float]:
    """Least-squares slope and intercept of ``dep`` on ``indep`` from exact integer sums.

    The sums are over the whole region: ``sum_cross`` is Σ(dep·indep), ``sum_dep`` is Σ(dep),
    ``sum_indep`` is Σ(indep) and ``sum_indep_sq`` is Σ(indep²). With no spread in the
    independent variable the slope is ``0.0`` and the intercept is the mean of the dependent
    variable.

    Parameters
    ----------
    sum_cross
        Sum of the products of the dependent and independent coordinates.
    sum_dep
        Sum of the dependent coordinates.
    sum_indep
        Sum of the independent coordinates.
    sum_indep_sq
        Sum of the squares of the independent coordinates.
    n
        Number of coordinates summed.

    Returns
    -------
    tuple[float, float]
        ``(slope, intercept)``.
    """
    denom = n * sum_indep_sq - sum_indep * sum_indep
    if denom == 0:
        return 0.0, sum_dep / n
    slope = (n * sum_cross - sum_indep * sum_dep) / denom
    return slope, (sum_dep - slope * sum_indep) / n


def straightness(rows: np.ndarray, cols: np.ndarray) -> float:
    """How straight a set of pixels is, whatever its direction: ((λ1 - λ2) / (λ1 + λ2))².

    λ1 ≥ λ2 are the eigenvalues of the 2x2 covariance of the pixel coordinates. A line scores 1
    and a round blob 0. For pixels along the 45° diagonal this is the least-squares R² of rows
    against columns; it is also the largest such R² over all rotations of the pixels, so unlike
    that R² it does not drop to 0 for a straight contrail along an image axis.

    Parameters
    ----------
    rows
        Row coordinates of the pixels.
    cols
        Column coordinates of the pixels.

    Returns
    -------
    float
        Straightness in ``[0, 1]``; ``0.0`` when all pixels coincide.
    """
    coords = np.stack([np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)])
    coords -= coords.mean(axis=1, keepdims=True)
    var_r, var_c = float((coords[0] ** 2).mean()), float((coords[1] ** 2).mean())
    cov = float((coords[0] * coords[1]).mean())
    total = var_r + var_c  # λ1 + λ2
    if total == 0.0:
        return 0.0
    gap = float(np.hypot(var_r - var_c, 2.0 * cov))  # λ1 - λ2
    return min((gap / total) ** 2, 1.0)


def is_long_and_straight(
    rows: np.ndarray,
    cols: np.ndarray,
    min_straightness: float,
    min_length: float,
    avg_size: int,
) -> bool:
    """Keep a region only if it is straight enough and long enough.

    Compares the region's :func:`straightness` with ``min_straightness``. The length runs between
    the means of the ``avg_size`` extreme coordinates on each axis, projected onto a line fitted
    along the more-spread axis from the exact integer sums of the whole region, and is compared
    with ``min_length``.

    Parameters
    ----------
    rows
        Row coordinates of the region pixels.
    cols
        Column coordinates of the region pixels.
    min_straightness
        Minimum straightness, in [0, 1].
    min_length
        Minimum contrail pixel length.
    avg_size
        Number of extreme endpoints averaged per axis.

    Returns
    -------
    bool
        True if the region is both linear and long enough.
    """
    rows_arr = np.asarray(rows, dtype=float)
    cols_arr = np.asarray(cols, dtype=float)
    n = len(rows_arr)

    if n < avg_size * 2:
        return False

    # Endpoint means: the avg_size smallest and largest coordinates on each axis. The
    # coordinates are integers, so partitioning for the extremes and averaging gives the same
    # values as a full sort would.
    r_lo, r_hi = _axis_endpoints(rows_arr, n, avg_size)
    c_lo, c_hi = _axis_endpoints(cols_arr, n, avg_size)

    # Exact integer sums of the whole region for the dominant-axis fit.
    r_int = rows_arr.astype(np.int64)
    c_int = cols_arr.astype(np.int64)
    sum_r = int(r_int.sum())
    sum_c = int(c_int.sum())
    sum_r2 = int((r_int * r_int).sum())
    sum_c2 = int((c_int * c_int).sum())
    sum_rc = int((r_int * c_int).sum())

    if abs(c_lo - c_hi) > abs(r_lo - r_hi):
        # Columns spread more: predict rows from columns.
        slope, intercept = _dominant_fit(sum_rc, sum_r, sum_c, sum_c2, n)
        r_lo, r_hi = c_lo * slope + intercept, c_hi * slope + intercept
    else:
        # Rows spread more: predict columns from rows.
        slope, intercept = _dominant_fit(sum_rc, sum_c, sum_r, sum_r2, n)
        c_lo, c_hi = r_lo * slope + intercept, r_hi * slope + intercept

    length = float(np.hypot(c_lo - c_hi, r_lo - r_hi))
    linearity = straightness(rows_arr, cols_arr)
    return bool(linearity > min_straightness and length > min_length)


def _axis_endpoints(values: np.ndarray, n: int, avg_size: int) -> tuple[float, float]:
    """Means of the ``avg_size`` smallest and largest coordinates along one axis."""
    lo = np.partition(values, avg_size - 1)[:avg_size]
    hi = np.partition(values, n - avg_size)[n - avg_size :]
    return float(lo.mean()), float(hi.mean())


def candidate_score(
    rows: np.ndarray,
    cols: np.ndarray,
    saturation: float,
    avg_size: int,
) -> float:
    """Quality score of a contrail: straightness * tanh(n_pixels / saturation).

    Returns ``0.0`` when the blob has fewer than ``2 * avg_size`` pixels.

    Parameters
    ----------
    rows
        Row coordinates of the blob pixels.
    cols
        Column coordinates of the blob pixels.
    saturation
        Pixel count at which the size term approaches 1.
    avg_size
        Number of end pixels averaged when measuring the length (sets the minimum size).

    Returns
    -------
    float
        Quality score in ``[0, 1]``.
    """
    rows_arr = np.asarray(rows)
    cols_arr = np.asarray(cols)

    if len(rows_arr) < 2 * avg_size:
        return 0.0

    n_pixels = float(len(rows_arr))
    return float(straightness(rows_arr, cols_arr) * np.tanh(n_pixels / saturation))
