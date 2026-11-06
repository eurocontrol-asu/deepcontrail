"""Least-squares line fit and the shape tests a growing region must pass.

Private module (leading underscore); not exported from ``deepcontrail.instance``.

The fit and the direction-independent straightness follow the Mannstein-style detection of
Mannstein, Meyer & Wendling (1999), "Operational detection of contrails from NOAA-AVHRR
data", Int. J. Remote Sensing 20(8), 1641-1660, doi:10.1080/014311699212650: a candidate
region is kept only when it is both straight and long.
"""

from __future__ import annotations

import numpy as np


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Closed-form ordinary least squares fit returning (R², slope, intercept).

    Reproduces scikit-learn ``LinearRegression`` exactly, including its
    degenerate cases: when the input variance is zero the slope is ``0.0`` and
    the intercept is the mean of ``y``; when the total sum of squares is zero
    the squared R is ``1.0`` if the residual sum of squares is zero, else
    ``0.0``.

    Results agree with scikit-learn to rounding error, so a comparison exactly at a
    threshold can go either way: with a whole-number ``alignment_threshold`` (e.g. 2.0) a
    pixel exactly that far from the fitted line may be accepted by one and rejected by the
    other. Non-integer thresholds avoid such ties.

    Parameters
    ----------
    x
        Independent-variable coordinates.
    y
        Dependent-variable coordinates.

    Returns
    -------
    tuple[float, float, float]
        ``(r_squared, slope, intercept)``.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_mean = float(x_arr.mean())
    y_mean = float(y_arr.mean())
    dx = x_arr - x_mean
    dy = y_arr - y_mean
    denom = float((dx * dx).sum())
    if denom == 0.0:
        slope = 0.0
        intercept = y_mean
    else:
        slope = float((dx * dy).sum() / denom)
        intercept = y_mean - slope * x_mean
    residuals = y_arr - (slope * x_arr + intercept)
    rss = float((residuals * residuals).sum())
    tss = float((dy * dy).sum())
    r_squared = (1.0 if rss == 0.0 else 0.0) if tss == 0.0 else 1.0 - rss / tss
    return r_squared, slope, intercept


def linear_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Fit a linear regression and return (R², slope, intercept).

    Parameters
    ----------
    xs
        Independent-variable coordinates.
    ys
        Dependent-variable coordinates.

    Returns
    -------
    tuple[float, float, float]
        ``(r_squared, slope, intercept)`` from closed-form ordinary least
        squares.
    """
    return _ols(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))


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
    min_r_score: float,
    min_length: float,
    avg_size: int,
) -> bool:
    """Keep a region only if it is straight enough and long enough.

    Compares the region's :func:`straightness` with ``min_r_score²`` (so ``min_r_score`` reads
    like a correlation threshold). The length is measured between the means of the
    ``avg_size`` extreme coordinates, projected on a line fitted along the dominant axis, and
    compared with ``min_length``.

    Parameters
    ----------
    rows
        Row coordinates of the region pixels.
    cols
        Column coordinates of the region pixels.
    min_r_score
        Straightness threshold, squared before the comparison.
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

    if len(rows_arr) < avg_size * 2:
        return False

    sorted_rows = np.argsort(rows_arr)
    y1 = float(np.mean(rows_arr[sorted_rows][:avg_size]))
    y2 = float(np.mean(rows_arr[sorted_rows][-avg_size:]))
    sorted_cols = np.argsort(cols_arr)
    x1 = float(np.mean(cols_arr[sorted_cols][:avg_size]))
    x2 = float(np.mean(cols_arr[sorted_cols][-avg_size:]))

    if abs(x1 - x2) > abs(y1 - y2):
        _, slope, intercept = linear_fit(xs=cols_arr, ys=rows_arr)
        y1, y2 = x1 * slope + intercept, x2 * slope + intercept
    else:
        _, slope, intercept = linear_fit(xs=rows_arr, ys=cols_arr)
        x1, x2 = y1 * slope + intercept, y2 * slope + intercept

    pixel_length = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    straight = straightness(rows_arr, cols_arr)
    return bool(straight > np.square(min_r_score) and pixel_length > min_length)


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
