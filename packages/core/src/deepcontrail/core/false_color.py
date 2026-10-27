"""False-colour composites built from named spectral bands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from deepcontrail.core.bands import SPECTRAL_BAND_KEYS, SpectralBand

if TYPE_CHECKING:
    from pathlib import Path


class FalseColorChannel(BaseModel):
    """One composite channel: a band, or the difference of two bands, scaled by ``bounds``.

    ``bands=(a,)`` gives ``a``; ``bands=(a, b)`` gives ``a - b``. The value is
    mapped linearly from ``bounds = (lo, hi)`` to ``[0, 1]`` and clipped.
    """

    model_config = ConfigDict(frozen=True)

    bands: tuple[SpectralBand] | tuple[SpectralBand, SpectralBand]
    bounds: tuple[float, float]

    @model_validator(mode="after")
    def _validate_bounds(self) -> FalseColorChannel:
        lo, hi = self.bounds
        if not lo < hi:
            msg = f"bounds must be increasing (lo < hi), got {self.bounds}"
            raise ValueError(msg)
        return self


class FalseColorConfig(BaseModel):
    """Bands and bounds of the 3-channel composite built from ``spectral_bands/``.

    The default is the recipe every model before this option was trained on:
    the ash RGB layout but with ``data_10um`` (ABI band 13) in green. The
    standard ash RGB uses ``cloud_top`` (ABI band 11) there instead.
    """

    model_config = ConfigDict(frozen=True)

    channels: tuple[FalseColorChannel, FalseColorChannel, FalseColorChannel] = (
        FalseColorChannel(bands=("data_12um", "data_11um"), bounds=(-4.0, 2.0)),
        FalseColorChannel(bands=("data_11um", "data_10um"), bounds=(-4.0, 5.0)),
        FalseColorChannel(bands=("data_11um",), bounds=(243.0, 303.0)),
    )


_BAND_INDEX = {key: i for i, key in enumerate(SPECTRAL_BAND_KEYS)}


def false_color_from_spectral(bands: np.ndarray, config: FalseColorConfig) -> np.ndarray:
    """Build the 3-channel composite from a 10-channel spectral array.

    Parameters
    ----------
    bands
        Brightness temperatures with shape ``(H, W, 10)``, channels in
        ``SPECTRAL_BAND_KEYS`` order (the ``spectral_bands/*.npy`` layout).
    config
        Bands and bounds of each output channel.

    Returns
    -------
    np.ndarray
        Float32 ``(H, W, 3)`` array with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``bands`` is not ``(H, W, 10)``.
    """
    if bands.ndim != 3 or bands.shape[-1] != len(SPECTRAL_BAND_KEYS):
        n = len(SPECTRAL_BAND_KEYS)
        msg = f"expected spectral bands of shape (H, W, {n}), got {bands.shape}"
        raise ValueError(msg)

    channels = []
    for channel in config.channels:
        value = bands[..., _BAND_INDEX[channel.bands[0]]]
        if len(channel.bands) == 2:
            value = value - bands[..., _BAND_INDEX[channel.bands[1]]]
        lo, hi = channel.bounds
        channels.append((value - lo) / (hi - lo))
    return np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)


def load_spectral_false_color(path: str | Path, config: FalseColorConfig) -> np.ndarray:
    """Load a ``spectral_bands/*.npy`` file and build its composite with ``config``."""
    return false_color_from_spectral(np.load(path), config)
