"""OpenContrails spectral band keys and what each one holds.

Channel order of every ``spectral_bands/{split}/{stem}.npy`` file written by the converter.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, get_args

SpectralBand = Literal[
    "data_10um",
    "data_11um",
    "data_12um",
    "brightness_temperature_difference",
    "cloud_top",
    "co2",
    "low_vapor",
    "mid_vapor",
    "upper_vapor",
    "ozone",
]
"""OpenContrails band key.

``data_10um`` is ABI band 13 (10.3 µm); ``cloud_top`` is ABI band 11 (8.4 µm).
"""

SPECTRAL_BAND_KEYS: tuple[str, ...] = get_args(SpectralBand)
"""Channel order of every ``spectral_bands/{split}/{stem}.npy`` file, shape ``(H, W, 10)``."""


@dataclasses.dataclass(frozen=True)
class SpectralBandInfo:
    """What one OpenContrails band key holds.

    Parameters
    ----------
    abi_band
        GOES-16 ABI band number, or ``None`` for a derived channel.
    wavelength_um
        ABI central wavelength in µm, or ``None`` for a derived channel.
    name
        NOAA name of the band (or a description of the derived channel).
    derived_from
        ``(a, b)`` when the channel is ``a - b`` of two other keys.
    """

    abi_band: int | None
    wavelength_um: float | None
    name: str
    derived_from: tuple[str, str] | None = None


# Verified by exact match of OpenContrails TFRecord keys against Kaggle's labelled band_XX files.
SPECTRAL_BAND_INFO: dict[str, SpectralBandInfo] = {
    "data_10um": SpectralBandInfo(13, 10.33, "Clean IR longwave window"),
    "data_11um": SpectralBandInfo(14, 11.19, "IR longwave window"),
    "data_12um": SpectralBandInfo(15, 12.27, "Dirty IR longwave window"),
    "brightness_temperature_difference": SpectralBandInfo(
        None, None, "Split-window difference", derived_from=("data_11um", "data_12um")
    ),
    "cloud_top": SpectralBandInfo(11, 8.44, "Cloud-top phase"),
    "co2": SpectralBandInfo(16, 13.27, "CO2 longwave IR"),
    "low_vapor": SpectralBandInfo(10, 7.34, "Lower-level tropospheric water vapor"),
    "mid_vapor": SpectralBandInfo(9, 6.93, "Mid-level tropospheric water vapor"),
    "upper_vapor": SpectralBandInfo(8, 6.19, "Upper-level tropospheric water vapor"),
    "ozone": SpectralBandInfo(12, 9.61, "Ozone"),
}
"""Band number, wavelength and name of each ``SPECTRAL_BAND_KEYS`` entry."""
