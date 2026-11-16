"""MTG/FCI band keys, the bands.json manifest, and the standard ash false-colour recipe.

The toolkit stores three channels per frame: ``[T12.3 - T10.5, T10.5 - T8.7, T10.5]``. This
module maps them to the physical brightness-temperature bands the model consumes and describes
them for the dataset's ``spectral_bands/bands.json``.
"""

from __future__ import annotations

import numpy as np

BAND_KEYS = ("ir_105", "ir_87", "ir_123_minus_ir_105")
"""Channel order of every ``spectral_bands/{split}/{stem}.npy`` file the converter writes."""


def bands_from_composite(composite: np.ndarray) -> np.ndarray:
    """Recover the physical bands from a toolkit composite array.

    Parameters
    ----------
    composite
        ``(H, W, 3)`` toolkit composite ``[T12.3 - T10.5, T10.5 - T8.7, T10.5]``.

    Returns
    -------
    np.ndarray
        ``float32`` ``(H, W, 3)`` array with the bands in :data:`BAND_KEYS` order:
        ``[T10.5, T8.7, T12.3 - T10.5]``. ``NaN`` propagates.

    Raises
    ------
    ValueError
        If ``composite`` is not three-dimensional with three channels.
    """
    arr = np.asarray(composite, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        msg = f"expected a (H, W, 3) composite, got shape {arr.shape}"
        raise ValueError(msg)
    btd = arr[..., 0]  # T12.3 - T10.5
    split_window = arr[..., 1]  # T10.5 - T8.7
    t105 = arr[..., 2]  # T10.5
    ir_105 = t105
    ir_87 = t105 - split_window
    ir_123_minus_ir_105 = btd
    return np.stack((ir_105, ir_87, ir_123_minus_ir_105), axis=-1).astype(np.float32)


def band_manifest() -> dict:
    """Return the content of ``spectral_bands/bands.json`` for an MTG/FCI dataset.

    Returns
    -------
    dict
        A ``BandManifest`` payload (``format_version`` 1, sensor ``"MTG-I1 FCI"``, ``dtype``
        ``"float32"``, ``nodata`` ``"nan"`` and three channels in :data:`BAND_KEYS` order).
    """
    return {
        "format_version": 1,
        "sensor": "MTG-I1 FCI",
        "dtype": "float32",
        "nodata": "nan",
        "channels": [
            {
                "index": 0,
                "key": "ir_105",
                "units": "K",
                "wavelength_um": 10.5,
                "name": "10.5 um brightness temperature (1 km, HRFI)",
            },
            {
                "index": 1,
                "key": "ir_87",
                "units": "K",
                "wavelength_um": 8.7,
                "name": "8.7 um brightness temperature (2 km, resampled to the grid)",
            },
            {
                "index": 2,
                "key": "ir_123_minus_ir_105",
                "units": "K",
                "wavelength_um": None,
                "name": (
                    "12.3 um minus 10.5 um brightness temperature difference "
                    "(both at 2 km, resampled to the grid)"
                ),
            },
        ],
    }


FALSE_COLOR = {
    "channels": [
        {"bands": ["ir_123_minus_ir_105"], "bounds": [-4.0, 2.0]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4.0, 5.0]},
        {"bands": ["ir_105"], "bounds": [243.0, 303.0]},
    ]
}
"""The standard ash composite (same bounds as the GOES-16 model) on these bands."""
