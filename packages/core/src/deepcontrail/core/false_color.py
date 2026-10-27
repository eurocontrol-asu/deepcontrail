"""False-colour composites built from spectral bands named in the dataset's manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepcontrail.core.contract import DatasetContractError

if TYPE_CHECKING:
    from pathlib import Path

    from deepcontrail.core.contract import BandManifest

BandKey = Annotated[str, Field(min_length=1)]


class FalseColorChannel(BaseModel):
    """One composite channel: a band, or the difference of two bands, scaled by ``bounds``.

    ``bands=(a,)`` gives ``a``; ``bands=(a, b)`` gives ``a - b``, where ``a`` and ``b``
    are keys of the dataset's ``spectral_bands/bands.json``. The value is mapped
    linearly from ``bounds = (lo, hi)`` to ``[0, 1]`` and clipped.
    """

    model_config = ConfigDict(frozen=True)

    bands: tuple[BandKey] | tuple[BandKey, BandKey]
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

    There is no default recipe: band keys belong to the dataset, so the recipe is
    always written in the config (e.g. the standard ash RGB in
    ``configs/experiment.yaml``).
    """

    model_config = ConfigDict(frozen=True)

    channels: tuple[FalseColorChannel, FalseColorChannel, FalseColorChannel]


def check_false_color(config: FalseColorConfig, manifest: BandManifest) -> None:
    """Check that every band of the recipe is a key of the manifest.

    Raises
    ------
    DatasetContractError
        Naming every unknown key and listing the available ones.
    """
    unknown = [k for channel in config.channels for k in channel.bands if k not in manifest.keys]
    if unknown:
        msg = (
            f"false_color uses band keys not in spectral_bands/bands.json: {unknown}; "
            f"available: {', '.join(manifest.keys)}"
        )
        raise DatasetContractError(msg)


def false_color_from_spectral(
    bands: np.ndarray, config: FalseColorConfig, manifest: BandManifest
) -> np.ndarray:
    """Build the 3-channel composite from a spectral array.

    Parameters
    ----------
    bands
        Spectral array with shape ``(H, W, C)``, channels in manifest order (the
        ``spectral_bands/*.npy`` layout).
    config
        Bands and bounds of each output channel.
    manifest
        The dataset's band manifest, mapping each key to its channel.

    Returns
    -------
    np.ndarray
        Float32 ``(H, W, 3)`` array with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``bands`` is not ``(H, W, C)`` with the manifest's ``C`` channels.
    DatasetContractError
        If the recipe uses a key the manifest does not declare.
    """
    n = len(manifest.channels)
    if bands.ndim != 3 or bands.shape[-1] != n:
        msg = f"expected spectral bands of shape (H, W, {n}) ({n} channels), got {bands.shape}"
        raise ValueError(msg)

    channels = []
    for channel in config.channels:
        value = bands[..., manifest.index_of(channel.bands[0])]
        if len(channel.bands) == 2:
            value = value - bands[..., manifest.index_of(channel.bands[1])]
        lo, hi = channel.bounds
        channels.append((value - lo) / (hi - lo))
    return np.clip(np.stack(channels, axis=-1), 0, 1).astype(np.float32)


def load_spectral_false_color(
    path: str | Path, config: FalseColorConfig, manifest: BandManifest
) -> np.ndarray:
    """Load a ``spectral_bands/*.npy`` file and build its composite with ``config``."""
    return false_color_from_spectral(np.load(path), config, manifest)
