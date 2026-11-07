"""DeepContrail dataset format, version 1: the metadata every dataset root carries.

A dataset is any directory that follows the format; the OpenContrails converter is one
producer. Two JSON files describe it:

- ``spectral_bands/bands.json`` (``BandManifest``): the channel order of every
  ``spectral_bands/{split}/<id>.npy`` file, with the key configs use for each channel.
- ``metadata/dataset_info.json`` (``DatasetInfo``): where the data comes from, how many
  labelled scenes each split holds, which labels exist and the recommended pixel shift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FORMAT_VERSION = 1
BAND_MANIFEST_PATH = "spectral_bands/bands.json"
DATASET_INFO_PATH = "metadata/dataset_info.json"

_Schema = TypeVar("_Schema", bound=BaseModel)

Label = Literal["semantic", "probability", "annotator", "instance"]
"""A label directory: ``semantic_masks/``, ``probability_masks/``, ``annotator_masks/``,
``instance_masks/``."""


class DatasetContractError(ValueError):
    """A dataset root does not follow the DeepContrail dataset format."""


class BandChannel(BaseModel):
    """One channel (last axis) of the ``spectral_bands/`` arrays.

    Parameters
    ----------
    index
        0-based position on the last axis.
    key
        Name configs use to refer to the channel (``[a-z0-9_]+``).
    units
        Physical units, e.g. ``"K"`` for brightness temperatures.
    sensor_band
        The sensor's own band ID (``"13"`` for ABI, ``"IR10.5"`` for FCI). Never
        interpreted by the library; it lets people check a recipe.
    wavelength_um
        Central wavelength in µm.
    name
        Human-readable description.
    derived_from
        ``(a, b)`` when the channel is stored as ``a - b`` of two other channels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: Annotated[int, Field(ge=0)]
    key: Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]
    units: str
    sensor_band: str | None = None
    wavelength_um: float | None = None
    name: str | None = None
    derived_from: tuple[str, str] | None = None


class BandManifest(BaseModel):
    """``spectral_bands/bands.json``: what each channel of the spectral arrays holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Literal[1]
    sensor: str
    dtype: Literal["float32"]
    channels: Annotated[tuple[BandChannel, ...], Field(min_length=1)]
    description: str | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        """Channel keys in channel order."""
        return tuple(c.key for c in self.channels)

    def index_of(self, key: str) -> int:
        """Return the channel index of ``key``.

        Raises
        ------
        DatasetContractError
            If no channel has that key; the message lists the available keys.
        """
        try:
            return self.keys.index(key)
        except ValueError:
            msg = f"unknown band key {key!r}; available: {', '.join(self.keys)}"
            raise DatasetContractError(msg) from None

    @model_validator(mode="after")
    def _validate_channels(self) -> Self:
        keys = [c.key for c in self.channels]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            msg = f"duplicate channel keys: {duplicates}"
            raise ValueError(msg)
        for position, channel in enumerate(self.channels):
            if channel.index != position:
                msg = f"channel {channel.key!r} has index {channel.index}, expected {position}"
                raise ValueError(msg)
            unknown = [k for k in channel.derived_from or () if k not in keys]
            if unknown:
                msg = f"channel {channel.key!r}: derived_from refers to unknown keys {unknown}"
                raise ValueError(msg)
        return self


class DatasetInfo(BaseModel):
    """``metadata/dataset_info.json``: provenance, record counts, labels and pixel shift.

    Parameters
    ----------
    source
        What produced the data.
    records
        Number of labelled scenes per split.
    labels
        Label directories present in the dataset.
    recommended_pixel_shift
        Shift in pixels (right and down) that aligns the images with the masks;
        ``0.0`` when they are aligned, ``None`` when unknown.
    temporal
        Whether the dataset stores context frames, ``<id>_b<k>`` / ``<id>_a<k>`` next to scene
        ``<id>`` (see :func:`deepcontrail.core.splits.context_frames`); ``None`` means no.
    export, naming, pixel_shift_note
        Provenance notes for people; not read by the library.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format_version: Literal[1]
    source: str
    records: dict[Literal["train", "test"], Annotated[int, Field(ge=0)]]
    labels: tuple[Label, ...]
    recommended_pixel_shift: Annotated[float, Field(ge=0.0)] | None
    temporal: bool | None = None
    export: str | None = None
    naming: str | None = None
    pixel_shift_note: str | None = None


def _read(root: str | Path, relative: str, schema: type[_Schema]) -> _Schema:
    path = Path(root) / relative
    if not path.is_file():
        msg = f"{path}: missing; every dataset root needs {relative} (DeepContrail format v1)"
        raise DatasetContractError(msg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        msg = f"{path}: must be UTF-8 encoded ({exc})"
        raise DatasetContractError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"{path}: not valid JSON ({exc})"
        raise DatasetContractError(msg) from exc
    if isinstance(payload, dict) and "format_version" not in payload:
        msg = (
            f"{path}: written before DeepContrail format v1 (no format_version). Regenerate the "
            "metadata; for OpenContrails data: deepcontrail-convert-opencontrails "
            "--metadata-only --output-dir <root>"
        )
        raise DatasetContractError(msg)
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        msg = f"{path}: does not follow DeepContrail format v1:\n{exc}"
        raise DatasetContractError(msg) from exc


def read_band_manifest(root: str | Path) -> BandManifest:
    """Read and validate ``<root>/spectral_bands/bands.json``.

    Raises
    ------
    DatasetContractError
        If the file is missing, is not JSON, or does not follow format v1.
    """
    return _read(root, BAND_MANIFEST_PATH, BandManifest)


def read_dataset_info(root: str | Path) -> DatasetInfo:
    """Read and validate ``<root>/metadata/dataset_info.json``.

    Raises
    ------
    DatasetContractError
        If the file is missing, is not JSON, or does not follow format v1.
    """
    return _read(root, DATASET_INFO_PATH, DatasetInfo)
