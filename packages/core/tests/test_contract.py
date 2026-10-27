"""Tests for deepcontrail.core.contract — dataset format v1 metadata schemas and readers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _channel(index: int, key: str, **extra: object) -> dict[str, Any]:
    return {"index": index, "key": key, "units": "K", **extra}


def _manifest(**overrides: object) -> dict[str, Any]:
    return {
        "format_version": 1,
        "sensor": "GOES-16 ABI",
        "dtype": "float32",
        "channels": [
            _channel(0, "data_11um", sensor_band="14", wavelength_um=11.19),
            _channel(1, "data_12um", sensor_band="15"),
            _channel(2, "btd", derived_from=["data_11um", "data_12um"]),
        ],
    } | overrides


def _info(**overrides: object) -> dict[str, Any]:
    return {
        "format_version": 1,
        "source": "OpenContrails GOES-16 ABI TFRecords",
        "records": {"train": 3, "test": 1},
        "labels": ["semantic", "probability"],
        "recommended_pixel_shift": 0.0,
    } | overrides


def _write(root: Path, relative: str, payload: dict[str, Any]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# --- BandManifest ---------------------------------------------------------------


def test_band_manifest_accepts_a_valid_manifest():
    from deepcontrail.core.contract import BandManifest

    manifest = BandManifest.model_validate(_manifest())

    assert [c.key for c in manifest.channels] == ["data_11um", "data_12um", "btd"]


def test_band_manifest_accepts_a_non_goes_sensor():
    from deepcontrail.core.contract import BandManifest

    fci = _manifest(
        sensor="MTG-I1 FCI",
        channels=[
            _channel(0, "ir_87", sensor_band="IR8.7"),
            _channel(1, "ir_105", sensor_band="IR10.5"),
            _channel(2, "ir_123", sensor_band="IR12.3"),
        ],
    )

    manifest = BandManifest.model_validate(fci)

    assert manifest.sensor == "MTG-I1 FCI", "any sensor declares its own band keys"


def test_band_manifest_rejects_duplicate_keys():
    from deepcontrail.core.contract import BandManifest

    payload = _manifest(channels=[_channel(0, "ir_105"), _channel(1, "ir_105")])

    with pytest.raises(ValueError, match="duplicate"):
        BandManifest.model_validate(payload)


def test_band_manifest_rejects_index_not_matching_position():
    from deepcontrail.core.contract import BandManifest

    payload = _manifest(channels=[_channel(0, "a"), _channel(2, "b")])

    with pytest.raises(ValueError, match="index"):
        BandManifest.model_validate(payload)


@pytest.mark.parametrize("key", ["Data11", "data-11", "data 11", ""])
def test_band_manifest_rejects_keys_that_are_not_snake_case(key: str):
    from deepcontrail.core.contract import BandManifest

    with pytest.raises(ValueError, match="key"):
        BandManifest.model_validate(_manifest(channels=[_channel(0, key)]))


def test_band_manifest_rejects_derived_from_unknown_key():
    from deepcontrail.core.contract import BandManifest

    payload = _manifest(channels=[_channel(0, "btd", derived_from=["a", "b"])])

    with pytest.raises(ValueError, match="derived_from"):
        BandManifest.model_validate(payload)


def test_band_manifest_rejects_another_format_version():
    from deepcontrail.core.contract import BandManifest

    with pytest.raises(ValueError, match="format_version"):
        BandManifest.model_validate(_manifest(format_version=2))


def test_band_manifest_rejects_unknown_fields():
    from deepcontrail.core.contract import BandManifest

    payload = _manifest(channels=[_channel(0, "a", abi_band=13)])

    with pytest.raises(ValueError, match="abi_band"):
        BandManifest.model_validate(payload)


# --- DatasetInfo ------------------------------------------------------------------


def test_dataset_info_accepts_valid_metadata():
    from deepcontrail.core.contract import DatasetInfo

    info = DatasetInfo.model_validate(_info(export="20250305", temporal=False))

    assert info.records == {"train": 3, "test": 1}
    assert info.recommended_pixel_shift == 0.0


def test_dataset_info_allows_an_unknown_pixel_shift():
    from deepcontrail.core.contract import DatasetInfo

    info = DatasetInfo.model_validate(_info(recommended_pixel_shift=None))

    assert info.recommended_pixel_shift is None, "null means: measure the alignment yourself"


def test_dataset_info_rejects_negative_record_counts():
    from deepcontrail.core.contract import DatasetInfo

    with pytest.raises(ValueError, match="records"):
        DatasetInfo.model_validate(_info(records={"train": -1, "test": 0}))


def test_dataset_info_rejects_unknown_label_kind():
    from deepcontrail.core.contract import DatasetInfo

    with pytest.raises(ValueError, match="labels"):
        DatasetInfo.model_validate(_info(labels=["semantic", "bounding_boxes"]))


def test_dataset_info_requires_the_pixel_shift_field():
    from deepcontrail.core.contract import DatasetInfo

    payload = _info()
    del payload["recommended_pixel_shift"]

    with pytest.raises(ValueError, match="recommended_pixel_shift"):
        DatasetInfo.model_validate(payload)


# --- readers ------------------------------------------------------------------------


def test_read_band_manifest_reads_spectral_bands_json(tmp_path: Path):
    from deepcontrail.core.contract import read_band_manifest

    _write(tmp_path, "spectral_bands/bands.json", _manifest())

    manifest = read_band_manifest(tmp_path)

    assert manifest.sensor == "GOES-16 ABI"


def test_read_dataset_info_reads_metadata_json(tmp_path: Path):
    from deepcontrail.core.contract import read_dataset_info

    _write(tmp_path, "metadata/dataset_info.json", _info())

    info = read_dataset_info(tmp_path)

    assert info.labels == ("semantic", "probability")


def test_read_band_manifest_missing_file_names_the_path(tmp_path: Path):
    from deepcontrail.core.contract import DatasetContractError, read_band_manifest

    with pytest.raises(DatasetContractError, match=r"spectral_bands/bands\.json"):
        read_band_manifest(tmp_path)


def test_read_dataset_info_invalid_json_names_the_path(tmp_path: Path):
    from deepcontrail.core.contract import DatasetContractError, read_dataset_info

    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "dataset_info.json").write_text("{not json")

    with pytest.raises(DatasetContractError, match=r"dataset_info\.json"):
        read_dataset_info(tmp_path)


def test_read_band_manifest_without_format_version_explains_how_to_upgrade(tmp_path: Path):
    from deepcontrail.core.contract import DatasetContractError, read_band_manifest

    legacy = _manifest()
    del legacy["format_version"]
    _write(tmp_path, "spectral_bands/bands.json", legacy)

    with pytest.raises(DatasetContractError, match="--metadata-only"):
        read_band_manifest(tmp_path)


def test_dataset_contract_error_is_a_value_error():
    from deepcontrail.core.contract import DatasetContractError

    assert issubclass(DatasetContractError, ValueError)


# --- band lookup ----------------------------------------------------------------------


def test_band_manifest_index_of_returns_the_channel_position():
    from deepcontrail.core.contract import BandManifest

    manifest = BandManifest.model_validate(_manifest())

    assert manifest.index_of("data_12um") == 1
    assert manifest.keys == ("data_11um", "data_12um", "btd")


def test_band_manifest_index_of_unknown_key_lists_available_keys():
    from deepcontrail.core.contract import BandManifest, DatasetContractError

    manifest = BandManifest.model_validate(_manifest())

    with pytest.raises(DatasetContractError, match=r"cloud_top.*data_11um, data_12um, btd"):
        manifest.index_of("cloud_top")
