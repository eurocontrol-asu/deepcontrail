"""Tests for deepcontrail.opencontrails.bands — the OpenContrails band table."""

from __future__ import annotations


def test_spectral_band_keys_match_converter_channel_order():
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_KEYS

    expected = (
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
    )

    assert expected == SPECTRAL_BAND_KEYS, "channel order of spectral_bands/*.npy files on disk"


# --- SPECTRAL_BAND_INFO ---

_VERIFIED_ABI_BANDS = {
    "upper_vapor": 8,
    "mid_vapor": 9,
    "low_vapor": 10,
    "cloud_top": 11,
    "ozone": 12,
    "data_10um": 13,
    "data_11um": 14,
    "data_12um": 15,
    "co2": 16,
}


def test_spectral_band_info_describes_every_key():
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_INFO, SPECTRAL_BAND_KEYS

    assert set(SPECTRAL_BAND_INFO) == set(SPECTRAL_BAND_KEYS), "every channel needs a description"


def test_spectral_band_info_maps_keys_to_verified_abi_bands():
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_INFO

    got = {k: v.abi_band for k, v in SPECTRAL_BAND_INFO.items() if v.abi_band is not None}

    assert got == _VERIFIED_ABI_BANDS, (
        "mapping verified against Kaggle band_XX files: data_10um is band 13, cloud_top band 11"
    )


def test_spectral_band_info_btd_is_derived_not_an_abi_band():
    from deepcontrail.opencontrails.bands import SPECTRAL_BAND_INFO

    btd = SPECTRAL_BAND_INFO["brightness_temperature_difference"]

    assert btd.abi_band is None
    assert btd.wavelength_um is None
    assert btd.derived_from == ("data_11um", "data_12um"), "BTD = band 14 - band 15"
