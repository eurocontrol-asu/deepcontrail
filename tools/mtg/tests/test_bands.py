"""Tests for deepcontrail.mtg.bands — MTG/FCI band keys, manifest and ash recipe."""

from __future__ import annotations

import numpy as np
import pytest


def test_bands_from_composite_recovers_the_physical_bands():
    from deepcontrail.mtg.bands import bands_from_composite

    # T10.5=250, T8.7=252, T12.3=248 -> composite [T12.3-T10.5, T10.5-T8.7, T10.5] = [-2, -2, 250]
    composite = np.array([[[-2.0, -2.0, 250.0]]], dtype=np.float32)

    bands = bands_from_composite(composite)

    assert bands.dtype == np.float32
    # BAND_KEYS order: ir_105=T10.5, ir_87=T8.7, ir_123_minus_ir_105=T12.3-T10.5
    np.testing.assert_array_equal(bands[0, 0], [250.0, 252.0, -2.0])


def test_bands_from_composite_rejects_a_wrong_shape():
    from deepcontrail.mtg.bands import bands_from_composite

    with pytest.raises(ValueError, match="composite"):
        bands_from_composite(np.zeros((4, 4), np.float32))


def test_band_manifest_is_valid_for_core():
    from deepcontrail.core import BandManifest
    from deepcontrail.mtg.bands import BAND_KEYS, band_manifest

    manifest = BandManifest.model_validate(band_manifest())

    assert manifest.keys == BAND_KEYS, "channel order of the spectral arrays on disk"


def test_bands_rebuild_the_same_composite_through_core():
    from deepcontrail.core import BandManifest, FalseColorConfig, false_color_from_spectral
    from deepcontrail.mtg.bands import FALSE_COLOR, band_manifest, bands_from_composite

    manifest = BandManifest.model_validate(band_manifest())
    config = FalseColorConfig.model_validate(FALSE_COLOR)

    rng = np.random.default_rng(0)
    lo = np.array([-4.0, -4.0, 243.0])
    hi = np.array([2.0, 5.0, 303.0])
    composite = rng.uniform(lo, hi, size=(4, 5, 3)).astype(np.float32)

    bands = bands_from_composite(composite)
    fc = false_color_from_spectral(bands, config, manifest)

    expected = np.clip((composite - lo) / (hi - lo), 0.0, 1.0)
    np.testing.assert_allclose(fc, expected, atol=1e-5, err_msg="the recipe == a direct stretch")
