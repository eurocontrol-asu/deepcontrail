"""Tests for deepcontrail.core.false_color — composites from bands named by the manifest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from deepcontrail.core.contract import BandManifest
    from deepcontrail.core.false_color import FalseColorConfig

# Deliberately not the OpenContrails order: composites must follow the manifest.
_KEYS = ("ir_123", "ir_87", "ir_105", "wv_63")


def _manifest(keys: tuple[str, ...] = _KEYS) -> BandManifest:
    from deepcontrail.core.contract import BandManifest

    return BandManifest.model_validate(
        {
            "format_version": 1,
            "sensor": "MTG-I1 FCI",
            "dtype": "float32",
            "channels": [{"index": i, "key": k, "units": "K"} for i, k in enumerate(keys)],
        }
    )


def _ash(green: tuple[str, ...] = ("ir_105", "ir_87")) -> FalseColorConfig:
    from deepcontrail.core.false_color import FalseColorConfig

    return FalseColorConfig.model_validate(
        {
            "channels": [
                {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
                {"bands": list(green), "bounds": [-4, 5]},
                {"bands": ["ir_105"], "bounds": [243, 303]},
            ]
        }
    )


def _bands(**values: float) -> np.ndarray:
    base = dict.fromkeys(_KEYS, 260.0) | values
    return np.stack([np.full((2, 2), base[k], dtype=np.float32) for k in _KEYS], axis=-1)


def test_false_color_from_spectral_reads_bands_by_manifest_key():
    from deepcontrail.core.false_color import false_color_from_spectral

    bands = _bands(ir_123=269.0, ir_105=270.0, ir_87=271.0)

    fc = false_color_from_spectral(bands, _ash(), _manifest())

    np.testing.assert_allclose(fc[..., 0], (269 - 270 + 4) / 6, err_msg="R = ir_123 - ir_105")
    np.testing.assert_allclose(fc[..., 1], (270 - 271 + 4) / 9, err_msg="G = ir_105 - ir_87")
    np.testing.assert_allclose(fc[..., 2], (270 - 243) / 60, err_msg="B = ir_105")


def test_false_color_from_spectral_follows_the_configured_bands():
    from deepcontrail.core.false_color import false_color_from_spectral

    bands = _bands(ir_105=270.0, ir_87=271.0, wv_63=240.0)

    fc = false_color_from_spectral(bands, _ash(green=("ir_105", "wv_63")), _manifest())

    np.testing.assert_allclose(
        fc[..., 1], np.clip((270 - 240 + 4) / 9, 0, 1), err_msg="G must use the configured bands"
    )


def test_false_color_from_spectral_shape_and_dtype():
    from deepcontrail.core.false_color import false_color_from_spectral

    fc = false_color_from_spectral(_bands(), _ash(), _manifest())

    assert fc.shape == (2, 2, 3)
    assert fc.dtype == np.float32


def test_false_color_from_spectral_clips_to_unit_range():
    from deepcontrail.core.false_color import false_color_from_spectral

    bands = _bands(ir_123=400.0, ir_105=100.0, ir_87=0.0)

    fc = false_color_from_spectral(bands, _ash(), _manifest())

    assert fc.min() >= 0.0
    assert fc.max() <= 1.0


def test_false_color_from_spectral_rejects_arrays_not_matching_the_manifest():
    from deepcontrail.core.false_color import false_color_from_spectral

    with pytest.raises(ValueError, match="4 channels"):
        false_color_from_spectral(np.zeros((2, 2, 3), dtype=np.float32), _ash(), _manifest())


def test_check_false_color_accepts_a_recipe_using_manifest_keys():
    from deepcontrail.core.false_color import check_false_color

    check_false_color(_ash(), _manifest())


def test_check_false_color_names_unknown_keys_and_lists_available():
    from deepcontrail.core.contract import DatasetContractError
    from deepcontrail.core.false_color import check_false_color

    recipe = _ash(green=("ir_105", "cloud_top"))

    with pytest.raises(DatasetContractError, match=r"cloud_top.*ir_123, ir_87, ir_105, wv_63"):
        check_false_color(recipe, _manifest())


def test_false_color_config_has_no_default_recipe():
    from deepcontrail.core.false_color import FalseColorConfig

    with pytest.raises(ValidationError, match="channels"):
        FalseColorConfig()  # type: ignore[call-arg]


def test_false_color_channel_accepts_any_sensor_key():
    from deepcontrail.core.false_color import FalseColorChannel

    channel = FalseColorChannel(bands=("ir_105",), bounds=(243.0, 303.0))

    assert channel.bands == ("ir_105",), "band names come from the dataset, not a fixed list"


def test_false_color_channel_rejects_more_than_two_bands():
    from deepcontrail.core.false_color import FalseColorChannel

    with pytest.raises(ValidationError):
        FalseColorChannel(bands=("a", "b", "c"), bounds=(0.0, 1.0))  # type: ignore[arg-type]
