"""Tests for deepcontrail.core.false_color — composites from named spectral bands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from deepcontrail.core.false_color import FalseColorConfig


# ---------------------------------------------------------------------------
# false_color_from_spectral
# ---------------------------------------------------------------------------


_KEYS = (
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


def _spectral(**values: float) -> np.ndarray:
    base = dict.fromkeys(_KEYS, 260.0) | values
    return np.stack([np.full((2, 2), base[k], dtype=np.float32) for k in _KEYS], axis=-1)


def _config_with_green(*bands: str) -> FalseColorConfig:
    from deepcontrail.core.false_color import FalseColorConfig

    default = FalseColorConfig()
    green = default.channels[1].model_copy(update={"bands": bands})
    return FalseColorConfig(channels=(default.channels[0], green, default.channels[2]))


def test_false_color_from_spectral_default_is_band13_legacy_recipe():
    from deepcontrail.core.false_color import FalseColorConfig, false_color_from_spectral

    bands = _spectral(data_12um=269.0, data_11um=270.0, data_10um=268.0, cloud_top=271.0)

    fc = false_color_from_spectral(bands, FalseColorConfig())

    np.testing.assert_allclose(
        fc[..., 0], (269 - 270 + 4) / 6, err_msg="R = (12um - 11um) / (-4, 2)"
    )
    np.testing.assert_allclose(fc[..., 1], (270 - 268 + 4) / 9, err_msg="G = 11um - data_10um")
    np.testing.assert_allclose(fc[..., 2], (270 - 243) / 60, err_msg="B = 11um over (243, 303)")


def test_false_color_from_spectral_uses_configured_green_bands():
    from deepcontrail.core.false_color import false_color_from_spectral

    bands = _spectral(data_11um=270.0, data_10um=268.0, cloud_top=271.0)

    fc = false_color_from_spectral(bands, _config_with_green("data_11um", "cloud_top"))

    np.testing.assert_allclose(
        fc[..., 1], (270 - 271 + 4) / 9, err_msg="G must follow the configured bands (cloud_top)"
    )


def test_false_color_from_spectral_shape_and_dtype():
    from deepcontrail.core.false_color import FalseColorConfig, false_color_from_spectral

    fc = false_color_from_spectral(_spectral(), FalseColorConfig())

    assert fc.shape == (2, 2, 3)
    assert fc.dtype == np.float32


def test_false_color_from_spectral_clips_to_unit_range():
    from deepcontrail.core.false_color import FalseColorConfig, false_color_from_spectral

    bands = _spectral(data_12um=400.0, data_11um=100.0, data_10um=0.0)

    fc = false_color_from_spectral(bands, FalseColorConfig())

    assert fc.min() >= 0.0
    assert fc.max() <= 1.0


def test_false_color_from_spectral_rejects_wrong_channel_count():
    from deepcontrail.core.false_color import FalseColorConfig, false_color_from_spectral

    with pytest.raises(ValueError, match="10"):
        false_color_from_spectral(np.zeros((2, 2, 3), dtype=np.float32), FalseColorConfig())
