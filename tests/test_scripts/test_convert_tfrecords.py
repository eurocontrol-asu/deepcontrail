"""Tests for the TFRecord converter's ash false-color composite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

# OpenContrails key -> ABI band, verified against Kaggle's labelled band_XX files:
# data_10um=band13 (10.3um), data_11um=band14, data_12um=band15, cloud_top=band11 (8.4um).


def _false_color(bands: dict[str, np.ndarray], t: int) -> np.ndarray:
    from scripts.convert_tfrecords import _false_color_frame

    return _false_color_frame(bands, t)


def _bands(n_frames: int = 1, **values: float) -> dict[str, np.ndarray]:
    from scripts.convert_tfrecords import SPECTRAL_BAND_KEYS

    defaults = dict.fromkeys(SPECTRAL_BAND_KEYS, 260.0)
    defaults.update(values)
    return {k: np.full((2, 2, n_frames), v, dtype=np.float64) for k, v in defaults.items()}


def test_false_color_frame_green_uses_cloud_top_band():
    bands = _bands(data_11um=270.0, cloud_top=268.0, data_10um=271.0)

    fc = _false_color(bands, t=0)

    expected_g = (270.0 - 268.0 + 4.0) / 9.0
    assert np.allclose(fc[..., 1], expected_g), (
        "G must be band14 - band11 (data_11um - cloud_top), not band14 - band13 (data_10um)"
    )


def test_false_color_frame_when_data_10um_changes_output_is_unchanged():
    low = _bands(data_10um=250.0)
    high = _bands(data_10um=290.0)

    fc_low = _false_color(low, t=0)
    fc_high = _false_color(high, t=0)

    assert np.array_equal(fc_low, fc_high), "data_10um (band 13) is not part of the ash recipe"


def test_false_color_frame_red_and_blue_channels_match_recipe():
    bands = _bands(data_12um=269.0, data_11um=270.0)

    fc = _false_color(bands, t=0)

    assert np.allclose(fc[..., 0], (269.0 - 270.0 + 4.0) / 6.0), "R = band15 - band14 over (-4, 2)"
    assert np.allclose(fc[..., 2], (270.0 - 243.0) / 60.0), "B = band14 over (243, 303)"


def test_false_color_frame_reads_cloud_top_at_requested_frame():
    bands = _bands(n_frames=2, data_11um=270.0)
    bands["cloud_top"][..., 0] = 266.0
    bands["cloud_top"][..., 1] = 270.0

    fc = _false_color(bands, t=1)

    assert np.allclose(fc[..., 1], 4.0 / 9.0), "G must use cloud_top from frame t, not frame 0"


def test_convert_tfrecords_import_does_not_load_tensorflow():
    repo_root = Path(__file__).resolve().parents[2]
    code = "import sys, scripts.convert_tfrecords; print('tensorflow' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", (
        "tensorflow must be imported lazily: loading it next to torch segfaults the test suite"
    )
