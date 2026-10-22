"""Tests for the TFRecord converter's per-frame outputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def _bands_dict(n_frames: int = 1) -> dict[str, np.ndarray]:
    from deepcontrail.data.config import SPECTRAL_BAND_KEYS

    return {
        key: np.full((2, 2, n_frames), 200.0 + i, dtype=np.float64)
        for i, key in enumerate(SPECTRAL_BAND_KEYS)
    }


def test_spectral_bands_frame_stacks_keys_in_shared_channel_order():
    from scripts.convert_tfrecords import _spectral_bands_frame

    bands = _bands_dict()

    frame = _spectral_bands_frame(bands, 0)

    assert frame.shape == (2, 2, 10)
    assert frame.dtype == np.float32
    np.testing.assert_array_equal(
        frame[0, 0],
        200.0 + np.arange(10),
        err_msg="channel i must be SPECTRAL_BAND_KEYS[i], the order the data loader assumes",
    )


def test_spectral_bands_frame_reads_requested_frame():
    from scripts.convert_tfrecords import _spectral_bands_frame

    bands = _bands_dict(n_frames=2)
    bands["cloud_top"][..., 1] = 999.0

    frame = _spectral_bands_frame(bands, 1)

    assert np.all(frame[..., 4] == 999.0), "frame t must come from time index t"


def test_preview_rgb_is_default_false_color_as_uint8():
    from deepcontrail.data._loading import false_color_from_spectral
    from deepcontrail.data.config import FalseColorConfig
    from scripts.convert_tfrecords import _preview_rgb

    rng = np.random.RandomState(0)
    spectral = rng.uniform(230, 300, (4, 4, 10)).astype(np.float32)

    preview = _preview_rgb(spectral)

    expected = (false_color_from_spectral(spectral, FalseColorConfig()) * 255).astype(np.uint8)
    assert preview.dtype == np.uint8
    np.testing.assert_array_equal(
        preview, expected, err_msg="preview JPEGs use the default (band13 legacy) composite"
    )


def test_convert_tfrecords_import_does_not_load_tensorflow():
    repo_root = Path(__file__).resolve().parents[2]
    code = "import sys, scripts.convert_tfrecords; print('tensorflow' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", (
        "tensorflow must be imported lazily: loading it next to torch segfaults the test suite"
    )
