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


def _spectral_frame(**values: float) -> np.ndarray:
    from deepcontrail.data.config import SPECTRAL_BAND_KEYS

    base = dict.fromkeys(SPECTRAL_BAND_KEYS, 260.0) | values
    return np.stack([np.full((2, 2), base[k], dtype=np.float32) for k in SPECTRAL_BAND_KEYS], -1)


def test_preview_rgb_uses_standard_ash_recipe():
    from scripts.convert_tfrecords import _preview_rgb

    spectral = _spectral_frame(data_12um=269.0, data_11um=270.0, cloud_top=268.0, data_10um=275.0)

    preview = _preview_rgb(spectral)

    assert preview.dtype == np.uint8
    expected = [(269 - 270 + 4) / 6, (270 - 268 + 4) / 9, (270 - 243) / 60]
    for c, value in enumerate(expected):
        assert np.all(preview[..., c] == np.uint8(np.float32(value) * 255)), (
            f"channel {c}: preview must be the standard ash RGB (G = band14 - band11 = cloud_top)"
        )


def test_preview_rgb_when_data_10um_changes_preview_is_unchanged():
    from scripts.convert_tfrecords import _preview_rgb

    low = _preview_rgb(_spectral_frame(data_10um=250.0))
    high = _preview_rgb(_spectral_frame(data_10um=290.0))

    assert np.array_equal(low, high), "data_10um (band 13) is not part of the standard ash recipe"


def test_convert_tfrecords_import_does_not_load_tensorflow():
    repo_root = Path(__file__).resolve().parents[2]
    code = "import sys, scripts.convert_tfrecords; print('tensorflow' in sys.modules)"

    result = subprocess.run(
        [sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False", (
        "tensorflow must be imported lazily: loading it next to torch segfaults the test suite"
    )


def test_prepare_output_dir_writes_band_manifest(tmp_path: Path):
    import json

    from deepcontrail.data.config import SPECTRAL_BAND_INFO, SPECTRAL_BAND_KEYS
    from scripts.convert_tfrecords import _prepare_output_dir

    _prepare_output_dir(tmp_path)

    manifest = json.loads((tmp_path / "spectral_bands" / "bands.json").read_text())
    channels = manifest["channels"]
    assert [c["key"] for c in channels] == list(SPECTRAL_BAND_KEYS), (
        "channel i = SPECTRAL_BAND_KEYS[i]"
    )
    assert [c["index"] for c in channels] == list(range(len(SPECTRAL_BAND_KEYS)))
    assert [c["abi_band"] for c in channels] == [
        SPECTRAL_BAND_INFO[k].abi_band for k in SPECTRAL_BAND_KEYS
    ]
    assert manifest["dtype"] == "float32"
    assert manifest["units"] == "K"


def test_prepare_output_dir_creates_split_directories(tmp_path: Path):
    from scripts.convert_tfrecords import _prepare_output_dir

    _prepare_output_dir(tmp_path)

    for sub in ("images_semantic", "semantic_masks", "probability_masks", "spectral_bands"):
        for split in ("train", "val", "test"):
            assert (tmp_path / sub / split).is_dir(), f"{sub}/{split} must exist before conversion"
    assert (tmp_path / "metadata").is_dir()


def test_prepare_output_dir_is_idempotent(tmp_path: Path):
    from scripts.convert_tfrecords import _prepare_output_dir

    _prepare_output_dir(tmp_path)
    first = (tmp_path / "spectral_bands" / "bands.json").read_text()

    _prepare_output_dir(tmp_path)

    assert (tmp_path / "spectral_bands" / "bands.json").read_text() == first, (
        "re-runs must not change it"
    )
