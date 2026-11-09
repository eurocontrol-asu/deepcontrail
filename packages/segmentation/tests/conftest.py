"""Shared fixtures for the segmentation data tests: tiny datasets in the DeepContrail format."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

# albumentations checks PyPI for a newer version when it is imported; conftest loads before the
# tests that import it, so a library must not make a network call on import. An explicit value is
# kept.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

if TYPE_CHECKING:
    from pathlib import Path

# Band keys of the synthetic datasets; the tests' RECIPE builds its composite from them.
BAND_KEYS = ("ir_87", "ir_105", "ir_123")


def write_metadata(root: Path, *, recommended_pixel_shift: float | None = 0.0) -> None:
    """Write ``spectral_bands/bands.json`` and ``metadata/dataset_info.json`` (format v1)."""
    manifest = {
        "format_version": 1,
        "sensor": "synthetic",
        "dtype": "float32",
        "channels": [{"index": i, "key": k, "units": "K"} for i, k in enumerate(BAND_KEYS)],
    }
    (root / "spectral_bands").mkdir(parents=True, exist_ok=True)
    (root / "spectral_bands" / "bands.json").write_text(json.dumps(manifest))
    info = {
        "format_version": 1,
        "source": "synthetic",
        "records": {"train": 0, "test": 0},
        "labels": ["semantic", "probability"],
        "recommended_pixel_shift": recommended_pixel_shift,
    }
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata" / "dataset_info.json").write_text(json.dumps(info))


def write_bands(path: Path, rng: np.random.RandomState, size: tuple[int, int] = (16, 16)) -> None:
    """Write one ``spectral_bands`` array: brightness temperatures in K."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, rng.uniform(230, 300, (*size, len(BAND_KEYS))).astype(np.float32))


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a consensus mask PNG with values {0, 255}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """Create a minimal dataset root with 5 train and 2 test records.

    Layout follows the DeepContrail dataset format::

        tmp_path/
          spectral_bands/bands.json  (3 channels: ir_87, ir_105, ir_123)
          spectral_bands/{train,test}/  (float32 npy, 16x16x3, Kelvin)
          semantic_masks/{train,test}/  (grayscale PNGs, 16x16, values {0, 255})
          probability_masks/{train,test}/  (float32 npy, 16x16)
          metadata/dataset_info.json  (recommended_pixel_shift 0.0)

    The train/val split is handled on-the-fly by the dataset using
    ``val_fraction``.
    """
    mask_rng = np.random.RandomState(42)
    band_rng = np.random.RandomState(7)
    size = (16, 16)
    intermediate_stems = {"train_000001", "test_000002"}

    write_metadata(tmp_path)
    for split, count, prefix in [("train", 5, "train"), ("test", 2, "test")]:
        (tmp_path / "probability_masks" / split).mkdir(parents=True)
        for i in range(1, count + 1):
            stem = f"{prefix}_{i:06d}"
            write_bands(tmp_path / "spectral_bands" / split / f"{stem}.npy", band_rng, size)

            mask_arr = mask_rng.choice([0, 255], size=size).astype(np.uint8)
            write_mask(tmp_path / "semantic_masks" / split / f"{stem}.png", mask_arr)

            prob_mask = mask_arr.astype(np.float32) / 255.0
            if stem in intermediate_stems:
                prob_mask[prob_mask > 0] = 0.75
            np.save(tmp_path / "probability_masks" / split / f"{stem}.npy", prob_mask)

    return tmp_path


@pytest.fixture
def synthetic_root_allzero_mask(tmp_path: Path) -> Path:
    """Dataset root with one sample whose mask is entirely zero (no contrail)."""
    write_metadata(tmp_path)
    write_bands(
        tmp_path / "spectral_bands" / "train" / "train_000001.npy", np.random.RandomState(1)
    )
    write_mask(
        tmp_path / "semantic_masks" / "train" / "train_000001.png", np.zeros((16, 16), np.uint8)
    )
    return tmp_path


@pytest.fixture
def synthetic_root_missing_mask(tmp_path: Path) -> Path:
    """Dataset root where one image has no matching mask."""
    rng = np.random.RandomState(99)
    write_metadata(tmp_path)
    for stem in ["train_000001", "train_000002"]:
        write_bands(tmp_path / "spectral_bands" / "train" / f"{stem}.npy", rng)
    mask_arr = rng.choice([0, 255], size=(16, 16)).astype(np.uint8)
    write_mask(tmp_path / "semantic_masks" / "train" / "train_000001.png", mask_arr)
    return tmp_path
