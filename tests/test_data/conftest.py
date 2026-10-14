"""Shared fixtures for data module tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """Create a minimal dataset root with 5 train and 2 test pairs.

    Layout mirrors the real dataset::

        tmp_path/
          images_semantic/train/  (5 RGB JPEGs, 16x16)
          images_semantic/test/   (2 RGB JPEGs, 16x16)
          semantic_masks/train/   (5 grayscale PNGs, 16x16, values {0, 255})
          semantic_masks/test/    (2 grayscale PNGs, 16x16, values {0, 255})
          probability_masks/train/ (5 float32 npy, 16x16)
          probability_masks/test/  (2 float32 npy, 16x16)

    The train/val split is handled on-the-fly by the dataset using
    ``val_fraction``.  With 5 stems and ``val_fraction=0.2``, the last
    stem goes to validation and the first 4 to training.
    """
    rng = np.random.RandomState(42)
    size = (16, 16)

    intermediate_stems = {"train_000001", "test_000002"}

    for split, count, prefix in [("train", 5, "train"), ("test", 2, "test")]:
        img_dir = tmp_path / "images_semantic" / split
        binary_mask_dir = tmp_path / "semantic_masks" / split
        prob_mask_dir = tmp_path / "probability_masks" / split
        fc_dir = tmp_path / "false_color" / split
        img_dir.mkdir(parents=True)
        binary_mask_dir.mkdir(parents=True)
        prob_mask_dir.mkdir(parents=True)
        fc_dir.mkdir(parents=True)

        for i in range(1, count + 1):
            stem = f"{prefix}_{i:06d}"

            img_arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
            Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")

            mask_arr = rng.choice([0, 255], size=size).astype(np.uint8)
            Image.fromarray(mask_arr, mode="L").save(binary_mask_dir / f"{stem}.png")

            prob_mask = mask_arr.astype(np.float32) / 255.0
            if stem in intermediate_stems:
                prob_mask[prob_mask > 0] = 0.75
            np.save(prob_mask_dir / f"{stem}.npy", prob_mask)

            fc = rng.uniform(0, 1, (*size, 3)).astype(np.float32)
            np.save(fc_dir / f"{stem}.npy", fc)

    return tmp_path


@pytest.fixture
def synthetic_root_allzero_mask(tmp_path: Path) -> Path:
    """Dataset root with one sample whose mask is entirely zero (no contrail)."""
    size = (16, 16)
    img_dir = tmp_path / "images_semantic" / "train"
    mask_dir = tmp_path / "semantic_masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    stem = "train_000001"
    img_arr = np.full((*size, 3), 128, dtype=np.uint8)
    Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")

    mask_arr = np.zeros(size, dtype=np.uint8)
    Image.fromarray(mask_arr, mode="L").save(mask_dir / f"{stem}.png")

    return tmp_path


@pytest.fixture
def synthetic_root_missing_mask(tmp_path: Path) -> Path:
    """Dataset root where one image has no matching mask."""
    size = (16, 16)
    rng = np.random.RandomState(99)

    img_dir = tmp_path / "images_semantic" / "train"
    mask_dir = tmp_path / "semantic_masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for stem in ["train_000001", "train_000002"]:
        img_arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
        Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")

    mask_arr = rng.choice([0, 255], size=size).astype(np.uint8)
    Image.fromarray(mask_arr, mode="L").save(mask_dir / "train_000001.png")

    return tmp_path
