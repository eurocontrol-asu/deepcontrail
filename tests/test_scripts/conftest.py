"""Fixtures for script tests — datasets with known defects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def synthetic_root_corrupt_image(tmp_path: Path) -> Path:
    """Dataset root where one image is truncated/corrupt."""
    size = (16, 16)
    rng = np.random.RandomState(99)

    img_dir = tmp_path / "images_semantic" / "train"
    mask_dir = tmp_path / "semantic_masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    img_arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(img_arr, mode="RGB").save(img_dir / "train_000001.jpg")
    mask_arr = rng.choice([0, 255], size=size).astype(np.uint8)
    Image.fromarray(mask_arr, mode="L").save(mask_dir / "train_000001.png")

    (img_dir / "train_000002.jpg").write_bytes(b"\xff\xd8\xff\xe0truncated")
    Image.fromarray(mask_arr, mode="L").save(mask_dir / "train_000002.png")

    return tmp_path


@pytest.fixture
def synthetic_root_size_mismatch(tmp_path: Path) -> Path:
    """Dataset root where one image has a different size."""
    rng = np.random.RandomState(99)

    img_dir = tmp_path / "images_semantic" / "train"
    mask_dir = tmp_path / "semantic_masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    img_arr = rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(img_arr, mode="RGB").save(img_dir / "train_000001.jpg")
    mask_arr = rng.choice([0, 255], size=(16, 16)).astype(np.uint8)
    Image.fromarray(mask_arr, mode="L").save(mask_dir / "train_000001.png")

    img_arr2 = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(img_arr2, mode="RGB").save(img_dir / "train_000002.jpg")
    mask_arr2 = rng.choice([0, 255], size=(32, 32)).astype(np.uint8)
    Image.fromarray(mask_arr2, mode="L").save(mask_dir / "train_000002.png")

    return tmp_path
