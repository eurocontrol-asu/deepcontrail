"""Tests for deepcontrail.data.transforms — augmentation pipeline builder."""

from __future__ import annotations

import numpy as np
import pytest

albumentations = pytest.importorskip("albumentations")


def test_build_augmentation_returns_compose():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig()
    transform = build_augmentation(cfg, seed=42)

    assert isinstance(transform, albumentations.Compose)


def test_build_augmentation_preserves_mask_binary():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig()

    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    mask = rng.choice([0, 1], size=(16, 16)).astype(np.uint8)

    for seed in range(20):
        transform = build_augmentation(cfg, seed=seed)
        result = transform(image=image, mask=mask)
        unique = set(np.unique(result["mask"]).tolist())
        assert unique <= {0, 1}, f"augmented mask values {unique} not in {{0, 1}}"


def test_build_augmentation_preserves_spatial_dims():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig()
    transform = build_augmentation(cfg, seed=42)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)

    result = transform(image=image, mask=mask)

    assert result["image"].shape == (16, 16, 3), "spatial dims must be preserved"
    assert result["mask"].shape == (16, 16), "mask spatial dims must be preserved"


def test_build_augmentation_respects_zero_probs():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig(horizontal_flip=0.0, vertical_flip=0.0, random_rotate_90=0.0)
    transform = build_augmentation(cfg, seed=42)

    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    mask = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16) % 2

    result = transform(image=image, mask=mask)

    np.testing.assert_array_equal(result["image"], image, err_msg="p=0 should not modify image")
    np.testing.assert_array_equal(result["mask"], mask, err_msg="p=0 should not modify mask")


def test_build_augmentation_seeded_is_reproducible():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig()
    image = np.random.RandomState(99).randint(0, 256, (16, 16, 3), dtype=np.uint8)
    mask = np.random.RandomState(99).choice([0, 1], size=(16, 16)).astype(np.uint8)

    t1 = build_augmentation(cfg, seed=7)
    result1 = t1(image=image, mask=mask)

    t2 = build_augmentation(cfg, seed=7)
    result2 = t2(image=image, mask=mask)

    np.testing.assert_array_equal(
        result1["image"], result2["image"], err_msg="same seed must produce identical output"
    )
    np.testing.assert_array_equal(
        result1["mask"], result2["mask"], err_msg="same seed must produce identical mask"
    )


def test_build_augmentation_does_not_resize():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig(horizontal_flip=0.0, vertical_flip=0.0, random_rotate_90=0.0)
    transform = build_augmentation(cfg, seed=42)

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)

    result = transform(image=image, mask=mask)

    assert result["image"].shape == (16, 16, 3), "augmentation must not resize"
    assert result["mask"].shape == (16, 16), "augmentation must not resize the mask"


def test_build_image_resize_only_resizes_image():
    from deepcontrail.data.transforms import build_image_resize

    transform = build_image_resize(8)

    image = np.zeros((16, 16, 3), dtype=np.uint8)

    result = transform(image=image)

    assert result["image"].shape == (8, 8, 3), "resize should target the image only"
    assert "mask" not in result, "resize must not touch the mask"
