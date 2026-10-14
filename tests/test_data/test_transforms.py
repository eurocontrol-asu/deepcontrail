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


def test_build_augmentation_all_extras_disabled_by_default():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig()
    transform = build_augmentation(cfg, seed=42)

    # Only the three base spatial transforms when every extra is at p=0.0.
    assert len(transform.transforms) == 3
    assert {type(t).__name__ for t in transform.transforms} == {
        "HorizontalFlip",
        "VerticalFlip",
        "RandomRotate90",
    }


def test_build_augmentation_shift_scale_rotate_included():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig(shift_scale_rotate=0.5)
    transform = build_augmentation(cfg, seed=42)

    # 3 base transforms + the extra ShiftScaleRotate.
    assert len(transform.transforms) == 4
    assert any(isinstance(t, albumentations.ShiftScaleRotate) for t in transform.transforms)


def test_build_augmentation_photometric_preserves_mask():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig(
        horizontal_flip=0.0,
        vertical_flip=0.0,
        random_rotate_90=0.0,
        gaussian_blur=1.0,
        brightness_contrast=1.0,
    )

    rng = np.random.RandomState(42)
    image = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:32, :32] = 1  # known top-left quadrant

    # Photometric transforms must never touch the mask, regardless of seed.
    for seed in range(10):
        transform = build_augmentation(cfg, seed=seed)
        result = transform(image=image, mask=mask)
        np.testing.assert_array_equal(
            result["mask"], mask, err_msg="photometric transforms must not alter the mask"
        )


def test_build_augmentation_config_new_fields_default_zero():
    from deepcontrail.data.config import AugmentationConfig

    cfg = AugmentationConfig()

    assert cfg.shift_scale_rotate == 0.0
    assert cfg.gaussian_blur == 0.0
    assert cfg.brightness_contrast == 0.0


def test_build_augmentation_all_extras_build_and_run():
    from deepcontrail.data.config import AugmentationConfig
    from deepcontrail.data.transforms import build_augmentation

    cfg = AugmentationConfig(
        shift_scale_rotate=1.0,
        gaussian_blur=1.0,
        brightness_contrast=1.0,
    )
    transform = build_augmentation(cfg, seed=42)

    # One extra per enabled flag: 3 base + 3 extras.
    assert len(transform.transforms) == 6

    rng = np.random.RandomState(42)
    image = rng.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    mask = rng.choice([0, 1], size=(48, 48)).astype(np.uint8)

    result = transform(image=image, mask=mask)

    # Spatial extras (shift/rotate, mask co-transform) still keep the mask binary.
    assert set(np.unique(result["mask"]).tolist()) <= {0, 1}
    assert result["image"].shape == (48, 48, 3)
