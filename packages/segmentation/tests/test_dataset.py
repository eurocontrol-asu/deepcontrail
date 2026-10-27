"""Tests for deepcontrail.segmentation.dataset — GOESSemanticDataset."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

_SYNTH_SIZE = 16


# ---------------------------------------------------------------------------
# __init__ / __len__ / pairing
# ---------------------------------------------------------------------------


def test_init_discovers_paired_samples(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert len(ds) == 4, "5 stems with val_fraction=0.1 → 4 train"


def test_init_splits_train_val_on_the_fly(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    train_ds = GOESDataset(
        SegmentationDataConfig(
            root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE, val_fraction=0.2
        )
    )
    val_ds = GOESDataset(
        SegmentationDataConfig(
            root=str(synthetic_root), split="val", image_size=_SYNTH_SIZE, val_fraction=0.2
        )
    )

    assert len(train_ds) == 4, "80% of 5 → 4 train"
    assert len(val_ds) == 1, "20% of 5 → 1 val"
    assert set(train_ds.stems).isdisjoint(val_ds.stems), "train and val must not overlap"
    assert set(train_ds.stems) | set(val_ds.stems) == {f"train_{i:06d}" for i in range(1, 6)}, (
        "train + val must cover all stems"
    )


def test_init_raises_on_missing_mask(synthetic_root_missing_mask: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root_missing_mask), split="train", image_size=_SYNTH_SIZE
    )

    with pytest.raises(ValueError, match="train_000002"):
        GOESDataset(cfg)


def test_init_skips_sequence_files_without_masks(tmp_path: Path):
    from PIL import Image

    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    img_dir = tmp_path / "images_semantic" / "train"
    mask_dir = tmp_path / "semantic_masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.RandomState(99)
    size = (_SYNTH_SIZE, _SYNTH_SIZE)

    # Base image with mask
    for stem in ["train_000001"]:
        img_arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
        Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")
        mask_arr = rng.choice([0, 255], size=size).astype(np.uint8)
        Image.fromarray(mask_arr, mode="L").save(mask_dir / f"{stem}.png")

    # Sequence images without masks (should be silently skipped)
    for stem in ["train_000001_a1", "train_000001_a2", "train_000001_b1"]:
        img_arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
        Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")

    cfg = SegmentationDataConfig(
        root=str(tmp_path),
        split="val",
        image_size=_SYNTH_SIZE,
        val_fraction=0.9,
    )
    ds = GOESDataset(cfg)

    assert ds.stems == ["train_000001"], f"only base stem expected, got {ds.stems}"


def test_init_discovers_test_samples(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="test", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert len(ds) == 2, "should discover 2 test pairs"


def test_init_raises_on_invalid_split():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="split"):
        SegmentationDataConfig(root="/tmp/fake", split="other")


def test_stems_are_sorted(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert ds.stems == sorted(ds.stems), "stems must be deterministically sorted"


# ---------------------------------------------------------------------------
# val_fraction splitting
# ---------------------------------------------------------------------------


def test_val_split_reads_from_train_directory(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    val_ds = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="val", val_fraction=0.4)
    )

    assert len(val_ds) == 2, "40% of 5 → 2 val stems"
    assert val_ds._image_dir == synthetic_root / "images_semantic" / "train"


def test_val_split_is_seeded_and_reproducible(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    ds1 = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="val", val_fraction=0.2, seed=42)
    )
    ds2 = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="val", val_fraction=0.2, seed=42)
    )

    assert ds1.stems == ds2.stems, "same seed must produce identical val split"
    assert len(ds1.stems) == 1, "20% of 5 → 1 val stem"


def test_train_val_no_overlap_and_full_coverage(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    for frac in [0.1, 0.2, 0.4, 0.5]:
        train_ds = GOESDataset(
            SegmentationDataConfig(root=str(synthetic_root), split="train", val_fraction=frac)
        )
        val_ds = GOESDataset(
            SegmentationDataConfig(root=str(synthetic_root), split="val", val_fraction=frac)
        )

        all_stems = set(train_ds.stems) | set(val_ds.stems)
        assert len(all_stems) == 5, f"frac={frac}: must cover all 5 stems"
        assert set(train_ds.stems).isdisjoint(val_ds.stems), f"frac={frac}: overlap"


# ---------------------------------------------------------------------------
# __getitem__ — with torch
# ---------------------------------------------------------------------------


def test_getitem_returns_correct_shapes(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "image should be CHW"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE), "mask should be HW"


def test_getitem_image_dtype_float32(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, _ = ds[0]

    assert image.dtype == torch.float32


def test_getitem_mask_dtype_long(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.long


def test_getitem_mask_is_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        unique = set(mask.unique().tolist())
        assert unique <= {0, 1}, f"mask[{i}] values {unique} not in {{0, 1}}"


def test_getitem_allzero_mask_is_valid(synthetic_root_allzero_mask: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root_allzero_mask),
        split="val",
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert mask.sum().item() == 0, "all-zero mask should have sum 0"
    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_image_normalized(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, _ = ds[0]

    assert image.min() < 0.0 or image.max() > 1.0, (
        "after ImageNet normalization, values should not all be in [0, 1]"
    )


# ---------------------------------------------------------------------------
# __getitem__ — numpy fallback (no torch)
# ---------------------------------------------------------------------------


def test_getitem_without_torch_returns_numpy(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, mask = ds[0]

    assert isinstance(image, np.ndarray), "should fall back to numpy"
    assert isinstance(mask, np.ndarray)
    assert image.dtype == np.float32
    assert image.ndim == 3, "numpy image should be HWC"
    assert mask.ndim == 2


def test_getitem_numpy_fallback_mask_binary(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    with mock.patch.dict("sys.modules", {"torch": None}):
        _, mask = ds[0]

    unique = set(np.unique(mask).tolist())
    assert unique <= {0, 1}, f"numpy mask values {unique} not in {{0, 1}}"


# ---------------------------------------------------------------------------
# seed reproducibility
# ---------------------------------------------------------------------------


def test_getitem_seeded_augmentation_is_reproducible(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    aug = AugmentationConfig()
    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        seed=7,
        augmentation=aug,
        image_size=_SYNTH_SIZE,
    )

    ds1 = GOESDataset(cfg)
    img1, mask1 = ds1[0]

    ds2 = GOESDataset(cfg)
    img2, mask2 = ds2[0]

    assert torch.equal(img1, img2), "same seed must produce identical image"
    assert torch.equal(mask1, mask2), "same seed must produce identical mask"


# ---------------------------------------------------------------------------
# loud ImportError when albumentations missing
# ---------------------------------------------------------------------------


def test_build_transform_raises_without_albumentations(synthetic_root: Path):
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    aug = AugmentationConfig()
    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        augmentation=aug,
        image_size=_SYNTH_SIZE,
    )

    transforms_mod = sys.modules.pop("deepcontrail.segmentation.transforms", None)
    try:
        with (
            mock.patch.dict(
                "sys.modules",
                {"albumentations": None, "deepcontrail.segmentation.transforms": None},
            ),
            pytest.raises(ImportError, match="deepcontrail\\[train\\]"),
        ):
            GOESDataset(cfg)
    finally:
        if transforms_mod is not None:
            sys.modules["deepcontrail.segmentation.transforms"] = transforms_mod


# ---------------------------------------------------------------------------
# image_size resize
# ---------------------------------------------------------------------------


def test_getitem_resizes_image_only_stays_mask_native(synthetic_root: Path):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=8)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, 8, 8), f"expected (3, 8, 8), got {image.shape}"
    assert mask.shape == (16, 16), f"mask stays native, expected (16, 16), got {mask.shape}"
    assert mask.shape != image.shape[1:], "image and mask must have different spatial sizes"


def test_dataset_reports_native_resolution(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train")
    ds = GOESDataset(cfg)

    assert ds.native_size == _SYNTH_SIZE
    assert ds.native_resolution == _SYNTH_SIZE


def test_getitem_no_resize_when_image_size_matches_native(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "no resize needed when sizes match"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_resize_keeps_mask_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(root=str(synthetic_root), split="train", image_size=8)
    ds = GOESDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        unique = set(mask.unique().tolist())
        assert unique <= {0, 1}, f"resized mask[{i}] values {unique} not in {{0, 1}}"


# ---------------------------------------------------------------------------
# pixel_shift
# ---------------------------------------------------------------------------


def test_dataset_no_shift_by_default():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.pixel_shift == 0.0, "pixel_shift must default to 0.0"


def test_dataset_applies_pixel_shift(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg_no_shift = SegmentationDataConfig(
        root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE, pixel_shift=0.0
    )
    cfg_with_shift = SegmentationDataConfig(
        root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE, pixel_shift=0.5
    )

    ds_no = GOESDataset(cfg_no_shift)
    ds_yes = GOESDataset(cfg_with_shift)

    img_no, _ = ds_no[0]
    img_yes, _ = ds_yes[0]

    assert not torch.equal(img_no, img_yes), (
        "pixel_shift=0.5 must produce different image values than pixel_shift=0.0"
    )


# ---------------------------------------------------------------------------
# k-fold cross-validation
# ---------------------------------------------------------------------------


def test_fold_defaults_to_none():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(root="/tmp/fake")

    assert cfg.fold is None, "CV must be disabled by default"
    assert cfg.num_folds == 4


def test_fold_partitions_train_stems(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cv_train = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="train", fold=0, num_folds=5)
    )
    cv_val = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="val", fold=0, num_folds=5)
    )

    union = set(cv_train.stems) | set(cv_val.stems)
    assert len(union) == 5, "CV fold must cover all 5 train stems"
    assert set(cv_train.stems).isdisjoint(cv_val.stems), "train and val folds must not overlap"


def test_fold_getitem_loads_from_correct_directory(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cv_ds = GOESDataset(
        SegmentationDataConfig(
            root=str(synthetic_root),
            split="train",
            fold=0,
            num_folds=5,
            image_size=_SYNTH_SIZE,
        )
    )

    for i in range(len(cv_ds)):
        image, mask = cv_ds[i]
        assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), f"sample {i}: bad image shape"
        assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE), f"sample {i}: bad mask shape"


def test_fold_test_is_not_folded(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    test_ds = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="test", fold=1, num_folds=4)
    )

    assert len(test_ds) == 2
    assert set(test_ds.stems) == {"test_000001", "test_000002"}


# ---------------------------------------------------------------------------
# mask_type: probability masks
# ---------------------------------------------------------------------------


def test_binary_mask_is_long_tensor(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="binary",
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.long, "binary mask_type must return a long tensor"


def test_probability_mask_returns_float32_tensor(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.float32, "probability mask_type must return a float32 tensor"


def test_probability_mask_values_in_range_zero_to_one(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.min() >= 0.0, f"probability mask min {mask.min()} is below 0"
    assert mask.max() <= 1.0, f"probability mask max {mask.max()} is above 1"


def test_probability_mask_shape_is_hw(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE), (
        "probability mask should be a 2-D (H, W) array with no channel dimension"
    )


def test_probability_mask_not_validated_as_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        assert mask.ndim == 2, f"mask[{i}] should be 2-D"


def test_stems_discovered_with_probability_mask_type(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    binary_ds = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="train", mask_type="binary")
    )
    prob_ds = GOESDataset(
        SegmentationDataConfig(root=str(synthetic_root), split="train", mask_type="probability")
    )

    assert prob_ds.stems == binary_ds.stems, "probability mask_type must discover the same stems"


def test_probability_mask_with_augmentation_stays_float(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        augmentation=AugmentationConfig(),
        seed=7,
        image_size=_SYNTH_SIZE,
    )
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.float32, "augmented probability mask must stay float32"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_probability_mask_augmentation_is_reproducible(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        augmentation=AugmentationConfig(),
        seed=7,
        image_size=_SYNTH_SIZE,
    )

    ds1 = GOESDataset(cfg)
    img1, mask1 = ds1[0]

    ds2 = GOESDataset(cfg)
    img2, mask2 = ds2[0]

    assert torch.equal(img1, img2), "same seed must produce identical image"
    assert torch.equal(mask1, mask2), "same seed must produce identical probability mask"


# ---------------------------------------------------------------------------
# image_source: spectral
# ---------------------------------------------------------------------------


def test_getitem_spectral_source_returns_correct_shapes(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        image_source="spectral",
    )
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "spectral image should be CHW"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_spectral_source_differs_from_jpeg(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg_jpeg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        image_source="jpeg",
    )
    cfg_spectral = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        image_source="spectral",
    )

    ds_jpeg = GOESDataset(cfg_jpeg)
    ds_spectral = GOESDataset(cfg_spectral)

    img_jpeg, _ = ds_jpeg[0]
    img_spectral, _ = ds_spectral[0]

    assert not torch.equal(img_jpeg, img_spectral), (
        "spectral source should produce different values than JPEG source"
    )


def test_getitem_spectral_source_with_pixel_shift(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg_no = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        image_source="spectral",
        pixel_shift=0.0,
    )
    cfg_yes = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        image_source="spectral",
        pixel_shift=0.5,
    )

    ds_no = GOESDataset(cfg_no)
    ds_yes = GOESDataset(cfg_yes)

    img_no, _ = ds_no[0]
    img_yes, _ = ds_yes[0]

    assert not torch.equal(img_no, img_yes), (
        "pixel_shift should change spectral-sourced image values"
    )


def test_getitem_spectral_source_numpy_fallback(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        image_source="spectral",
    )
    ds = GOESDataset(cfg)

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, _mask = ds[0]

    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32
    assert image.shape == (_SYNTH_SIZE, _SYNTH_SIZE, 3)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_getitem_spectral_source_builds_image_from_spectral_bands(synthetic_root: Path):
    from deepcontrail.core.false_color import false_color_from_spectral
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    cfg = SegmentationDataConfig(
        root=str(synthetic_root), split="test", image_size=_SYNTH_SIZE, image_source="spectral"
    )
    ds = GOESDataset(cfg)
    bands = np.load(synthetic_root / "spectral_bands" / "test" / f"{ds.stems[0]}.npy")

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, _mask = ds[0]

    np.testing.assert_allclose(
        image,
        false_color_from_spectral(bands, cfg.false_color),
        err_msg="spectral image must be built from spectral_bands/ with the configured channels",
    )


def test_getitem_spectral_source_green_follows_configured_bands(synthetic_root: Path):
    from deepcontrail.core.false_color import FalseColorConfig
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import GOESSemanticDataset as GOESDataset

    legacy = FalseColorConfig()
    green = legacy.channels[1].model_copy(update={"bands": ("data_11um", "cloud_top")})
    standard = FalseColorConfig(channels=(legacy.channels[0], green, legacy.channels[2]))

    def first_image(fc: FalseColorConfig) -> np.ndarray:
        cfg = SegmentationDataConfig(
            root=str(synthetic_root),
            split="test",
            image_size=_SYNTH_SIZE,
            image_source="spectral",
            false_color=fc,
        )
        with mock.patch.dict("sys.modules", {"torch": None}):
            return GOESDataset(cfg)[0][0]

    img_legacy = first_image(legacy)
    img_standard = first_image(standard)

    assert np.array_equal(img_legacy[..., [0, 2]], img_standard[..., [0, 2]]), "R and B unchanged"
    assert not np.array_equal(img_legacy[..., 1], img_standard[..., 1]), "G must change with bands"
