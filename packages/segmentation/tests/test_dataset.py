"""Tests for deepcontrail.segmentation.dataset — SegmentationDataset."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

_SYNTH_SIZE = 16

# Ash-style recipe over the synthetic datasets' band keys (see conftest.py).
RECIPE = {
    "channels": [
        {"bands": ["ir_123", "ir_105"], "bounds": [-4, 2]},
        {"bands": ["ir_105", "ir_87"], "bounds": [-4, 5]},
        {"bands": ["ir_105"], "bounds": [243, 303]},
    ]
}


# ---------------------------------------------------------------------------
# __init__ / __len__ / pairing
# ---------------------------------------------------------------------------


def test_init_discovers_paired_samples(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    assert len(ds) == 4, "5 stems with val_fraction=0.1 → 4 train"


def test_init_splits_train_val_on_the_fly(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    train_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE,
            root=str(synthetic_root),
            split="train",
            image_size=_SYNTH_SIZE,
            val_fraction=0.2,
        )
    )
    val_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE,
            root=str(synthetic_root),
            split="val",
            image_size=_SYNTH_SIZE,
            val_fraction=0.2,
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
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root_missing_mask),
        split="train",
        image_size=_SYNTH_SIZE,
    )

    with pytest.raises(ValueError, match="train_000002"):
        SegmentationDataset(cfg)


def _declare_temporal(root: Path, temporal: bool) -> None:
    """Set ``temporal`` in the dataset's ``metadata/dataset_info.json``."""
    import json

    path = root / "metadata" / "dataset_info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["temporal"] = temporal
    path.write_text(json.dumps(info), encoding="utf-8")


def _add_scene(root: Path, stem: str, *, with_mask: bool) -> None:
    from .conftest import write_bands, write_mask

    write_bands(root / "spectral_bands" / "train" / f"{stem}.npy", np.random.RandomState(3))
    if with_mask:
        mask = np.zeros((_SYNTH_SIZE, _SYNTH_SIZE), np.uint8)
        write_mask(root / "semantic_masks" / "train" / f"{stem}.png", mask)


def test_init_loads_suffix_like_ids_when_no_context_frames_are_declared(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    for stem in ("tile_a1", "site_b2"):
        _add_scene(synthetic_root, stem, with_mask=True)

    stems = [
        stem
        for split in ("train", "val")
        for stem in SegmentationDataset(
            SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), split=split)
        ).stems
    ]

    assert {"tile_a1", "site_b2"} <= set(stems), (
        f"single-frame data may use any id, _a1/_b2 endings included: {stems}"
    )


@pytest.mark.parametrize("temporal", [False, True])
def test_init_reports_a_missing_mask_for_an_id_that_is_no_context_frame(
    synthetic_root: Path, temporal: bool
):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    _declare_temporal(synthetic_root, temporal)
    _add_scene(synthetic_root, "tile_a1", with_mask=False)  # no scene "tile": never a context frame

    with pytest.raises(ValueError, match="tile_a1"):
        SegmentationDataset(
            SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), split="train")
        )


def test_init_skips_sequence_files_without_masks(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    # Context frames of train_000001 (no masks) next to the labelled scenes, in a dataset
    # that declares context frames.
    _declare_temporal(synthetic_root, True)
    for stem in ["train_000001_a1", "train_000001_a2", "train_000001_b1"]:
        frame = np.zeros((_SYNTH_SIZE, _SYNTH_SIZE, 3), dtype=np.float32)
        np.save(synthetic_root / "spectral_bands" / "train" / f"{stem}.npy", frame)

    stems = [
        stem
        for split in ("train", "val")
        for stem in SegmentationDataset(
            SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), split=split)
        ).stems
    ]

    assert sorted(stems) == [f"train_{i:06d}" for i in range(1, 6)], (
        f"context frames without masks must be skipped, got {stems}"
    )


def test_init_discovers_test_samples(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="test", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    assert len(ds) == 2, "should discover 2 test pairs"


def test_init_raises_on_invalid_split():
    from pydantic import ValidationError

    from deepcontrail.segmentation.config import SegmentationDataConfig

    with pytest.raises(ValidationError, match="split"):
        SegmentationDataConfig(false_color=RECIPE, root="/tmp/fake", split="other")


def test_stems_are_sorted(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    assert ds.stems == sorted(ds.stems), "stems must be deterministically sorted"


# ---------------------------------------------------------------------------
# val_fraction splitting
# ---------------------------------------------------------------------------


def test_val_split_reads_from_train_directory(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    val_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="val", val_fraction=0.4
        )
    )

    assert len(val_ds) == 2, "40% of 5 → 2 val stems"
    assert val_ds._image_dir == synthetic_root / "spectral_bands" / "train"


def test_val_split_is_seeded_and_reproducible(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    ds1 = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="val", val_fraction=0.2, seed=42
        )
    )
    ds2 = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="val", val_fraction=0.2, seed=42
        )
    )

    assert ds1.stems == ds2.stems, "same seed must produce identical val split"
    assert len(ds1.stems) == 1, "20% of 5 → 1 val stem"


def test_train_val_no_overlap_and_full_coverage(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    for frac in [0.1, 0.2, 0.4, 0.5]:
        train_ds = SegmentationDataset(
            SegmentationDataConfig(
                false_color=RECIPE, root=str(synthetic_root), split="train", val_fraction=frac
            )
        )
        val_ds = SegmentationDataset(
            SegmentationDataConfig(
                false_color=RECIPE, root=str(synthetic_root), split="val", val_fraction=frac
            )
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
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    image, mask = ds[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "image should be CHW"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE), "mask should be HW"


def test_getitem_image_dtype_float32(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    image, _ = ds[0]

    assert image.dtype == torch.float32


def test_getitem_mask_dtype_long(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.long


def test_getitem_mask_is_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        unique = set(mask.unique().tolist())
        assert unique <= {0, 1}, f"mask[{i}] values {unique} not in {{0, 1}}"


def test_getitem_allzero_mask_is_valid(synthetic_root_allzero_mask: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root_allzero_mask),
        split="val",
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    image, mask = ds[0]

    assert mask.sum().item() == 0, "all-zero mask should have sum 0"
    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_image_normalized(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    image, _ = ds[0]

    assert image.min() < 0.0 or image.max() > 1.0, (
        "after ImageNet normalization, values should not all be in [0, 1]"
    )


# ---------------------------------------------------------------------------
# __getitem__ — numpy fallback (no torch)
# ---------------------------------------------------------------------------


def test_getitem_without_torch_returns_numpy(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, mask = ds[0]

    assert isinstance(image, np.ndarray), "should fall back to numpy"
    assert isinstance(mask, np.ndarray)
    assert image.dtype == np.float32
    assert image.ndim == 3, "numpy image should be HWC"
    assert mask.ndim == 2


def test_getitem_numpy_fallback_mask_binary(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

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
    from deepcontrail.segmentation.dataset import SegmentationDataset

    aug = AugmentationConfig()
    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        seed=7,
        augmentation=aug,
        image_size=_SYNTH_SIZE,
    )

    ds1 = SegmentationDataset(cfg)
    img1, mask1 = ds1[0]

    ds2 = SegmentationDataset(cfg)
    img2, mask2 = ds2[0]

    assert torch.equal(img1, img2), "same seed must produce identical image"
    assert torch.equal(mask1, mask2), "same seed must produce identical mask"


# ---------------------------------------------------------------------------
# loud ImportError when albumentations missing
# ---------------------------------------------------------------------------


def test_build_transform_raises_without_albumentations(synthetic_root: Path):
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    aug = AugmentationConfig()
    cfg = SegmentationDataConfig(
        false_color=RECIPE,
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
            SegmentationDataset(cfg)
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
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=8
    )
    ds = SegmentationDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, 8, 8), f"expected (3, 8, 8), got {image.shape}"
    assert mask.shape == (16, 16), f"mask stays native, expected (16, 16), got {mask.shape}"
    assert mask.shape != image.shape[1:], "image and mask must have different spatial sizes"


def test_dataset_reports_native_resolution(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), split="train")
    ds = SegmentationDataset(cfg)

    assert ds.native_size == _SYNTH_SIZE
    assert ds.native_resolution == _SYNTH_SIZE


def test_getitem_no_resize_when_image_size_matches_native(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "no resize needed when sizes match"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_resize_keeps_mask_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=8
    )
    ds = SegmentationDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        unique = set(mask.unique().tolist())
        assert unique <= {0, 1}, f"resized mask[{i}] values {unique} not in {{0, 1}}"


# ---------------------------------------------------------------------------
# pixel_shift
# ---------------------------------------------------------------------------


def test_dataset_no_shift_by_default():
    from deepcontrail.segmentation.config import SegmentationDataConfig

    cfg = SegmentationDataConfig(false_color=RECIPE, root="/tmp/fake")

    assert cfg.pixel_shift == 0.0, "pixel_shift must default to 0.0"


def test_dataset_applies_pixel_shift(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg_no_shift = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        pixel_shift=0.0,
    )
    cfg_with_shift = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        image_size=_SYNTH_SIZE,
        pixel_shift=0.5,
    )

    ds_no = SegmentationDataset(cfg_no_shift)
    with pytest.warns(UserWarning, match="pixel_shift"):
        ds_yes = SegmentationDataset(cfg_with_shift)

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

    cfg = SegmentationDataConfig(false_color=RECIPE, root="/tmp/fake")

    assert cfg.fold is None, "CV must be disabled by default"
    assert cfg.num_folds == 4


def test_fold_partitions_train_stems(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cv_train = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="train", fold=0, num_folds=5
        )
    )
    cv_val = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="val", fold=0, num_folds=5
        )
    )

    union = set(cv_train.stems) | set(cv_val.stems)
    assert len(union) == 5, "CV fold must cover all 5 train stems"
    assert set(cv_train.stems).isdisjoint(cv_val.stems), "train and val folds must not overlap"


def test_fold_getitem_loads_from_correct_directory(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cv_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE,
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
    from deepcontrail.segmentation.dataset import SegmentationDataset

    test_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="test", fold=1, num_folds=4
        )
    )

    assert len(test_ds) == 2
    assert set(test_ds.stems) == {"test_000001", "test_000002"}


# ---------------------------------------------------------------------------
# mask_type: probability masks
# ---------------------------------------------------------------------------


def test_binary_mask_is_long_tensor(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="binary",
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.long, "binary mask_type must return a long tensor"


def test_probability_mask_returns_float32_tensor(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.float32, "probability mask_type must return a float32 tensor"


def test_probability_mask_values_in_range_zero_to_one(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    _, mask = ds[0]

    assert mask.min() >= 0.0, f"probability mask min {mask.min()} is below 0"
    assert mask.max() <= 1.0, f"probability mask max {mask.max()} is above 1"


def test_probability_mask_shape_is_hw(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    _, mask = ds[0]

    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE), (
        "probability mask should be a 2-D (H, W) array with no channel dimension"
    )


def test_probability_mask_not_validated_as_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        assert mask.ndim == 2, f"mask[{i}] should be 2-D"


def test_stems_discovered_with_probability_mask_type(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    binary_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="train", mask_type="binary"
        )
    )
    prob_ds = SegmentationDataset(
        SegmentationDataConfig(
            false_color=RECIPE, root=str(synthetic_root), split="train", mask_type="probability"
        )
    )

    assert prob_ds.stems == binary_ds.stems, "probability mask_type must discover the same stems"


def test_probability_mask_with_augmentation_stays_float(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        augmentation=AugmentationConfig(),
        seed=7,
        image_size=_SYNTH_SIZE,
    )
    ds = SegmentationDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.float32, "augmented probability mask must stay float32"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_probability_mask_augmentation_is_reproducible(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE,
        root=str(synthetic_root),
        split="train",
        mask_type="probability",
        augmentation=AugmentationConfig(),
        seed=7,
        image_size=_SYNTH_SIZE,
    )

    ds1 = SegmentationDataset(cfg)
    img1, mask1 = ds1[0]

    ds2 = SegmentationDataset(cfg)
    img2, mask2 = ds2[0]

    assert torch.equal(img1, img2), "same seed must produce identical image"
    assert torch.equal(mask1, mask2), "same seed must produce identical probability mask"


# ---------------------------------------------------------------------------
# images built from spectral_bands/ with the recipe
# ---------------------------------------------------------------------------


def test_getitem_spectral_returns_correct_shapes(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )

    image, mask = SegmentationDataset(cfg)[0]

    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "image should be CHW"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_spectral_numpy_fallback(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, _mask = ds[0]

    assert isinstance(image, np.ndarray)
    assert image.dtype == np.float32
    assert image.shape == (_SYNTH_SIZE, _SYNTH_SIZE, 3)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_getitem_builds_image_from_spectral_bands_with_the_recipe(synthetic_root: Path):
    from deepcontrail.core.false_color import false_color_from_spectral
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(
        false_color=RECIPE, root=str(synthetic_root), split="test", image_size=_SYNTH_SIZE
    )
    ds = SegmentationDataset(cfg)
    bands = np.load(synthetic_root / "spectral_bands" / "test" / f"{ds.stems[0]}.npy")

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, _mask = ds[0]

    np.testing.assert_allclose(
        image,
        false_color_from_spectral(bands, cfg.false_color, ds.manifest),
        err_msg="image must be built from spectral_bands/ with the configured recipe",
    )


def test_getitem_green_follows_configured_bands(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    other_green = {
        "channels": [
            RECIPE["channels"][0],
            {"bands": ["ir_105", "ir_123"], "bounds": [-4, 5]},
            RECIPE["channels"][2],
        ]
    }

    def first_image(recipe: dict) -> np.ndarray:
        cfg = SegmentationDataConfig(
            false_color=recipe, root=str(synthetic_root), split="test", image_size=_SYNTH_SIZE
        )
        with mock.patch.dict("sys.modules", {"torch": None}):
            return SegmentationDataset(cfg)[0][0]

    img_a = first_image(RECIPE)
    img_b = first_image(other_green)

    assert np.array_equal(img_a[..., [0, 2]], img_b[..., [0, 2]]), "R and B unchanged"
    assert not np.array_equal(img_a[..., 1], img_b[..., 1]), "G must change with its bands"


# ---------------------------------------------------------------------------
# dataset format checks at construction
# ---------------------------------------------------------------------------


def test_init_rejects_a_recipe_with_a_band_key_the_dataset_lacks(synthetic_root: Path):
    from deepcontrail.core.contract import DatasetContractError
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    recipe = {"channels": [*RECIPE["channels"][:2], {"bands": ["cloud_top"], "bounds": [0, 1]}]}

    with pytest.raises(DatasetContractError, match=r"cloud_top.*ir_87, ir_105, ir_123"):
        SegmentationDataset(SegmentationDataConfig(false_color=recipe, root=str(synthetic_root)))


def test_init_without_band_manifest_raises(synthetic_root: Path):
    from deepcontrail.core.contract import DatasetContractError
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    (synthetic_root / "spectral_bands" / "bands.json").unlink()

    with pytest.raises(DatasetContractError, match=r"bands\.json"):
        SegmentationDataset(SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root)))


def test_init_warns_when_pixel_shift_differs_from_the_recommendation(synthetic_root: Path):
    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), pixel_shift=0.5)

    with pytest.warns(UserWarning, match=r"pixel_shift=0\.5.*recommends 0\.0"):
        SegmentationDataset(cfg)


def test_init_does_not_warn_when_pixel_shift_matches(synthetic_root: Path):
    import warnings

    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    cfg = SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), pixel_shift=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SegmentationDataset(cfg)


def test_init_does_not_warn_when_no_shift_is_recommended(synthetic_root: Path):
    import json
    import warnings

    from deepcontrail.segmentation.config import SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    info_path = synthetic_root / "metadata" / "dataset_info.json"
    info = json.loads(info_path.read_text()) | {"recommended_pixel_shift": None}
    info_path.write_text(json.dumps(info))
    cfg = SegmentationDataConfig(false_color=RECIPE, root=str(synthetic_root), pixel_shift=0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SegmentationDataset(cfg)


def test_dataloader_workers_draw_different_augmentations(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from torch.utils.data import DataLoader

    from deepcontrail.segmentation.config import AugmentationConfig, SegmentationDataConfig
    from deepcontrail.segmentation.dataset import SegmentationDataset

    from .conftest import write_bands, write_mask, write_metadata

    write_metadata(tmp_path)
    rng = np.random.RandomState(0)
    for i in range(16):
        write_bands(tmp_path / "spectral_bands" / "train" / f"s{i:02d}.npy", rng)
        write_mask(
            tmp_path / "semantic_masks" / "train" / f"s{i:02d}.png", np.zeros((16, 16), np.uint8)
        )
    base = {"false_color": RECIPE, "root": str(tmp_path), "split": "train"}
    plain = SegmentationDataset(SegmentationDataConfig(**base, image_size=_SYNTH_SIZE))
    augmented = SegmentationDataset(
        SegmentationDataConfig(
            **base,
            image_size=_SYNTH_SIZE,
            augmentation=AugmentationConfig(
                horizontal_flip=0.5, vertical_flip=0.5, random_rotate_90=0.5
            ),
        )
    )

    def variant(out, ref) -> str:
        for k in range(4):
            if torch.equal(out, torch.rot90(ref, k, dims=(1, 2))):
                return f"rot{90 * k}"
            if torch.equal(out, torch.rot90(torch.flip(ref, dims=(2,)), k, dims=(1, 2))):
                return f"mirror+rot{90 * k}"
        return "other"

    torch.manual_seed(0)
    loader = DataLoader(augmented, batch_size=1, shuffle=False, num_workers=2)  # type: ignore[arg-type]

    ops = [variant(image[0], plain[i][0]) for i, (image, _) in enumerate(loader)]

    # With batch_size=1 and two workers, even indices come from worker 0 and odd from worker 1.
    assert "other" not in ops, f"every output is a flip/rotation of its scene: {ops}"
    assert ops[0::2] != ops[1::2], (
        f"each worker must draw its own augmentations, not replay the other's: {ops}"
    )
