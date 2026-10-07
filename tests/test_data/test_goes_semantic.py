"""Tests for deepcontrail.data.goes_semantic — GOESSemanticDataset."""

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
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert len(ds) == 3, "should discover 3 train pairs"


def test_init_discovers_val_samples(synthetic_root: Path):
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="val", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert len(ds) == 2, "should discover 2 val pairs"


def test_init_raises_on_missing_mask(synthetic_root_missing_mask: Path):
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root_missing_mask), split="train", image_size=_SYNTH_SIZE)

    with pytest.raises(FileNotFoundError, match="train_000002"):
        GOESDataset(cfg)


def test_init_discovers_test_samples(synthetic_root: Path):
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="test", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert len(ds) == 2, "should discover 2 test pairs"


def test_init_raises_on_invalid_split():
    from pydantic import ValidationError

    from deepcontrail.data.config import DataConfig

    with pytest.raises(ValidationError, match="split"):
        DataConfig(root="/tmp/fake", split="other")


def test_stems_are_sorted(synthetic_root: Path):
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    assert ds.stems == sorted(ds.stems), "stems must be deterministically sorted"


# ---------------------------------------------------------------------------
# __getitem__ — with torch
# ---------------------------------------------------------------------------


def test_getitem_returns_correct_shapes(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "image should be CHW"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE), "mask should be HW"


def test_getitem_image_dtype_float32(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, _ = ds[0]

    assert image.dtype == torch.float32


def test_getitem_mask_dtype_long(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    _, mask = ds[0]

    assert mask.dtype == torch.long


def test_getitem_mask_is_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        unique = set(mask.unique().tolist())
        assert unique <= {0, 1}, f"mask[{i}] values {unique} not in {{0, 1}}"


def test_getitem_allzero_mask_is_valid(synthetic_root_allzero_mask: Path):
    pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root_allzero_mask), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert mask.sum().item() == 0, "all-zero mask should have sum 0"
    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_image_normalized(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, _ = ds[0]

    assert image.min() < 0.0 or image.max() > 1.0, (
        "after ImageNet normalization, values should not all be in [0, 1]"
    )


# ---------------------------------------------------------------------------
# __getitem__ — numpy fallback (no torch)
# ---------------------------------------------------------------------------


def test_getitem_without_torch_returns_numpy(synthetic_root: Path):
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    with mock.patch.dict("sys.modules", {"torch": None}):
        image, mask = ds[0]

    assert isinstance(image, np.ndarray), "should fall back to numpy"
    assert isinstance(mask, np.ndarray)
    assert image.dtype == np.float32
    assert image.ndim == 3, "numpy image should be HWC"
    assert mask.ndim == 2


def test_getitem_numpy_fallback_mask_binary(synthetic_root: Path):
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE)
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
    from deepcontrail.data.config import AugmentationConfig, DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    aug = AugmentationConfig()
    cfg = DataConfig(
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
    from deepcontrail.data.config import AugmentationConfig, DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    aug = AugmentationConfig()
    cfg = DataConfig(
        root=str(synthetic_root),
        split="train",
        augmentation=aug,
        image_size=_SYNTH_SIZE,
    )

    transforms_mod = sys.modules.pop("deepcontrail.data.transforms", None)
    try:
        with (
            mock.patch.dict(
                "sys.modules",
                {"albumentations": None, "deepcontrail.data.transforms": None},
            ),
            pytest.raises(ImportError, match="deepcontrail\\[train\\]"),
        ):
            GOESDataset(cfg)
    finally:
        if transforms_mod is not None:
            sys.modules["deepcontrail.data.transforms"] = transforms_mod


# ---------------------------------------------------------------------------
# image_size resize
# ---------------------------------------------------------------------------


def test_getitem_resizes_when_image_size_differs(synthetic_root: Path):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="val", image_size=8)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, 8, 8), f"expected (3, 8, 8), got {image.shape}"
    assert mask.shape == (8, 8), f"expected (8, 8), got {mask.shape}"


def test_getitem_no_resize_when_image_size_matches_native(synthetic_root: Path):
    pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="val", image_size=_SYNTH_SIZE)
    ds = GOESDataset(cfg)

    image, mask = ds[0]

    assert image.shape == (3, _SYNTH_SIZE, _SYNTH_SIZE), "no resize needed when sizes match"
    assert mask.shape == (_SYNTH_SIZE, _SYNTH_SIZE)


def test_getitem_resize_keeps_mask_binary(synthetic_root: Path):
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg = DataConfig(root=str(synthetic_root), split="train", image_size=8)
    ds = GOESDataset(cfg)

    for i in range(len(ds)):
        _, mask = ds[i]
        unique = set(mask.unique().tolist())
        assert unique <= {0, 1}, f"resized mask[{i}] values {unique} not in {{0, 1}}"


# ---------------------------------------------------------------------------
# pixel_shift
# ---------------------------------------------------------------------------


def test_dataset_no_shift_by_default():
    from deepcontrail.data.config import DataConfig

    cfg = DataConfig(root="/tmp/fake")

    assert cfg.pixel_shift == 0.0, "pixel_shift must default to 0.0"


def test_dataset_applies_pixel_shift(synthetic_root: Path):
    torch = pytest.importorskip("torch")
    from deepcontrail.data.config import DataConfig
    from deepcontrail.data.goes_semantic import GOESSemanticDataset as GOESDataset

    cfg_no_shift = DataConfig(
        root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE, pixel_shift=0.0
    )
    cfg_with_shift = DataConfig(
        root=str(synthetic_root), split="train", image_size=_SYNTH_SIZE, pixel_shift=0.5
    )

    ds_no = GOESDataset(cfg_no_shift)
    ds_yes = GOESDataset(cfg_with_shift)

    img_no, _ = ds_no[0]
    img_yes, _ = ds_yes[0]

    assert not torch.equal(img_no, img_yes), (
        "pixel_shift=0.5 must produce different image values than pixel_shift=0.0"
    )
