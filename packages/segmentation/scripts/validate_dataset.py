#!/usr/bin/env python
"""Validate a DeepContrail dataset for integrity before training.

Scans every image and mask, checking for corruption, truncation, and size
mismatches.  Run once on a new dataset to catch problems before they crash
a training run mid-epoch.

Usage
-----
::

    uv run python packages/segmentation/scripts/validate_dataset.py \
        --root /data/.../goes_contrails_dataset/20250305

    # With probability masks
    uv run python packages/segmentation/scripts/validate_dataset.py \
        --root /data/.../goes_contrails_dataset/20250305 \
        --mask-type probability
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _validate_split(
    image_dir: Path,
    mask_dir: Path,
    mask_ext: str,
) -> list[str]:
    """Validate all image/mask pairs in a single split directory."""
    errors: list[str] = []
    image_paths = sorted(image_dir.glob("*.jpg"))

    if not image_paths:
        return errors

    ref_size: tuple[int, int] | None = None

    for img_path in image_paths:
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}{mask_ext}"

        if not mask_path.exists():
            errors.append(f"{stem}: no mask (expected {mask_path})")
            continue

        try:
            with Image.open(img_path) as img:
                img.load()
                img_size = img.size
        except Exception as e:
            errors.append(f"{stem}: corrupt image ({e})")
            continue

        if ref_size is None:
            ref_size = img_size
        elif img_size != ref_size:
            errors.append(f"{stem}: image size {img_size} != expected {ref_size}")
            continue

        try:
            if mask_ext == ".npy":
                mask = np.load(mask_path)
                mask_hw = (mask.shape[0], mask.shape[1])
            else:
                with Image.open(mask_path) as m:
                    m.load()
                    mask_hw = (m.size[1], m.size[0])
        except Exception as e:
            errors.append(f"{stem}: corrupt mask ({e})")
            continue

        if mask_hw != (img_size[1], img_size[0]):
            errors.append(f"{stem}: mask {mask_hw} != image {(img_size[1], img_size[0])}")

    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate a dataset and return 0 (clean) or 1 (problems found)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument(
        "--mask-type",
        default="binary",
        choices=["binary", "probability"],
        help="Mask type: binary (.png) or probability (.npy). Default: binary.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    mask_subdir = "probability_masks" if args.mask_type == "probability" else "semantic_masks"
    mask_ext = ".npy" if args.mask_type == "probability" else ".png"

    all_errors: list[str] = []

    for split in ["train", "test"]:
        image_dir = root / "images_semantic" / split
        mask_dir = root / mask_subdir / split

        if not image_dir.exists():
            continue

        n_images = len(list(image_dir.glob("*.jpg")))
        print(f"Validating {split}/ ({n_images} images)...")

        errors = _validate_split(image_dir, mask_dir, mask_ext)
        all_errors.extend(f"[{split}] {e}" for e in errors)

    if all_errors:
        print(f"\nValidation found {len(all_errors)} problem(s):")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("\nDataset is clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
