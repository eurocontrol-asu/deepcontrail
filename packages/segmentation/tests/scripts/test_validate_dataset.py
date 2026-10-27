"""Tests for the standalone dataset validation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def _make_clean_dataset(tmp_path: Path, n_train: int = 3, n_test: int = 2) -> Path:
    """Create a small valid dataset for testing."""
    rng = np.random.RandomState(42)
    size = (16, 16)

    for split, count, prefix in [("train", n_train, "train"), ("test", n_test, "test")]:
        img_dir = tmp_path / "images_semantic" / split
        mask_dir = tmp_path / "semantic_masks" / split
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        for i in range(1, count + 1):
            stem = f"{prefix}_{i:06d}"
            img_arr = rng.randint(0, 256, (*size, 3), dtype=np.uint8)
            Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")
            mask_arr = rng.choice([0, 255], size=size).astype(np.uint8)
            Image.fromarray(mask_arr, mode="L").save(mask_dir / f"{stem}.png")

    return tmp_path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_dataset.py"


def _load_main() -> Callable[..., int]:
    """Import ``scripts/validate_dataset.py`` by path; ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location("validate_dataset", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_validate_clean_dataset_returns_zero(tmp_path: Path) -> None:
    main = _load_main()

    root = _make_clean_dataset(tmp_path)

    exit_code = main(["--root", str(root)])

    assert exit_code == 0, "clean dataset should pass validation"


def test_validate_corrupt_image_reports_error(
    synthetic_root_corrupt_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = _load_main()

    exit_code = main(["--root", str(synthetic_root_corrupt_image)])

    assert exit_code == 1, "corrupt image should fail validation"
    captured = capsys.readouterr()
    assert "train_000002" in captured.out, "should identify the corrupt stem"
    assert "corrupt image" in captured.out.lower()


def test_validate_size_mismatch_reports_error(
    synthetic_root_size_mismatch: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = _load_main()

    exit_code = main(["--root", str(synthetic_root_size_mismatch)])

    assert exit_code == 1, "size mismatch should fail validation"
    captured = capsys.readouterr()
    assert "image size" in captured.out.lower()


def test_validate_missing_mask_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main = _load_main()

    img_dir = tmp_path / "images_semantic" / "train"
    mask_dir = tmp_path / "semantic_masks" / "train"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    rng = np.random.RandomState(99)
    for stem in ["train_000001", "train_000002"]:
        img_arr = rng.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(img_arr, mode="RGB").save(img_dir / f"{stem}.jpg")

    mask_arr = rng.choice([0, 255], size=(16, 16)).astype(np.uint8)
    Image.fromarray(mask_arr, mode="L").save(mask_dir / "train_000001.png")

    exit_code = main(["--root", str(tmp_path)])

    assert exit_code == 1, "missing mask should fail validation"
    captured = capsys.readouterr()
    assert "train_000002" in captured.out
    assert "no mask" in captured.out.lower()
