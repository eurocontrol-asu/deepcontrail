"""GOES-16 semantic segmentation dataset.

Loads paired RGB images and binary masks from the filesystem.
``import deepcontrail.data`` requires only numpy + Pillow — torch is
imported lazily inside ``__getitem__`` and is optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from deepcontrail.data._loading import (
    binarize_mask,
    load_image,
    load_mask,
    shift_image,
    to_numpy,
    to_tensors,
    validate_binary_mask,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepcontrail.data.config import DataConfig


class GOESSemanticDataset:
    """Per-sample dataset for GOES-16 semantic segmentation images.

    Parameters
    ----------
    config
        A ``DataConfig`` specifying root path, split, augmentation, and
        normalization parameters.

    Raises
    ------
    FileNotFoundError
        If any image in the split directory has no matching mask.
    ImportError
        If augmentation or resize is requested but albumentations is not
        installed.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self._root = Path(config.root)

        self._image_dir = self._root / "images_semantic" / config.split
        self._mask_dir = self._root / "semantic_masks" / config.split

        self.stems = self._discover_stems()
        self._transform = self._build_transform()

    def _discover_stems(self) -> list[str]:
        """Scan image directory, verify each has a matching mask, return sorted stems."""
        image_paths = sorted(self._image_dir.glob("*.jpg"))
        stems: list[str] = []
        for img_path in image_paths:
            stem = img_path.stem
            mask_path = self._mask_dir / f"{stem}.png"
            if not mask_path.exists():
                msg = f"No mask found for image {stem!r} (expected {mask_path})"
                raise FileNotFoundError(msg)
            stems.append(stem)
        return stems

    def _needs_resize(self) -> bool:
        """Check if loaded images need resizing to match config.image_size."""
        if not self.stems:
            return False
        with Image.open(self._image_dir / f"{self.stems[0]}.jpg") as img:
            w, h = img.size
        return w != self.config.image_size or h != self.config.image_size

    def _build_transform(self) -> Callable[..., dict[str, Any]] | None:
        """Build albumentations pipeline from config, or None."""
        has_augmentation = self.config.augmentation is not None
        needs_resize = self._needs_resize()

        if not has_augmentation and not needs_resize:
            return None

        try:
            from deepcontrail.data.transforms import build_augmentation, build_resize
        except ImportError:
            msg = (
                "albumentations is required for augmentation and resizing. "
                "Install it with: pip install deepcontrail[train]"
            )
            raise ImportError(msg) from None

        if has_augmentation:
            return build_augmentation(
                self.config.augmentation,  # type: ignore[arg-type]
                seed=self.config.seed,
                image_size=self.config.image_size if needs_resize else None,
            )
        return build_resize(self.config.image_size)

    def __len__(self) -> int:
        """Return the number of image/mask pairs in this split."""
        return len(self.stems)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Load and return ``(image, mask)`` for the given index.

        With torch installed, returns:
        - ``image``: float32 tensor, shape ``(C, H, W)``, ImageNet-normalized.
        - ``mask``: long tensor, shape ``(H, W)``, values in ``{0, 1}``.

        Without torch, returns numpy arrays:
        - ``image``: float32 ndarray, shape ``(H, W, C)``, in ``[0, 1]``.
        - ``mask``: uint8 ndarray, shape ``(H, W)``, values in ``{0, 1}``.
        """
        stem = self.stems[index]

        image = load_image(self._image_dir / f"{stem}.jpg")
        if self.config.pixel_shift != 0.0:
            image = shift_image(image, self.config.pixel_shift)
        mask = load_mask(self._mask_dir / f"{stem}.png")

        mask = binarize_mask(mask)
        assert image.shape[:2] == mask.shape, (
            f"spatial mismatch: image {image.shape[:2]} vs mask {mask.shape}"
        )

        if self._transform is not None:
            transformed = self._transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        validate_binary_mask(mask)

        try:
            import torch as _  # noqa: F401
        except ImportError:
            return to_numpy(image, mask)
        return to_tensors(image, mask, self.config.normalize)
