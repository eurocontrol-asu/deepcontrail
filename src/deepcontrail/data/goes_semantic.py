"""GOES-16 semantic segmentation dataset.

Loads paired RGB images and binary masks from the filesystem.
``import deepcontrail.data`` requires only numpy + Pillow — torch is
imported lazily inside ``__getitem__`` and is optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from deepcontrail.data._loading import (
    binarize_mask,
    load_false_color,
    load_image,
    load_mask,
    shift_image,
    to_numpy,
    to_tensors,
    to_tensors_soft,
    validate_binary_mask,
)

if TYPE_CHECKING:
    import albumentations as A

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

        img_subdir = "false_color" if config.image_source == "spectral" else "images_semantic"
        mask_subdir = "probability_masks" if config.mask_type == "probability" else "semantic_masks"
        ds = "test" if config.split == "test" else "train"
        self._dir_pairs = [
            (self._root / img_subdir / ds, self._root / mask_subdir / ds),
        ]
        self._image_dir, self._mask_dir = self._dir_pairs[0]

        self._stem_dirs: dict[str, tuple[Path, Path]] = {}
        self.stems = self._discover_stems()
        self.native_size = self._native_size()
        self._aug_transform, self._resize_transform = self._build_transform()

    @property
    def native_resolution(self) -> int:
        """Native square resolution of the images (256 for GOES-16)."""
        return self.native_size

    def _discover_stems(self) -> list[str]:
        """Scan the image directory and partition stems for the requested split.

        ``"test"`` returns every stem in ``test/``.  ``"train"`` and ``"val"``
        both read from ``train/`` and split deterministically:

        * When ``fold`` is set, k-fold CV partitioning (modular arithmetic)
          takes precedence.
        * Otherwise ``val_fraction`` determines the split — stems are
          shuffled with ``seed`` and the first ``val_fraction * N`` go to
          validation, the rest to training.
        """
        all_stems: list[str] = []
        for img_dir, mask_dir in self._dir_pairs:
            scanned = self._scan_stems(img_dir, mask_dir)
            for stem in scanned:
                self._stem_dirs[stem] = (img_dir, mask_dir)
            all_stems.extend(scanned)
        stems = sorted(all_stems)

        if self.config.split == "test":
            return stems

        fold = self.config.fold
        if fold is not None:
            num_folds = self.config.num_folds
            if self.config.split == "val":
                return [s for i, s in enumerate(stems) if i % num_folds == fold]
            return [s for i, s in enumerate(stems) if i % num_folds != fold]

        vf = self.config.val_fraction
        rng = np.random.RandomState(self.config.seed)
        indices = rng.permutation(len(stems))
        n_val = max(1, round(len(stems) * vf))
        val_idx = set(indices[:n_val].tolist())
        if self.config.split == "val":
            return [s for i, s in enumerate(stems) if i in val_idx]
        return [s for i, s in enumerate(stems) if i not in val_idx]

    def _scan_stems(self, image_dir: Path, mask_dir: Path) -> list[str]:
        """Scan ``image_dir`` for paired images/masks, return sorted stems.

        Checks that each image has a matching mask file.  For deeper
        validation (corruption, size mismatches), run the standalone
        ``scripts/validate_dataset.py`` on the dataset root.
        """
        mask_ext = ".npy" if self.config.mask_type == "probability" else ".png"
        img_ext = ".npy" if self.config.image_source == "spectral" else ".jpg"
        image_paths = sorted(image_dir.glob(f"*{img_ext}"))
        stems: list[str] = []
        missing: list[str] = []

        for img_path in image_paths:
            stem = img_path.stem
            mask_path = mask_dir / f"{stem}{mask_ext}"
            if not mask_path.exists():
                missing.append(f"{stem}: no mask (expected {mask_path})")
                continue
            stems.append(stem)

        if missing:
            msg = f"Dataset validation found {len(missing)} missing mask(s):\n"
            msg += "\n".join(f"  - {e}" for e in missing)
            raise ValueError(msg)

        return stems

    def _native_size(self) -> int:
        """Native square resolution of the images (256 for GOES-16)."""
        if not self.stems:
            return self.config.image_size
        if self.config.image_source == "spectral":
            arr = np.load(self._image_dir / f"{self.stems[0]}.npy")
            return arr.shape[0]
        with Image.open(self._image_dir / f"{self.stems[0]}.jpg") as img:
            width, _height = img.size
        return width

    def _needs_resize(self) -> bool:
        """Whether images must be resized to match ``image_size``.

        Compares the native resolution against ``image_size``; the mask is
        always returned at native resolution regardless.
        """
        if not self.stems:
            return False
        return self.native_size != self.config.image_size

    def _build_transform(
        self,
    ) -> tuple[A.Compose | None, A.Resize | None]:
        """Build the augmentation and (optional) image-only resize transforms.

        Returns
        -------
        tuple[A.Compose | None, A.Resize | None]
            ``(augmentation, image_resize)``. ``augmentation`` co-transforms
            the image and mask at native resolution; ``image_resize`` resizes
            only the image, when ``image_size`` differs from native.
        """
        has_augmentation = self.config.augmentation is not None
        needs_resize = self._needs_resize()

        if not has_augmentation and not needs_resize:
            return None, None

        try:
            from deepcontrail.data.transforms import build_augmentation, build_image_resize
        except ImportError:
            msg = (
                "albumentations is required for augmentation and resizing. "
                "Install it with: pip install deepcontrail[train]"
            )
            raise ImportError(msg) from None

        augmentation = (
            build_augmentation(self.config.augmentation, seed=self.config.seed)  # type: ignore[arg-type]
            if has_augmentation
            else None
        )
        image_resize = build_image_resize(self.config.image_size) if needs_resize else None
        return augmentation, image_resize

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

        img_dir, mask_dir = self._stem_dirs.get(stem, (self._image_dir, self._mask_dir))
        if self.config.image_source == "spectral":
            image = load_false_color(img_dir / f"{stem}.npy")
        else:
            image = load_image(img_dir / f"{stem}.jpg")
        if self.config.pixel_shift != 0.0:
            image = shift_image(image, self.config.pixel_shift)
        if self.config.mask_type == "probability":
            mask = np.load(mask_dir / f"{stem}.npy")
        else:
            mask = load_mask(mask_dir / f"{stem}.png")
            mask = binarize_mask(mask)

        assert image.shape[:2] == mask.shape, (
            f"spatial mismatch at native resolution: image {image.shape[:2]} vs mask {mask.shape}"
        )

        # Augmentation runs at native resolution on BOTH image and mask so they
        # stay aligned and the mask is never upscaled by interpolation.
        if self._aug_transform is not None:
            transformed = self._aug_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # The image alone is resized to image_size; the mask stays native.
        if self._resize_transform is not None:
            image = self._resize_transform(image=image)["image"]

        if self.config.mask_type != "probability":
            validate_binary_mask(mask)

        try:
            import torch as _  # noqa: F401
        except ImportError:
            return to_numpy(image, mask)
        if self.config.mask_type == "probability":
            return to_tensors_soft(image, mask, self.config.normalize)
        return to_tensors(image, mask, self.config.normalize)
