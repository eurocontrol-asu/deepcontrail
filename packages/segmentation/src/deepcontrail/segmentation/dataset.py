"""Semantic segmentation dataset over a DeepContrail-format dataset root.

Builds each image from ``spectral_bands/`` with the configured false-colour recipe
(band keys from the root's ``bands.json``) and pairs it with its mask.
``import deepcontrail.segmentation`` requires only numpy + Pillow — torch is
imported lazily inside ``__getitem__`` and is optional.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from deepcontrail.core.contract import read_band_manifest, read_dataset_info, scene_stems
from deepcontrail.core.false_color import check_false_color, load_spectral_false_color
from deepcontrail.core.readers import (
    binarize_mask,
    load_mask,
    shift_image,
    to_numpy,
    validate_binary_mask,
)
from deepcontrail.core.splits import context_frames, partition_stems
from deepcontrail.segmentation.tensors import to_tensors, to_tensors_soft

if TYPE_CHECKING:
    import albumentations as A

    from deepcontrail.segmentation.config import SegmentationDataConfig


class SegmentationDataset:
    """Per-sample semantic segmentation dataset over a DeepContrail-format root.

    Parameters
    ----------
    config
        A ``SegmentationDataConfig`` specifying root path, split, augmentation, and
        normalization parameters.

    Raises
    ------
    DatasetContractError
        If the root's ``bands.json`` or ``dataset_info.json`` is missing or invalid,
        or if ``false_color`` uses a band key the manifest does not declare.
    ValueError
        If any labelled image in the split directory has no matching mask.
    ImportError
        If augmentation or resize is requested but albumentations is not
        installed.
    """

    def __init__(self, config: SegmentationDataConfig) -> None:
        self.config = config
        self._root = Path(config.root)
        self.manifest = read_band_manifest(self._root)
        check_false_color(config.false_color, self.manifest)
        self._warn_on_pixel_shift_mismatch()

        img_subdir = "spectral_bands"
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
        # DataLoader worker the augmentation was last reseeded for (None: main process).
        self._augmentation_worker: int | None = None

    def _reseed_augmentation_in_worker(self) -> None:
        """Give each DataLoader worker its own augmentation stream.

        Every worker starts with a copy of the dataset, generators included, so without this
        all workers would draw the same augmentations. The first item a worker loads reseeds the
        copy from PyTorch's per-worker seed, which is reproducible under a fixed global seed and
        changes each epoch unless the workers persist. The main process keeps the
        configuration's seed.
        """
        try:
            from torch.utils.data import get_worker_info
        except ImportError:  # without torch there are no DataLoader workers
            return
        info = get_worker_info()
        if info is None or info.id == self._augmentation_worker:
            return
        from deepcontrail.segmentation.transforms import seed_augmentation

        seed_augmentation(self._aug_transform, info.seed)  # type: ignore[arg-type]
        self._augmentation_worker = info.id

    def _warn_on_pixel_shift_mismatch(self) -> None:
        """Warn when ``pixel_shift`` differs from the dataset's recommendation."""
        recommended = read_dataset_info(self._root).recommended_pixel_shift
        if recommended is not None and recommended != self.config.pixel_shift:
            msg = (
                f"pixel_shift={self.config.pixel_shift} but {self._root} recommends "
                f"{recommended} (metadata/dataset_info.json): images and masks may be misaligned"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)

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
        return partition_stems(
            all_stems,
            self.config.split,
            seed=self.config.seed,
            val_fraction=self.config.val_fraction,
            fold=self.config.fold,
            num_folds=self.config.num_folds,
        )

    def _scan_stems(self, image_dir: Path, mask_dir: Path) -> list[str]:
        """Scan ``image_dir`` for paired images/masks, return sorted stems.

        Checks that each image has a matching mask file; only the context frames of a dataset
        that declares them (``temporal: true``) may have none. For deeper validation
        (corruption, sizes, mask values), run ``deepcontrail-validate <root> --deep`` on the
        dataset root.
        """
        mask_ext = ".npy" if self.config.mask_type == "probability" else ".png"
        image_stems = scene_stems(image_dir, ".npy")
        context = context_frames(image_stems, temporal=read_dataset_info(self._root).temporal)
        stems: list[str] = []
        missing: list[str] = []

        for stem in image_stems:
            mask_path = mask_dir / f"{stem}{mask_ext}"
            if mask_path.exists():
                stems.append(stem)
            elif stem not in context:
                missing.append(f"{stem}: no mask (expected {mask_path})")

        if missing:
            msg = f"Dataset validation found {len(missing)} missing mask(s):\n"
            msg += "\n".join(f"  - {e}" for e in missing)
            raise ValueError(msg)

        return stems

    def _native_size(self) -> int:
        """Native square resolution of the images (256 for GOES-16)."""
        if not self.stems:
            return self.config.image_size
        arr = np.load(self._image_dir / f"{self.stems[0]}.npy", mmap_mode="r")
        return arr.shape[0]

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
            from deepcontrail.segmentation.transforms import build_augmentation, build_image_resize
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
        image = load_spectral_false_color(
            img_dir / f"{stem}.npy", self.config.false_color, self.manifest
        )
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
            self._reseed_augmentation_in_worker()
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
