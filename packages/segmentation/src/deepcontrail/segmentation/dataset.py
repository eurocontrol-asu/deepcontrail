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

from deepcontrail.core.contract import (
    NEIGHBOURS_PATH,
    DatasetContractError,
    context_ids,
    read_band_manifest,
    read_dataset_info,
    read_neighbours,
    scene_stems,
)
from deepcontrail.core.false_color import check_false_color, false_color_from_spectral
from deepcontrail.core.readers import (
    binarize_mask,
    load_mask,
    shift_image,
    to_numpy,
    validate_binary_mask,
)
from deepcontrail.core.splits import context_frames, partition_stems, read_stem_list
from deepcontrail.segmentation.tensors import to_tensors, to_tensors_soft

if TYPE_CHECKING:
    import albumentations as A

    from deepcontrail.segmentation.config import SegmentationDataConfig


def tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering 0..size with windows of ``tile``.

    0, stride, ..., and a last one at ``size - tile`` so the final window ends at the edge (no
    duplicates).
    """
    origins = list(range(0, size - tile + 1, stride))
    if not origins or origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


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

        # Temporal context: stack the composites of the k frames before and after each scene.
        # Read the neighbours once here; a context_frames > 0 config needs the file to exist.
        self._context_k = config.context_frames
        self._neighbours: dict[str, tuple[str | None, str | None]] = {}
        if self._context_k > 0:
            neighbours = read_neighbours(self._root)
            if neighbours is None:
                msg = (
                    f"{self._root / NEIGHBOURS_PATH}: missing; context_frames={self._context_k} "
                    "needs metadata/neighbours.json"
                )
                raise DatasetContractError(msg)
            self._neighbours = neighbours

        img_subdir = "spectral_bands"
        mask_subdir = "probability_masks" if config.mask_type == "probability" else "semantic_masks"
        ds = "test" if config.split == "test" else "train"
        self._dir_pairs = [
            (self._root / img_subdir / ds, self._root / mask_subdir / ds),
        ]
        self._image_dir, self._mask_dir = self._dir_pairs[0]

        self._stem_dirs: dict[str, tuple[Path, Path]] = {}
        self.stems = self._discover_stems()
        self.scene_size = self._native_size()
        if config.crop_size is not None and config.crop_size > self.scene_size:
            msg = (
                f"crop_size {config.crop_size} is larger than the scenes "
                f"({self.scene_size} px) of {self._root}"
            )
            raise ValueError(msg)
        # The model's native size is the crop when cropping, otherwise the whole scene.
        self.native_size = config.crop_size or self.scene_size
        self._aug_transform, self._resize_transform = self._build_transform()
        # DataLoader worker the augmentation was last reseeded for (None: main process).
        self._augmentation_worker: int | None = None
        # Tiles covering each scene, (stem index, row0, col0). Non-overlapping (stride == crop_size)
        # for validation/test with crop_size; overlapping at ``train_tile_stride`` for the train
        # split when that is set (the epoch is every tile of every scene). Empty otherwise.
        self._tiles: list[tuple[int, int, int]] = []
        if config.crop_size is not None and (
            config.split != "train" or config.train_tile_stride is not None
        ):
            # Non-train splits always tile non-overlapping (stride == crop_size); only the train
            # split with train_tile_stride overlaps at that stride.
            stride = config.crop_size
            if config.split == "train" and config.train_tile_stride is not None:
                stride = config.train_tile_stride
            origins = tile_origins(self.scene_size, config.crop_size, stride)
            for stem_index, _stem in enumerate(self.stems):
                for row0 in origins:
                    for col0 in origins:
                        self._tiles.append((stem_index, row0, col0))
        # Random-crop stream for training with crop_size (reseeded per DataLoader worker).
        self._crop_rng = np.random.default_rng(config.seed)

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

        # Without augmentation there is nothing to seed, but the random-crop stream still needs
        # its own per-worker generator so workers do not draw the same crops.
        if self._aug_transform is not None:
            seed_augmentation(self._aug_transform, info.seed)  # type: ignore[arg-type]
        self._crop_rng = np.random.default_rng(info.seed)
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
        val_ids_path = self.config.val_ids_path()
        train_ids_path = self.config.train_ids_path()
        return partition_stems(
            all_stems,
            self.config.split,
            seed=self.config.seed,
            val_fraction=self.config.val_fraction,
            fold=self.config.fold,
            num_folds=self.config.num_folds,
            val_stems=read_stem_list(val_ids_path) if val_ids_path is not None else None,
            train_stems=read_stem_list(train_ids_path) if train_ids_path is not None else None,
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

    def _composite_for(self, stem: str, img_dir: Path, r0: int, c0: int, size: int) -> np.ndarray:
        """Build the 3-channel composite of one scene's ``[r0:r0+size, c0:c0+size]`` window.

        The bands are memory-mapped and sliced to the window *before* the composite is built, so
        a large frame (e.g. 1024 px) is never fully read into memory. On a crop the pixel shift's
        edge padding falls at the crop edge rather than the scene edge. Neighbour frames are read
        from the same split directory as the centre scene, so this takes only the image directory.
        """
        bands = np.load(img_dir / f"{stem}.npy", mmap_mode="r")[r0 : r0 + size, c0 : c0 + size]
        bands = np.asarray(bands, dtype=np.float32)
        image = false_color_from_spectral(bands, self.config.false_color, self.manifest)
        if self.config.pixel_shift != 0.0:
            image = shift_image(image, self.config.pixel_shift)
        return image

    def _mask_for(self, stem: str, mask_dir: Path, r0: int, c0: int, size: int) -> np.ndarray:
        """Read one scene's ``[r0:r0+size, c0:c0+size]`` mask window.

        Binary uint8 for ``semantic_masks``, float32 probabilities for ``probability_masks``.
        """
        if self.config.mask_type == "probability":
            mask_full = np.load(mask_dir / f"{stem}.npy", mmap_mode="r")
            return np.asarray(mask_full[r0 : r0 + size, c0 : c0 + size], dtype=np.float32)
        return binarize_mask(load_mask(mask_dir / f"{stem}.png"))[r0 : r0 + size, c0 : c0 + size]

    def _load_window(
        self, stem: str, img_dir: Path, mask_dir: Path, r0: int, c0: int, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read one square window of a scene and build its ``(image, mask)`` pair.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(image, mask)``: a float32 ``(size, size, 3)`` composite in ``[0, 1]`` and its
            ``(size, size)`` mask (binary uint8, or float32 probabilities).
        """
        return (
            self._composite_for(stem, img_dir, r0, c0, size),
            self._mask_for(stem, mask_dir, r0, c0, size),
        )

    def _load_pair(
        self, stem: str, img_dir: Path, mask_dir: Path, r0: int, c0: int, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(image, mask)`` of one item: a single composite, or a temporal stack.

        With ``context_frames == 0`` the image is the centre scene's 3-channel composite. With
        ``context_frames == k > 0`` it stacks the composites of the ``2k+1`` ids of
        :func:`deepcontrail.core.contract.context_ids` (same window and pixel shift for every
        frame) along the channel axis into ``(size, size, 3*(2k+1))``. The mask is always the
        centre scene's, at the same window.
        """
        if self._context_k == 0:
            return self._load_window(stem, img_dir, mask_dir, r0, c0, size)
        ids = context_ids(stem, self._neighbours, self._context_k)
        image = np.concatenate(
            [self._composite_for(cid, img_dir, r0, c0, size) for cid in ids], axis=-1
        )
        mask = self._mask_for(stem, mask_dir, r0, c0, size)
        return image, mask

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
        """Return the number of items in this split.

        With ``crop_size`` on a non-train split, each scene contributes one item per
        covering tile, so the count is the number of tiles. With ``crop_size`` and
        ``train_tile_stride`` on the train split, each scene contributes one item per
        overlapping tile, again the number of tiles. With ``crop_size`` on the train split
        without ``train_tile_stride``, each scene contributes ``crops_per_scene`` independent
        random crops, so the count is scenes x crops_per_scene. Otherwise it is the number of
        stems (training without crop_size, and non-cropped splits, included).
        """
        if self._tiles:
            return len(self._tiles)
        if self.config.crop_size is not None:
            return len(self.stems) * self.config.crops_per_scene
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
        if self._tiles:
            # Validation/test with crop_size: the non-overlapping tiles covering each scene.
            # Train with train_tile_stride: the overlapping tiles of each scene (augmentation
            # still applies below, as it does for every item).
            stem_index, r0, c0 = self._tiles[index]
            stem = self.stems[stem_index]
            img_dir, mask_dir = self._stem_dirs.get(stem, (self._image_dir, self._mask_dir))
            image, mask = self._load_pair(stem, img_dir, mask_dir, r0, c0, self.native_size)
        elif self.config.crop_size is not None:
            # Training with crop_size: crops_per_scene independent random crops per scene.
            # Index maps to a scene by // crops_per_scene; the remainder is unused — every
            # item draws a fresh random origin from self._crop_rng, so the crops differ.
            crops = self.config.crops_per_scene
            stem = self.stems[index // crops]
            img_dir, mask_dir = self._stem_dirs.get(stem, (self._image_dir, self._mask_dir))
            self._reseed_augmentation_in_worker()
            r0, c0 = self._crop_rng.integers(0, self.scene_size - self.native_size + 1, size=2)
            image, mask = self._load_pair(stem, img_dir, mask_dir, r0, c0, self.native_size)
        else:
            # Whole scene (no crop_size): the window is the scene itself.
            stem = self.stems[index]
            img_dir, mask_dir = self._stem_dirs.get(stem, (self._image_dir, self._mask_dir))
            image, mask = self._load_pair(stem, img_dir, mask_dir, 0, 0, self.scene_size)

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
