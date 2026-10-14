"""Transformation builders for the GOES-16 dataset.

Requires ``albumentations`` (included in the ``[train]`` extras).
This module is **not** imported by ``deepcontrail.data.__init__`` so that
``import deepcontrail.data`` stays free of the albumentations dependency.

Design note
-----------
The augmentation pipeline (flips, Rot90) runs on **both** image and mask at
native resolution, while the resize to ``image_size`` runs on the **image
only**, afterwards. Keeping the mask at native resolution avoids the
interpolation artifacts an upscaled binary mask would introduce at thin
structures such as contrails. Two separate builders expose those stages:

- :func:`build_augmentation` — flips + Rot90, applied to both image and mask.
- :func:`build_image_resize` — resize only, applied to the image only.
"""

from __future__ import annotations

import albumentations as A

from deepcontrail.data.config import AugmentationConfig  # noqa: TC001


def build_augmentation(config: AugmentationConfig, seed: int) -> A.Compose:
    """Build an albumentations augmentation pipeline from config.

    The base flips/rotate run on **both** the image and the mask so they stay
    aligned. Optionally, photometric transforms (``gaussian_blur``,
    ``brightness_contrast``) are appended — they affect the image only and
    albumentations leaves the mask untouched. Spatial transforms
    (``shift_scale_rotate``) co-transform the mask like the flips.
    Transforms are skipped when their probability is ``0.0`` so disabled
    options add nothing. It intentionally does **not** resize — resizing is a
    separate, image-only stage (see :func:`build_image_resize`).

    Parameters
    ----------
    config
        Probabilities (and transform parameters) for each augmentation.
    seed
        RNG seed passed to ``A.Compose`` for reproducible augmentation.

    Returns
    -------
    A.Compose
        An albumentations ``Compose`` that accepts ``image`` and ``mask``
        keyword arguments and co-transforms them at native resolution.
    """
    transforms: list[A.BasicTransform] = [
        A.HorizontalFlip(p=config.horizontal_flip),
        A.VerticalFlip(p=config.vertical_flip),
        A.RandomRotate90(p=config.random_rotate_90),
    ]
    if config.shift_scale_rotate > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=config.shift_limit,
                scale_limit=config.scale_limit,
                rotate_limit=config.rotate_limit,
                p=config.shift_scale_rotate,
            )
        )
    if config.gaussian_blur > 0:
        transforms.append(A.GaussianBlur(blur_limit=config.blur_limit, p=config.gaussian_blur))
    if config.brightness_contrast > 0:
        transforms.append(A.RandomBrightnessContrast(p=config.brightness_contrast))
    return A.Compose(transforms, seed=seed)  # type: ignore[arg-type]


def build_image_resize(image_size: int) -> A.Resize:
    """Build a resize-only transform for the image.

    Applied to the image only, after augmentation, so the mask keeps its
    native resolution.

    Parameters
    ----------
    image_size
        Target spatial size (height and width).

    Returns
    -------
    A.Resize
        A resize transform that, applied with ``image=`` only, returns the
        resized image and leaves any mask untouched.
    """
    return A.Resize(image_size, image_size)
