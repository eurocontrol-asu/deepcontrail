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
    """Build an albumentations spatial-augmentation pipeline from config.

    This pipeline flips and rotates **both** the image and the mask so they
    stay aligned. It intentionally does **not** resize — resizing is a
    separate, image-only stage (see :func:`build_image_resize`).

    Parameters
    ----------
    config
        Probabilities for each spatial transform.
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
