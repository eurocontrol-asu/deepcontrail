"""Transformation builders for the GOES-16 dataset.

Requires ``albumentations`` (included in the ``[train]`` extras).
This module is **not** imported by ``deepcontrail.segmentation.__init__`` so that
``import deepcontrail.segmentation`` stays free of the albumentations dependency.

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

import os
import random

import numpy as np

# albumentations checks PyPI for a newer version when it is imported; a library must not make
# network calls on its own. A value the user set explicitly is kept.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A

from deepcontrail.segmentation.config import AugmentationConfig  # noqa: TC001


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
        Seed of the random generators the transforms share (see :func:`seed_augmentation`).

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
            # What the deprecated A.ShiftScaleRotate(shift_limit, scale_limit, rotate_limit)
            # builds (it subclasses Affine; Affine's other defaults are the same): no shear,
            # linear image / nearest mask interpolation, a constant 0 border, same draws.
            A.Affine(
                scale=(1 - config.scale_limit, 1 + config.scale_limit),
                translate_percent={
                    "x": (-config.shift_limit, config.shift_limit),
                    "y": (-config.shift_limit, config.shift_limit),
                },
                rotate=(-config.rotate_limit, config.rotate_limit),
                p=config.shift_scale_rotate,
            )
        )
    if config.gaussian_blur > 0:
        transforms.append(A.GaussianBlur(blur_limit=config.blur_limit, p=config.gaussian_blur))
    if config.brightness_contrast > 0:
        transforms.append(A.RandomBrightnessContrast(p=config.brightness_contrast))
    compose = A.Compose(transforms)  # type: ignore[arg-type]
    seed_augmentation(compose, seed)
    return compose


def seed_augmentation(compose: A.Compose, seed: int) -> None:
    """Seed an augmentation pipeline with one random stream shared by all its transforms.

    ``A.Compose(..., seed=seed)`` gives every transform its own generator with the *same* seed,
    so transforms with the same probability fire together: both flips always coincide (a 180°
    rotation) and mirrored scenes never appear. Sharing one numpy and one Python generator makes
    the transforms draw independently while staying reproducible.

    Parameters
    ----------
    compose
        Pipeline to seed, in place.
    seed
        Seed of the shared generators; any non-negative integer.
    """
    compose.set_random_state(np.random.default_rng(seed), random.Random(seed))


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
