"""Augmentation pipeline builder for the GOES-16 dataset.

Requires ``albumentations`` (included in the ``[train]`` extras).
This module is **not** imported by ``deepcontrail.data.__init__`` so that
``import deepcontrail.data`` stays free of the albumentations dependency.
"""

from __future__ import annotations

import albumentations as A

from deepcontrail.data.config import AugmentationConfig  # noqa: TC001


def build_augmentation(
    config: AugmentationConfig,
    seed: int,
    image_size: int | None = None,
) -> A.Compose:
    """Build an albumentations spatial-augmentation pipeline from config.

    Parameters
    ----------
    config
        Probabilities for each spatial transform.
    seed
        RNG seed passed to ``A.Compose`` for reproducible augmentation.
    image_size
        If provided, prepend an ``A.Resize`` to this size.

    Returns
    -------
    A.Compose
        An albumentations ``Compose`` that accepts ``image`` and ``mask``
        keyword arguments and co-transforms them.
    """
    transforms: list[A.BasicTransform] = []
    if image_size is not None:
        transforms.append(A.Resize(image_size, image_size))
    transforms.extend(
        [
            A.HorizontalFlip(p=config.horizontal_flip),
            A.VerticalFlip(p=config.vertical_flip),
            A.RandomRotate90(p=config.random_rotate_90),
        ]
    )
    return A.Compose(transforms, seed=seed)  # type: ignore[arg-type]


def build_resize(image_size: int) -> A.Compose:
    """Build a resize-only pipeline (no augmentation).

    Used for validation splits when ``image_size`` differs from the native
    image resolution.

    Parameters
    ----------
    image_size
        Target spatial size (height and width).
    """
    return A.Compose([A.Resize(image_size, image_size)])
