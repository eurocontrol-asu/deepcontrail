"""D4 test-time augmentation for segmentation models.

Applies the 8 dihedral group transforms (4 rotations x 2 flips) at
inference time, averages the de-augmented sigmoid probabilities, and
returns the result.  Uses pure PyTorch tensor ops, no external deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    _Transform = Callable[[torch.Tensor], torch.Tensor]


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _rot90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=1, dims=(2, 3))


def _rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=2, dims=(2, 3))


def _rot270(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x, k=3, dims=(2, 3))


def _vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[2])


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[3])


def _transpose(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(2, 3)


def _antitranspose(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x.transpose(2, 3), dims=[2, 3])


D4_TRANSFORMS: list[tuple[_Transform, _Transform]] = [
    (_identity, _identity),
    (_rot90, _rot270),
    (_rot180, _rot180),
    (_rot270, _rot90),
    (_vflip, _vflip),
    (_hflip, _hflip),
    (_transpose, _transpose),
    (_antitranspose, _antitranspose),
]


@torch.no_grad()
def tta_predict(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Run D4 TTA and return averaged sigmoid probabilities.

    Parameters
    ----------
    model
        Segmentation model returning logits of shape ``(B, 1, H, W)``.
    images
        Input batch of shape ``(B, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Averaged sigmoid probabilities of shape ``(B, H, W)``.
    """
    probs_sum: torch.Tensor | None = None

    for augment, deaugment in D4_TRANSFORMS:
        augmented = augment(images)
        logits = model(augmented)
        deaugmented = deaugment(logits)
        probs = torch.sigmoid(deaugmented).squeeze(1)
        if probs_sum is None:
            probs_sum = torch.zeros_like(probs)
        probs_sum += probs

    assert probs_sum is not None
    return probs_sum / len(D4_TRANSFORMS)
